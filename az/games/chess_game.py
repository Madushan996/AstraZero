"""Chess adapter over python-chess, with AlphaZero's plane encoding.

Observation layout, per the paper, always from the side-to-move's perspective:

    T history steps x 14 planes each
        12  piece type x colour  (own P N B R Q K, then opponent's)
         2  repetition flags     (position seen >=2 times, >=3 times)
     7 constant planes
         1  side to move
         1  move count      (normalised)
         2  own castling rights   (kingside, queenside)
         2  opponent castling rights
         1  halfmove clock  (normalised)

With the paper's T=8 that is 119 planes. T is configurable: `history_length=1` gives
21 planes and is meaningfully faster, which is a reasonable trade at hobby scale --
history mainly helps with repetition-aware play.

Every plane is kept in [0, 1] so the replay buffer can store observations as uint8.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import chess
import numpy as np

from az.core.game import Game
from az.games.chess_encoding import ACTION_SIZE, index_to_move, legal_move_indices

PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
PIECE_PLANES = 12
REPETITION_PLANES = 2
STEP_PLANES = PIECE_PLANES + REPETITION_PLANES
CONSTANT_PLANES = 7

MAX_MOVE_COUNT_NORM = 100.0
MAX_HALFMOVE_NORM = 100.0

# Standard piece values, used ONLY to adjudicate shuffled draws (see `terminal_value`).
# They never touch the network's evaluation or the search -- the engine still learns
# what positions are worth entirely from self-play.
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


@dataclass
class ChessState:
    board: chess.Board
    # history[0] is the current position; later entries are older.
    history: deque = field(default_factory=deque)
    repetitions: dict = field(default_factory=dict)
    ply: int = 0
    _legal: Optional[dict] = field(default=None, repr=False, compare=False)
    _terminal: Optional[float] = field(default=None, repr=False, compare=False)
    _terminal_known: bool = field(default=False, repr=False, compare=False)


class ChessGame(Game[ChessState]):
    name = "chess"

    def __init__(
        self,
        history_length: int = 8,
        max_moves: int = 512,
        adjudicate_draw_at: int = 100,
        adjudicate_material_at: int = 0,
    ) -> None:
        """`max_moves` caps game length so self-play cannot stall on shuffling; the
        cap is scored as a draw. `adjudicate_draw_at` is the halfmove-clock threshold
        (100 halfmoves = the 50-move rule).

        `adjudicate_material_at` (0 = off) scores a shuffled draw as a win for whoever
        is ahead by at least that much material. It exists because of a measured
        pathology: in one 1,500-game sample, 32% of ALL games ended as draws with a
        piece or more on the board -- the engine wins material, cannot convert, and
        shuffles to the 50-move rule. Those positions get labelled 0.0, which teaches
        the value head that a queen is worth nothing. That is exactly the blindness
        seen in play, where it evaluated a free queen capture at 0.00.

        This is the one place human chess knowledge enters the system, and it enters
        only as a training LABEL -- never into the network, the encoding, or the search.
        Set it to 0 for a fully knowledge-free run, and expect to need far more games.
        """
        self.history_length = max(1, history_length)
        self.max_moves = max_moves
        self.adjudicate_draw_at = adjudicate_draw_at
        self.adjudicate_material_at = adjudicate_material_at

    # --- static description -------------------------------------------------

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (STEP_PLANES * self.history_length + CONSTANT_PLANES, 8, 8)

    # --- dynamics -----------------------------------------------------------

    def initial_state(self, fen: Optional[str] = None) -> ChessState:
        board = chess.Board(fen) if fen else chess.Board()
        state = ChessState(board=board, history=deque(maxlen=self.history_length))
        key = _position_key(board)
        state.repetitions = {key: 1}
        state.history.appendleft(_snapshot(board, 1))
        return state

    def copy(self, state: ChessState) -> ChessState:
        # stack=False skips copying the move stack, which is a large speedup inside
        # MCTS. We track repetition ourselves, so the stack is not needed.
        clone = ChessState(
            board=state.board.copy(stack=False),
            history=deque(state.history, maxlen=self.history_length),
            repetitions=state.repetitions,  # replaced, never mutated, on each move
            ply=state.ply,
        )
        clone._legal = state._legal
        clone._terminal = state._terminal
        clone._terminal_known = state._terminal_known
        return clone

    def next_state(self, state: ChessState, action: int) -> ChessState:
        move = self._legal_map(state).get(action)
        if move is None:
            raise ValueError(f"action {action} is not legal in this position")

        board = state.board.copy(stack=False)
        board.push(move)

        key = _position_key(board)
        # Copy-on-write: the parent's dict must stay untouched for sibling branches.
        repetitions = dict(state.repetitions)
        count = repetitions.get(key, 0) + 1
        repetitions[key] = count

        history = deque(state.history, maxlen=self.history_length)
        history.appendleft(_snapshot(board, count))

        return ChessState(
            board=board,
            history=history,
            repetitions=repetitions,
            ply=state.ply + 1,
        )

    def legal_actions(self, state: ChessState) -> np.ndarray:
        mask = np.zeros(ACTION_SIZE, dtype=bool)
        if self.terminal_value(state) is not None:
            return mask
        indices = list(self._legal_map(state))
        if indices:
            mask[indices] = True
        return mask

    def terminal_value(self, state: ChessState) -> Optional[float]:
        if state._terminal_known:
            return state._terminal

        board = state.board
        value: Optional[float] = None

        if board.is_checkmate():
            value = -1.0  # the side to move has been mated
        elif board.is_stalemate() or board.is_insufficient_material():
            # Genuine draws whatever the material: stalemate is stalemate, and
            # "insufficient material" means there is nothing to be ahead by.
            value = 0.0
        elif (
            board.halfmove_clock >= self.adjudicate_draw_at
            or state.repetitions.get(_position_key(board), 0) >= 3
            or state.ply >= self.max_moves
        ):
            # A shuffled draw. If one side is clearly ahead, calling this 0.0 actively
            # teaches the value head that material is worthless.
            value = self._adjudicate(board)

        state._terminal = value
        state._terminal_known = True
        return value

    def _adjudicate(self, board: chess.Board) -> float:
        """Score a shuffled draw, from the perspective of the player to move."""
        if self.adjudicate_material_at <= 0:
            return 0.0

        balance = 0
        for _, piece in board.piece_map().items():
            value = PIECE_VALUES.get(piece.piece_type, 0)
            balance += value if piece.color == board.turn else -value

        if balance >= self.adjudicate_material_at:
            return 1.0
        if balance <= -self.adjudicate_material_at:
            return -1.0
        return 0.0

    def encode(self, state: ChessState) -> np.ndarray:
        channels, _, _ = self.observation_shape
        planes = np.zeros((channels, 8, 8), dtype=np.float32)
        board = state.board
        flip = board.turn == chess.BLACK

        for step in range(self.history_length):
            if step >= len(state.history):
                break  # positions before the game started stay all-zero
            pieces, repetition_count = state.history[step]
            base = step * STEP_PLANES

            oriented = pieces[:, ::-1, :] if flip else pieces
            if flip:
                # Swap "white's pieces" and "black's pieces" so index 0-5 is always
                # the side to move.
                planes[base : base + 6] = oriented[6:12]
                planes[base + 6 : base + 12] = oriented[0:6]
            else:
                planes[base : base + 12] = oriented

            if repetition_count >= 2:
                planes[base + 12] = 1.0
            if repetition_count >= 3:
                planes[base + 13] = 1.0

        offset = STEP_PLANES * self.history_length
        us, them = board.turn, not board.turn

        planes[offset + 0] = 1.0 if flip else 0.0
        planes[offset + 1] = min(state.ply / MAX_MOVE_COUNT_NORM, 1.0)
        planes[offset + 2] = float(board.has_kingside_castling_rights(us))
        planes[offset + 3] = float(board.has_queenside_castling_rights(us))
        planes[offset + 4] = float(board.has_kingside_castling_rights(them))
        planes[offset + 5] = float(board.has_queenside_castling_rights(them))
        planes[offset + 6] = min(board.halfmove_clock / MAX_HALFMOVE_NORM, 1.0)

        return planes

    # --- text helpers -------------------------------------------------------

    def action_to_string(self, state: ChessState, action: int) -> str:
        move = self._legal_map(state).get(action)
        return move.uci() if move else f"<illegal:{action}>"

    def action_to_san(self, state: ChessState, action: int) -> str:
        move = self._legal_map(state).get(action)
        return state.board.san(move) if move else f"<illegal:{action}>"

    def string_to_action(self, state: ChessState, text: str) -> Optional[int]:
        board = state.board
        move: Optional[chess.Move] = None
        try:
            move = chess.Move.from_uci(text)
        except ValueError:
            try:
                move = board.parse_san(text)
            except ValueError:
                return None
        if move not in board.legal_moves:
            return None
        for index, candidate in self._legal_map(state).items():
            if candidate == move:
                return index
        return None

    def action_to_move(self, state: ChessState, action: int) -> Optional[chess.Move]:
        move = self._legal_map(state).get(action)
        if move is not None:
            return move
        return index_to_move(action, state.board)

    def render(self, state: ChessState) -> str:
        return str(state.board)

    def result_string(self, state: ChessState) -> str:
        value = self.terminal_value(state)
        if value is None:
            return "*"
        if value == 0.0:
            return "1/2-1/2"
        # The side to move lost.
        return "0-1" if state.board.turn == chess.WHITE else "1-0"

    # --- internals ----------------------------------------------------------

    def _legal_map(self, state: ChessState) -> dict:
        if state._legal is None:
            state._legal = legal_move_indices(state.board)
        return state._legal


def _snapshot(board: chess.Board, repetition_count: int) -> tuple[np.ndarray, int]:
    """Absolute-orientation piece planes: index p + 6*is_black, [rank, file]."""
    planes = np.zeros((PIECE_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = PIECE_TYPES.index(piece.piece_type)
        if piece.color == chess.BLACK:
            plane += 6
        planes[plane, chess.square_rank(square), chess.square_file(square)] = 1.0
    return planes, repetition_count


def _position_key(board: chess.Board) -> tuple:
    """Repetition identity: pieces + turn + castling + en-passant file."""
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.turn,
        board.castling_rights,
        board.ep_square,
    )
