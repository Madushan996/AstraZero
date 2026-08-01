"""AlphaZero's 8x8x73 = 4672 move encoding.

Every move is described as "pick a from-square, pick one of 73 move types":

    planes  0..55   queen-like moves: 8 directions x 7 distances
    planes 56..63   knight moves: 8 fixed offsets
    planes 64..72   underpromotions: 3 pawn directions x {knight, bishop, rook}

Queen promotions are NOT in the underpromotion planes -- they ride along as ordinary
forward/diagonal moves, which is why 9 planes cover promotion rather than 12.

Everything is expressed from the perspective of the side to move: for black, squares
are mirrored vertically first, so "forward" is always increasing rank. That is what lets
one network play both colours.

The flat index is `plane * 64 + from_square`, matching an (73, 8, 8) plane stack.
"""

from __future__ import annotations

from typing import Optional

import chess

BOARD_SQUARES = 64
QUEEN_PLANES = 56
KNIGHT_PLANES = 8
UNDERPROMOTION_PLANES = 9
TOTAL_PLANES = QUEEN_PLANES + KNIGHT_PLANES + UNDERPROMOTION_PLANES  # 73
ACTION_SIZE = TOTAL_PLANES * BOARD_SQUARES  # 4672

# (file delta, rank delta), in the paper's N, NE, E, SE, S, SW, W, NW order.
QUEEN_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)

KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

# Underpromotion piece order within each direction.
UNDERPROMOTION_PIECES: tuple[int, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

_DIRECTION_INDEX = {d: i for i, d in enumerate(QUEEN_DIRECTIONS)}
_KNIGHT_INDEX = {d: i for i, d in enumerate(KNIGHT_DELTAS)}
_UNDERPROMOTION_INDEX = {p: i for i, p in enumerate(UNDERPROMOTION_PIECES)}


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def orient(square: int, color: bool) -> int:
    """Map an absolute square into the side-to-move's frame."""
    return square if color == chess.WHITE else chess.square_mirror(square)


def move_to_index(move: chess.Move, color: bool) -> int:
    """Encode a move played by `color` into [0, 4672). Raises on impossible geometry."""
    from_square = orient(move.from_square, color)
    to_square = orient(move.to_square, color)

    from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
    to_file, to_rank = chess.square_file(to_square), chess.square_rank(to_square)
    delta_file, delta_rank = to_file - from_file, to_rank - from_rank

    if move.promotion is not None and move.promotion != chess.QUEEN:
        if move.promotion not in _UNDERPROMOTION_INDEX:
            raise ValueError(f"unsupported promotion piece: {move.promotion}")
        # After orientation a promoting pawn always advances one rank.
        direction = delta_file + 1  # -1, 0, +1  ->  0, 1, 2
        if not 0 <= direction <= 2:
            raise ValueError(f"bad promotion geometry for {move}")
        plane = (
            QUEEN_PLANES
            + KNIGHT_PLANES
            + direction * len(UNDERPROMOTION_PIECES)
            + _UNDERPROMOTION_INDEX[move.promotion]
        )
        return plane * BOARD_SQUARES + from_square

    knight_index = _KNIGHT_INDEX.get((delta_file, delta_rank))
    if knight_index is not None:
        plane = QUEEN_PLANES + knight_index
        return plane * BOARD_SQUARES + from_square

    direction = (_sign(delta_file), _sign(delta_rank))
    direction_index = _DIRECTION_INDEX.get(direction)
    distance = max(abs(delta_file), abs(delta_rank))
    if direction_index is None or not 1 <= distance <= 7:
        raise ValueError(f"cannot encode move {move}")
    # Reject non-straight, non-diagonal geometry that slipped past the knight check.
    if abs(delta_file) not in (0, distance) or abs(delta_rank) not in (0, distance):
        raise ValueError(f"cannot encode move {move}")

    plane = direction_index * 7 + (distance - 1)
    return plane * BOARD_SQUARES + from_square


def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """Decode an index back into a move, using `board` to resolve the two ambiguities
    the encoding cannot carry on its own: whether a forward pawn move promotes to a
    queen, and whose turn it is. Returns None if the result is not legal."""
    if not 0 <= index < ACTION_SIZE:
        return None

    plane, from_square = divmod(index, BOARD_SQUARES)
    color = board.turn

    if plane < QUEEN_PLANES:
        direction_index, distance = divmod(plane, 7)
        distance += 1
        delta_file, delta_rank = QUEEN_DIRECTIONS[direction_index]
        delta_file *= distance
        delta_rank *= distance
        promotion = None
    elif plane < QUEEN_PLANES + KNIGHT_PLANES:
        delta_file, delta_rank = KNIGHT_DELTAS[plane - QUEEN_PLANES]
        promotion = None
    else:
        offset = plane - QUEEN_PLANES - KNIGHT_PLANES
        direction, piece_index = divmod(offset, len(UNDERPROMOTION_PIECES))
        delta_file, delta_rank = direction - 1, 1
        promotion = UNDERPROMOTION_PIECES[piece_index]

    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file, to_rank = from_file + delta_file, from_rank + delta_rank
    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        return None

    to_square = chess.square(to_file, to_rank)

    # Back into absolute coordinates.
    absolute_from = orient(from_square, color)
    absolute_to = orient(to_square, color)

    move = chess.Move(absolute_from, absolute_to, promotion=promotion)
    if move in board.legal_moves:
        return move

    # A pawn reaching the last rank must promote; the queen case is encoded as a plain
    # forward/diagonal move, so retry with a queen before giving up.
    if promotion is None:
        queen_move = chess.Move(absolute_from, absolute_to, promotion=chess.QUEEN)
        if queen_move in board.legal_moves:
            return queen_move

    return None


def legal_move_indices(board: chess.Board) -> dict[int, chess.Move]:
    """Map every legal move in `board` to its action index."""
    mapping: dict[int, chess.Move] = {}
    for move in board.legal_moves:
        mapping[move_to_index(move, board.turn)] = move
    return mapping
