"""Connect 4 -- the validation game.

This exists so you can prove the pipeline learns before spending money on chess.
A correct implementation goes from random to "never loses a forced win" in well under
an hour on a laptop CPU. If that does not happen, the bug is in MCTS or the trainer,
not in your chess encoding -- which is exactly the ambiguity this game removes.

The board is always stored from the perspective of the player to move: +1 is "mine",
-1 is "theirs". Flipping on every move means `encode` needs no turn indicator and the
sign conventions in game.py hold trivially.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from az.core.game import Game

ROWS = 6
COLS = 7
CONNECT = 4


@dataclass
class Connect4State:
    board: np.ndarray  # (ROWS, COLS) int8, +1 = player to move
    move_count: int = 0
    terminal: Optional[float] = None  # cached result for the player to move


class Connect4(Game[Connect4State]):
    name = "connect4"

    @property
    def action_size(self) -> int:
        return COLS

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (3, ROWS, COLS)

    def initial_state(self) -> Connect4State:
        return Connect4State(board=np.zeros((ROWS, COLS), dtype=np.int8))

    def copy(self, state: Connect4State) -> Connect4State:
        return Connect4State(
            board=state.board.copy(),
            move_count=state.move_count,
            terminal=state.terminal,
        )

    def next_state(self, state: Connect4State, action: int) -> Connect4State:
        board = state.board.copy()
        row = _drop_row(board, action)
        if row is None:
            raise ValueError(f"column {action} is full")
        board[row, action] = 1

        won = _creates_line(board, row, action)
        board = -board  # hand the board to the opponent, still as "+1 = mine"
        move_count = state.move_count + 1

        if won:
            terminal = -1.0  # the player now to move has just lost
        elif move_count >= ROWS * COLS:
            terminal = 0.0
        else:
            terminal = None

        return Connect4State(board=board, move_count=move_count, terminal=terminal)

    def legal_actions(self, state: Connect4State) -> np.ndarray:
        if state.terminal is not None:
            return np.zeros(COLS, dtype=bool)
        return state.board[0] == 0

    def terminal_value(self, state: Connect4State) -> Optional[float]:
        return state.terminal

    def encode(self, state: Connect4State) -> np.ndarray:
        planes = np.zeros((3, ROWS, COLS), dtype=np.float32)
        planes[0] = (state.board == 1).astype(np.float32)
        planes[1] = (state.board == -1).astype(np.float32)
        planes[2] = 1.0  # constant plane: gives conv layers a board-edge reference
        return planes

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> Sequence[tuple[np.ndarray, np.ndarray]]:
        # Connect 4 is left-right symmetric, so every position is worth two samples.
        return [
            (encoded, policy),
            (encoded[:, :, ::-1].copy(), policy[::-1].copy()),
        ]

    def action_to_string(self, state: Connect4State, action: int) -> str:
        return str(action)

    def render(self, state: Connect4State) -> str:
        # Render in absolute terms so the output is readable regardless of whose turn
        # it is: X always means the player who moved first.
        perspective = 1 if state.move_count % 2 == 0 else -1
        absolute = state.board * perspective
        symbols = {1: "X", -1: "O", 0: "."}
        rows = [" ".join(symbols[int(v)] for v in row) for row in absolute]
        return "\n".join(rows) + "\n" + " ".join(str(c) for c in range(COLS))


def _drop_row(board: np.ndarray, column: int) -> Optional[int]:
    if not 0 <= column < COLS:
        return None
    empties = np.flatnonzero(board[:, column] == 0)
    return int(empties[-1]) if empties.size else None


def _creates_line(board: np.ndarray, row: int, col: int) -> bool:
    """Did the +1 piece just placed at (row, col) complete a line of 4?"""
    for d_row, d_col in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            r, c = row + sign * d_row, col + sign * d_col
            while 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == 1:
                count += 1
                r += sign * d_row
                c += sign * d_col
        if count >= CONNECT:
            return True
    return False
