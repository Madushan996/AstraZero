"""Game-agnostic interface that MCTS and the trainer are written against.

Everything below MCTS knows nothing about chess. Implement this interface and the
whole pipeline (self-play, training, checkpointing) works unchanged. Connect4 exists
to validate that pipeline cheaply; chess is the real target.

Conventions that the rest of the codebase depends on -- get these wrong and training
silently fails rather than crashing:

  * Values are ALWAYS from the perspective of the player to move at that state.
    A value of +1 means "the side to move is winning".
  * Encodings are ALWAYS from the perspective of the player to move (i.e. the board
    is flipped for black). The network therefore never needs a "whose turn" input
    beyond the constant planes a game chooses to add.
  * Actions are integers in [0, action_size).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Sequence, TypeVar

import numpy as np

State = TypeVar("State")


class Game(ABC, Generic[State]):
    """A two-player, zero-sum, perfect-information game."""

    name: str = "game"

    # --- static description -------------------------------------------------

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Total number of distinct action indices (including illegal ones)."""

    @property
    @abstractmethod
    def observation_shape(self) -> tuple[int, int, int]:
        """(channels, height, width) of the encoded observation."""

    # --- dynamics -----------------------------------------------------------

    @abstractmethod
    def initial_state(self) -> State:
        """A fresh starting position."""

    @abstractmethod
    def copy(self, state: State) -> State:
        """Deep enough copy that mutating the result cannot affect the original."""

    @abstractmethod
    def next_state(self, state: State, action: int) -> State:
        """Apply `action`. Must not mutate `state`."""

    @abstractmethod
    def legal_actions(self, state: State) -> np.ndarray:
        """Boolean mask of shape (action_size,)."""

    @abstractmethod
    def terminal_value(self, state: State) -> Optional[float]:
        """None if the game is ongoing, else the result in [-1, 1] from the
        perspective of the player to move at `state`.

        Note the subtlety: at a checkmated position the side to move has lost, so
        this returns -1. Draws return 0.0.
        """

    @abstractmethod
    def encode(self, state: State) -> np.ndarray:
        """Observation of shape `observation_shape`, float32, from the perspective
        of the player to move."""

    # --- optional niceties --------------------------------------------------

    def action_to_string(self, state: State, action: int) -> str:
        return str(action)

    def string_to_action(self, state: State, text: str) -> Optional[int]:
        try:
            value = int(text)
        except ValueError:
            return None
        return value if 0 <= value < self.action_size else None

    def render(self, state: State) -> str:
        return repr(state)

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> Sequence[tuple[np.ndarray, np.ndarray]]:
        """Data augmentation: equivalent (observation, policy) pairs.

        Default is identity only. Connect4 adds a horizontal mirror. Chess has no
        usable symmetry (castling and pawn direction break both axes), which is
        precisely why AlphaZero used none for chess.
        """
        return [(encoded, policy)]

    # --- convenience --------------------------------------------------------

    def is_terminal(self, state: State) -> bool:
        return self.terminal_value(state) is not None

    def describe(self) -> dict[str, Any]:
        c, h, w = self.observation_shape
        return {
            "name": self.name,
            "action_size": self.action_size,
            "observation_shape": [c, h, w],
        }


def registry() -> dict[str, Any]:
    """Lazily-imported map of game name -> constructor.

    Imported lazily so that a machine without `python-chess` can still run the
    Connect4 validation path.
    """
    from az.games.connect4 import Connect4

    games: dict[str, Any] = {"connect4": Connect4}

    try:
        from az.games.chess_game import ChessGame

        games["chess"] = ChessGame
    except ImportError:  # pragma: no cover - only when python-chess is absent
        pass

    return games


def make_game(name: str, **kwargs: Any) -> Game:
    games = registry()
    if name not in games:
        raise KeyError(f"unknown game {name!r}; available: {sorted(games)}")
    return games[name](**kwargs)
