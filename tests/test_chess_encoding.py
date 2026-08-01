"""The move encoding is the highest-risk part of a chess AlphaZero clone: a bug here
does not crash, it just silently caps the network's strength. These tests exhaustively
round-trip every legal move across random games, plus the awkward cases by hand.
"""

from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from az.games.chess_encoding import (
    ACTION_SIZE,
    index_to_move,
    legal_move_indices,
    move_to_index,
)
from az.games.chess_game import ChessGame


def test_action_size():
    assert ACTION_SIZE == 4672


def test_roundtrip_on_random_games():
    """Every legal move in every position of 40 random games must survive
    move -> index -> move unchanged."""
    rng = random.Random(1234)
    checked = 0

    for _ in range(40):
        board = chess.Board()
        while not board.is_game_over(claim_draw=False) and board.fullmove_number < 120:
            moves = list(board.legal_moves)
            mapping = legal_move_indices(board)

            # No two distinct legal moves may share an index.
            assert len(mapping) == len(moves), (
                f"index collision in {board.fen()}: "
                f"{len(moves)} moves -> {len(mapping)} indices"
            )

            for index, move in mapping.items():
                assert 0 <= index < ACTION_SIZE
                assert index_to_move(index, board) == move
                assert move_to_index(move, board.turn) == index
                checked += 1

            board.push(rng.choice(moves))

    assert checked > 5000, f"only exercised {checked} moves"


@pytest.mark.parametrize(
    "fen,uci",
    [
        # White underpromotions: push, and both capture directions.
        ("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8n"),
        ("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8r"),
        ("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8q"),
        ("1n2k3/2P5/8/8/8/8/8/4K3 w - - 0 1", "c7b8b"),
        # Black underpromotions must mirror correctly.
        ("4k3/8/8/8/8/8/6p1/4K2N b - - 0 1", "g2g1n"),
        ("4k3/8/8/8/8/8/6p1/4K2N b - - 0 1", "g2h1r"),
        # Castling, both colours and sides.
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
        # En passant, both colours.
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", "d4e3"),
        # Knight moves near the edge.
        ("4k3/8/8/8/8/8/8/N3K3 w - - 0 1", "a1b3"),
    ],
)
def test_special_moves_roundtrip(fen, uci):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{uci} is not legal in {fen}"

    index = move_to_index(move, board.turn)
    assert index_to_move(index, board) == move


def test_black_and_white_share_encoding():
    """The same position with colours reversed must produce the same indices --
    that is the property that lets one network play both sides."""
    white = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    black = chess.Board("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1")

    white_indices = set(legal_move_indices(white))
    black_indices = set(legal_move_indices(black))
    assert white_indices == black_indices


def test_encode_shape_and_range():
    game = ChessGame(history_length=8)
    state = game.initial_state()
    encoded = game.encode(state)

    assert encoded.shape == game.observation_shape == (119, 8, 8)
    assert encoded.dtype == np.float32
    # Everything must stay in [0, 1] so the replay buffer's uint8 storage is lossless
    # for binary planes and merely quantised for scalar ones.
    assert encoded.min() >= 0.0 and encoded.max() <= 1.0


def test_encode_is_side_to_move_relative():
    """A position and its colour-swapped mirror must encode identically."""
    game = ChessGame(history_length=1)
    white = game.initial_state("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    black = game.initial_state("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1")

    a, b = game.encode(white), game.encode(black)
    # Piece planes must match exactly; the side-to-move constant plane will differ.
    assert np.array_equal(a[:12], b[:12])


def test_history_planes_shrink_with_history_length():
    assert ChessGame(history_length=1).observation_shape == (21, 8, 8)
    assert ChessGame(history_length=4).observation_shape == (63, 8, 8)
    assert ChessGame(history_length=8).observation_shape == (119, 8, 8)
