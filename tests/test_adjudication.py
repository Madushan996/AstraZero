"""Material adjudication of shuffled draws, and retroactive relabelling.

Measured motivation: in a 1,500-game sample of real self-play, 32% of ALL games ended
as draws with a piece or more on the board. Labelling those 0.0 teaches the value head
that material is worthless, which is exactly what showed up in play -- the engine
evaluated a free queen capture at 0.00.
"""

from __future__ import annotations

import chess
import pytest

from az.core.relabel import relabel_directory, relabel_record
from az.core.replay import GameRecord, ReplayBuffer, read_shard
from az.games.chess_game import ChessGame

# White is up a queen; black has shuffled to the 50-move rule. White to move.
QUEEN_UP_SHUFFLE = "4k3/8/8/8/8/8/8/3QK3 w - - 100 80"
# Level king-and-rook each, 50-move rule reached.
LEVEL_SHUFFLE = "3rk3/8/8/8/8/8/8/3RK3 w - - 100 80"


def test_shuffled_draw_is_a_draw_when_adjudication_is_off():
    game = ChessGame(history_length=1, adjudicate_material_at=0)
    state = game.initial_state(QUEEN_UP_SHUFFLE)
    assert game.terminal_value(state) == 0.0


def test_shuffled_draw_is_a_win_for_the_material_leader():
    game = ChessGame(history_length=1, adjudicate_material_at=5)
    state = game.initial_state(QUEEN_UP_SHUFFLE)
    # White is to move and is a queen up.
    assert game.terminal_value(state) == 1.0


def test_adjudication_is_from_the_side_to_move_perspective():
    game = ChessGame(history_length=1, adjudicate_material_at=5)
    # Same position, black to move: black is a queen DOWN, so it is a loss for them.
    state = game.initial_state("4k3/8/8/8/8/8/8/3QK3 b - - 100 80")
    assert game.terminal_value(state) == -1.0


def test_level_shuffle_stays_a_draw():
    game = ChessGame(history_length=1, adjudicate_material_at=5)
    state = game.initial_state(LEVEL_SHUFFLE)
    assert game.terminal_value(state) == 0.0


def test_small_edge_stays_a_draw():
    """A single pawn should not be adjudicated as a win at a rook threshold."""
    game = ChessGame(history_length=1, adjudicate_material_at=5)
    state = game.initial_state("4k3/8/8/8/8/8/4P3/4K3 w - - 100 80")
    assert game.terminal_value(state) == 0.0


def test_stalemate_is_never_adjudicated():
    """Stalemate is a draw by rule no matter how far ahead you are."""
    game = ChessGame(history_length=1, adjudicate_material_at=5)
    state = game.initial_state("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert state.board.is_stalemate()
    assert game.terminal_value(state) == 0.0


def test_insufficient_material_is_never_adjudicated():
    game = ChessGame(history_length=1, adjudicate_material_at=5)
    state = game.initial_state("4k3/8/8/8/8/8/8/4K3 w - - 100 80")
    assert game.terminal_value(state) == 0.0


def test_checkmate_is_unaffected():
    game = ChessGame(history_length=1, adjudicate_material_at=5)
    state = game.initial_state(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    )
    assert game.terminal_value(state) == -1.0


def test_max_moves_cap_is_also_adjudicated():
    game = ChessGame(history_length=1, max_moves=2, adjudicate_material_at=5)
    state = game.initial_state("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    for uci in ("d1d4", "e8e7"):
        action = game.string_to_action(state, uci)
        state = game.next_state(state, action)
    assert state.ply >= game.max_moves
    assert game.terminal_value(state) == 1.0, "the cap should adjudicate, not draw"


# ------------------------------------------------------------------ relabelling


SHUFFLE_FEN = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"


def build_shuffle_record(game: ChessGame) -> GameRecord:
    """A short game that ends in a repetition draw with white a queen up.

    The origin is stored in metadata so relabelling replays the same game; real
    self-play records omit it and start from the standard position.
    """
    state = game.initial_state(SHUFFLE_FEN)
    moves = []
    for uci in ["d1d2", "e8e7", "d2d1", "e7e8", "d1d2", "e8e7", "d2d1", "e7e8"]:
        action = game.string_to_action(state, uci)
        assert action is not None, uci
        moves.append(action)
        state = game.next_state(state, action)

    plies = len(moves)
    return GameRecord(
        moves=moves,
        policy_indices=[[a] for a in moves],
        policy_probs=[[1.0] for _ in moves],
        values=[0.0] * plies,  # originally labelled a draw
        result=0.0,
        metadata={"fen": SHUFFLE_FEN},
    )


def test_relabel_turns_a_shuffled_draw_into_a_decisive_result():
    plain = ChessGame(history_length=1, adjudicate_material_at=0)
    record = build_shuffle_record(plain)
    assert record.result == 0.0

    adjudicating = ChessGame(history_length=1, adjudicate_material_at=5)
    updated, changed = relabel_record(adjudicating, record)

    assert changed
    assert updated.result != 0.0
    # Values must still alternate ply by ply.
    for earlier, later in zip(updated.values, updated.values[1:]):
        assert earlier == pytest.approx(-later)
    assert updated.metadata.get("relabelled") is True


def test_relabel_leaves_policy_targets_untouched():
    plain = ChessGame(history_length=1, adjudicate_material_at=0)
    record = build_shuffle_record(plain)
    adjudicating = ChessGame(history_length=1, adjudicate_material_at=5)

    updated, _ = relabel_record(adjudicating, record)
    assert updated.policy_indices == record.policy_indices
    assert updated.policy_probs == record.policy_probs
    assert updated.moves == record.moves


def test_relabel_is_a_no_op_when_nothing_changes():
    adjudicating = ChessGame(history_length=1, adjudicate_material_at=5)
    record = build_shuffle_record(adjudicating)
    # Built under the same rules, so relabelling must find nothing to do.
    relabelled, _ = relabel_record(adjudicating, record)
    again, changed = relabel_record(adjudicating, relabelled)
    assert not changed
    assert again.values == relabelled.values


def test_relabel_directory_rewrites_shards(tmp_path):
    plain = ChessGame(history_length=1, adjudicate_material_at=0)
    buffer = ReplayBuffer(tmp_path / "games")
    buffer.add_games([build_shuffle_record(plain) for _ in range(3)], generation=0)

    adjudicating = ChessGame(history_length=1, adjudicate_material_at=5)
    report = relabel_directory(adjudicating, buffer.directory)

    assert report["games"] == 3
    assert report["games_relabelled"] == 3
    assert report["shards_rewritten"] == 1
    assert report["to_win"] + report["to_loss"] == 3

    reloaded = read_shard(buffer.shards()[0])
    assert all(r.result != 0.0 for r in reloaded)


def test_relabel_survives_an_undecodable_record(tmp_path):
    """A record that cannot be replayed must be left alone, not dropped."""
    game = ChessGame(history_length=1, adjudicate_material_at=5)
    broken = GameRecord(
        moves=[9999], policy_indices=[[0]], policy_probs=[[1.0]],
        values=[0.0], result=0.0,
    )
    updated, changed = relabel_record(game, broken)
    assert not changed
    assert updated is broken
