"""End-to-end smoke test of the chess path.

Kept fast by capping game length and simulations -- the point is to prove that chess
flows through self-play, the replay buffer, training and checkpointing without a shape
or sign error, not to train anything.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from az.core.mcts import BatchedMCTS, MCTSConfig, uniform_evaluator
from az.core.pipeline import TrainingSession, default_config
from az.core.replay import materialize
from az.core.selfplay import SelfPlayConfig, play_games
from az.games.chess_game import ChessGame


@pytest.fixture(scope="module")
def game():
    return ChessGame(history_length=2, max_moves=24)


def test_initial_position_has_twenty_legal_moves(game):
    state = game.initial_state()
    assert int(game.legal_actions(state).sum()) == 20


def test_checkmate_is_a_loss_for_the_side_to_move(game):
    # Fool's mate: black to move is mated.
    state = game.initial_state("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert state.board.is_checkmate()
    assert game.terminal_value(state) == -1.0
    assert game.legal_actions(state).sum() == 0


def test_stalemate_is_a_draw(game):
    state = game.initial_state("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert state.board.is_stalemate()
    assert game.terminal_value(state) == 0.0


def test_threefold_repetition_is_adjudicated(game):
    state = game.initial_state()
    # Shuffle knights back and forth to repeat the starting position.
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        assert game.terminal_value(state) is None or uci == "f6g8"
        action = game.string_to_action(state, uci)
        assert action is not None, f"could not parse {uci}"
        state = game.next_state(state, action)
    assert game.terminal_value(state) == 0.0, "repetition should be scored as a draw"


def test_max_moves_cap_ends_the_game(game):
    state = game.initial_state()
    rng = np.random.default_rng(0)
    while game.terminal_value(state) is None:
        legal = np.flatnonzero(game.legal_actions(state))
        state = game.next_state(state, int(rng.choice(legal)))
        assert state.ply <= game.max_moves
    assert game.terminal_value(state) is not None


def test_state_copy_is_isolated(game):
    """MCTS relies on next_state never mutating its input, across sibling branches."""
    state = game.initial_state()
    fen_before = state.board.fen()

    legal = np.flatnonzero(game.legal_actions(state))
    for action in legal[:5]:
        child = game.next_state(state, int(action))
        assert child.board.fen() != fen_before
        assert state.board.fen() == fen_before, "next_state mutated its input"
        # The repetition table must be copy-on-write, not shared.
        assert child.repetitions is not state.repetitions


def test_mcts_runs_on_chess(game):
    state = game.initial_state()
    mcts = BatchedMCTS(game, uniform_evaluator(game), MCTSConfig(num_simulations=32))
    result = mcts.search([state], add_root_noise=False)[0]

    assert result.best_action >= 0
    assert game.legal_actions(state)[result.best_action]
    move = game.action_to_move(state, result.best_action)
    assert move in state.board.legal_moves


def test_selfplay_produces_valid_chess_records(game):
    config = SelfPlayConfig(
        num_games=2, parallel_games=2, num_simulations=8, temperature_moves=4, seed=5
    )
    result = play_games(game, uniform_evaluator(game), config)
    assert result.games == 2

    for record in result.records:
        # Every recorded action must be replayable from the start position.
        state = game.initial_state()
        for action in record.moves:
            assert game.legal_actions(state)[action], "recorded an illegal move"
            state = game.next_state(state, action)
        assert game.terminal_value(state) is not None
        assert record.result in (-1.0, 0.0, 1.0)


def test_materialize_shapes_match_the_network_input(game):
    config = SelfPlayConfig(num_games=2, parallel_games=2, num_simulations=8, seed=6)
    records = play_games(game, uniform_evaluator(game), config).records

    data = materialize(game, records, max_positions=50)
    assert data.observations.shape[1:] == game.observation_shape
    assert data.action_size == 4672
    assert len(data.values) == len(data)
    # Chess has no usable symmetry, so there must be no augmentation.
    assert len(data) <= sum(len(r) for r in records)

    dense = data.policy_batch(np.arange(len(data)))
    assert dense.shape == (len(data), 4672)
    assert np.allclose(dense.sum(axis=1), 1.0, atol=1e-4)


def test_chess_policy_targets_stay_sparse(game):
    """At chess scale the dense form is gigabytes; confirm we never build it."""
    config = SelfPlayConfig(num_games=2, parallel_games=2, num_simulations=8, seed=9)
    records = play_games(game, uniform_evaluator(game), config).records

    data = materialize(game, records)
    dense_bytes = len(data) * 4672 * 4
    # Chess has ~35 legal moves per position out of 4672, so this must be a large win.
    assert data.nbytes() < dense_bytes
    average_legal = len(data.policy_indices) / max(len(data), 1)
    assert average_legal < 100, f"unexpectedly dense policy rows ({average_legal:.0f})"


def test_full_chess_generation_and_resume(tmp_path):
    config = default_config("chess", profile="tiny")
    config.game_kwargs = {"history_length": 1, "max_moves": 20}
    config.selfplay.update(
        num_games=2, parallel_games=2, num_simulations=6, temperature_moves=4
    )
    config.train.update(steps_per_generation=2, batch_size=8, max_positions=200)
    config.net = dict(blocks=1, filters=16, value_hidden=16)
    config.selfplay_processes = 1  # in-process: subprocess startup would dominate

    run_dir = tmp_path / "chess_run"
    session = TrainingSession(run_dir, config=config, device="cpu")
    entry = session.run_generation()

    assert entry["selfplay"]["games"] == 2
    assert entry["train"]["steps"] == 2
    assert session.run_state.generation == 1

    resumed = TrainingSession(run_dir, device="cpu")
    assert resumed.run_state.generation == 1
    assert resumed.run_state.total_games == session.run_state.total_games

    # The checkpoint must be self-describing enough for uci.py to load it cold.
    checkpoint = resumed.manager.load()
    assert checkpoint.game_name == "chess"
    assert checkpoint.game_kwargs == {"history_length": 1, "max_moves": 20}
    assert checkpoint.net_config.action_size == 4672
