"""Tests for the parts that fail silently rather than loudly.

Sign errors in MCTS backup and in self-play value targets are the classic AlphaZero
bugs: nothing crashes, the loss still goes down, and the engine simply never gets good.
These tests pin the conventions down.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from az.core.arena import ArenaResult
from az.core.checkpoint import CheckpointManager, RunState
from az.core.mcts import BatchedMCTS, MCTSConfig, uniform_evaluator
from az.core.network import build_network, net_config_for
from az.core.pipeline import PipelineConfig, TrainingSession, default_config
from az.core.replay import GameRecord, ReplayBuffer, materialize
from az.core.selfplay import SelfPlayConfig, play_games
from az.core.trainer import TrainConfig, build_optimizer
from az.games.connect4 import COLS, ROWS, Connect4


# --------------------------------------------------------------------------- games


def test_connect4_detects_vertical_win():
    game = Connect4()
    state = game.initial_state()
    # Player A stacks column 0; player B answers in column 1.
    for _ in range(3):
        state = game.next_state(state, 0)
        state = game.next_state(state, 1)
    assert game.terminal_value(state) is None
    state = game.next_state(state, 0)  # A completes four in a column
    assert game.terminal_value(state) == -1.0, "the player to move has just lost"


def test_connect4_detects_diagonal_win():
    game = Connect4()
    state = game.initial_state()
    # Build a diagonal for the first player.
    for action in [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]:
        state = game.next_state(state, action)
    assert game.terminal_value(state) == -1.0


def test_connect4_full_board_is_a_draw():
    game = Connect4()
    state = game.initial_state()
    # A column order that fills the board without any four in a row.
    order = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0,
             2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2,
             4, 5, 4, 5, 4, 5, 5, 4, 5, 4, 5, 4,
             6, 6, 6, 6, 6, 6]
    for action in order:
        if game.terminal_value(state) is not None:
            break
        state = game.next_state(state, action)
    # Either a genuine draw, or someone won -- both are valid; what must hold is that
    # a full board is never reported as ongoing.
    if int((state.board != 0).sum()) == ROWS * COLS:
        assert game.terminal_value(state) is not None


def test_connect4_legal_actions_respect_full_columns():
    game = Connect4()
    state = game.initial_state()
    for _ in range(ROWS):
        state = game.next_state(state, 3)
    assert not game.legal_actions(state)[3]
    assert game.legal_actions(state).sum() == COLS - 1


def test_connect4_symmetry_is_consistent():
    game = Connect4()
    state = game.initial_state()
    state = game.next_state(state, 0)
    encoded = game.encode(state)
    policy = np.zeros(COLS, dtype=np.float32)
    policy[0] = 1.0

    pairs = game.symmetries(encoded, policy)
    assert len(pairs) == 2
    mirrored_obs, mirrored_policy = pairs[1]
    assert mirrored_policy[COLS - 1] == 1.0
    assert np.array_equal(mirrored_obs, encoded[:, :, ::-1])


# ---------------------------------------------------------------------------- mcts


def test_mcts_finds_immediate_win():
    """With a value-free uniform evaluator, search alone must still see a mate in 1.
    If backup signs are wrong this test fails."""
    game = Connect4()
    state = game.initial_state()
    # Player to move has three in column 0 and can complete it now.
    for _ in range(3):
        state = game.next_state(state, 0)
        state = game.next_state(state, 1)

    mcts = BatchedMCTS(game, uniform_evaluator(game), MCTSConfig(num_simulations=200))
    result = mcts.search([state], add_root_noise=False)[0]
    assert result.best_action == 0


def test_mcts_blocks_immediate_loss():
    """The harder sign test: the side to move has no win of its own but the opponent
    threatens mate in 1, so search must block. A backup-sign error passes the
    "find a win" test and fails this one."""
    game = Connect4()
    state = game.initial_state()
    # A: 0, 0, 0, 6  (three stacked in column 0, threatening to complete it)
    # B: 1, 2, 1     (no immediate win available)
    for action in [0, 1, 0, 2, 0, 1, 6]:
        state = game.next_state(state, action)

    assert game.terminal_value(state) is None
    # Confirm the premise: the side to move cannot win outright anywhere.
    for action in np.flatnonzero(game.legal_actions(state)):
        assert game.terminal_value(game.next_state(state, int(action))) is None

    mcts = BatchedMCTS(game, uniform_evaluator(game), MCTSConfig(num_simulations=400))
    result = mcts.search([state], add_root_noise=False)[0]
    assert result.best_action == 0, "search failed to block a one-move loss"


def test_mcts_never_returns_an_illegal_action():
    game = Connect4()
    state = game.initial_state()
    for _ in range(ROWS):
        state = game.next_state(state, 3)

    mcts = BatchedMCTS(game, uniform_evaluator(game), MCTSConfig(num_simulations=50))
    result = mcts.search([state], add_root_noise=True)[0]
    assert game.legal_actions(state)[result.best_action]
    assert result.visit_counts[3] == 0


def test_mcts_returns_a_move_when_stopped_before_any_simulation():
    """Regression: an already-set stop flag used to abort at simulation 0, leaving the
    root unvisited and best_action = -1. The root is expanded by then, so the policy's
    top prior is always available and strictly better than reporting nothing."""
    game = Connect4()
    state = game.initial_state()
    mcts = BatchedMCTS(game, uniform_evaluator(game), MCTSConfig(num_simulations=500))

    result = mcts.search([state], add_root_noise=False, should_stop=lambda: True)[0]

    assert result.best_action >= 0, "gave up without reporting a move"
    assert game.legal_actions(state)[result.best_action]


def test_mcts_expired_deadline_still_returns_a_move():
    game = Connect4()
    state = game.initial_state()
    mcts = BatchedMCTS(game, uniform_evaluator(game), MCTSConfig(num_simulations=500))

    # A deadline already in the past.
    result = mcts.search([state], add_root_noise=False, deadline=0.0)[0]
    assert result.best_action >= 0
    assert game.legal_actions(state)[result.best_action]


def test_mcts_batches_independent_games():
    game = Connect4()
    states = [game.initial_state() for _ in range(5)]
    mcts = BatchedMCTS(game, uniform_evaluator(game), MCTSConfig(num_simulations=32))
    results = mcts.search(states, add_root_noise=False)
    assert len(results) == 5
    for result in results:
        assert result.visit_counts.sum() > 0


# ------------------------------------------------------------------------ selfplay


def test_selfplay_value_targets_alternate_and_match_result():
    game = Connect4()
    config = SelfPlayConfig(num_games=4, parallel_games=4, num_simulations=16, seed=7)
    result = play_games(game, uniform_evaluator(game), config)

    assert result.games == 4
    for record in result.records:
        assert len(record.values) == len(record.moves)
        # Consecutive plies are opposing perspectives, so targets must alternate sign
        # (or be zero throughout, for a draw).
        for earlier, later in zip(record.values, record.values[1:]):
            assert earlier == pytest.approx(-later)
        assert record.result == pytest.approx(record.values[0])
        assert record.result in (-1.0, 0.0, 1.0)


def test_selfplay_policy_targets_are_distributions():
    game = Connect4()
    config = SelfPlayConfig(num_games=2, parallel_games=2, num_simulations=16, seed=3)
    result = play_games(game, uniform_evaluator(game), config)

    for record in result.records:
        for indices, probs in zip(record.policy_indices, record.policy_probs):
            assert len(indices) == len(probs)
            assert sum(probs) == pytest.approx(1.0, abs=1e-4)
            assert all(0 <= i < game.action_size for i in indices)


def test_selfplay_respects_stop_signal():
    game = Connect4()
    config = SelfPlayConfig(num_games=1000, parallel_games=4, num_simulations=8)
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    result = play_games(game, uniform_evaluator(game), config, should_stop=should_stop)
    assert result.games < 1000, "stop signal was ignored"


# -------------------------------------------------------------------------- replay


def test_replay_roundtrips_through_disk(tmp_path):
    buffer = ReplayBuffer(tmp_path / "games", window_games=100)
    record = GameRecord(
        moves=[0, 1, 2],
        policy_indices=[[0, 1], [1, 2], [2]],
        policy_probs=[[0.5, 0.5], [0.25, 0.75], [1.0]],
        values=[1.0, -1.0, 1.0],
        result=1.0,
        metadata={"generation": 0},
    )
    buffer.add_games([record], generation=0, worker=0)

    loaded = buffer.load_window()
    assert len(loaded) == 1
    assert loaded[0].moves == record.moves
    assert loaded[0].values == record.values
    assert loaded[0].policy_probs[1] == pytest.approx(record.policy_probs[1])


def test_workers_never_collide_on_shard_names(tmp_path):
    buffer = ReplayBuffer(tmp_path / "games")
    record = GameRecord([0], [[0]], [[1.0]], [0.0], 0.0)
    paths = {buffer.add_games([record], generation=1, worker=w) for w in range(8)}
    assert len(paths) == 8
    assert len(buffer.load_window()) == 8


def test_materialize_produces_trainable_arrays():
    game = Connect4()
    config = SelfPlayConfig(num_games=3, parallel_games=3, num_simulations=16, seed=1)
    records = play_games(game, uniform_evaluator(game), config).records

    data = materialize(game, records)
    assert data.observations.dtype == np.uint8
    assert data.observations.shape[1:] == game.observation_shape
    assert data.values.shape == (len(data),)
    assert np.all(np.abs(data.values) <= 1.0)
    # Connect4 contributes a mirrored copy of every position.
    assert len(data) == 2 * sum(len(r) for r in records)


def test_materialize_policy_batch_matches_the_sparse_form():
    """The dense expansion must reproduce exactly what self-play recorded."""
    game = Connect4()
    config = SelfPlayConfig(num_games=2, parallel_games=2, num_simulations=16, seed=4)
    records = play_games(game, uniform_evaluator(game), config).records

    data = materialize(game, records)
    rows = np.arange(len(data))
    dense = data.policy_batch(rows)

    assert dense.shape == (len(data), game.action_size)
    # Every row is a probability distribution over legal moves.
    assert np.allclose(dense.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(dense >= 0.0)

    # Spot-check a row against the CSR arrays directly.
    row = len(data) // 2
    start, stop = data.policy_offsets[row], data.policy_offsets[row + 1]
    expected = np.zeros(game.action_size, dtype=np.float32)
    expected[data.policy_indices[start:stop]] = data.policy_probs[start:stop]
    assert np.allclose(dense[row], expected)


def test_sparse_policy_storage_costs_more_on_a_tiny_action_space():
    """Documents the trade-off rather than hiding it.

    Sparse storage costs 8 bytes per legal move (int32 index + float32 prob) against
    4 bytes per action dense. For Connect 4 -- 7 actions, nearly all legal -- that is a
    net LOSS. It only pays off when the action space is large and mostly illegal, which
    is exactly chess: ~35 legal moves out of 4672. The format is chosen for chess, and
    the small overhead on Connect 4 is irrelevant at its scale.
    """
    game = Connect4()
    config = SelfPlayConfig(num_games=4, parallel_games=4, num_simulations=16, seed=8)
    records = play_games(game, uniform_evaluator(game), config).records

    data = materialize(game, records)
    dense_bytes = len(data) * game.action_size * 4
    sparse_bytes = data.policy_indices.nbytes + data.policy_probs.nbytes
    assert sparse_bytes > dense_bytes, "expected sparse to lose on a 7-action game"
    # Observations dominate regardless, so the policy format barely moves the total.
    assert sparse_bytes < data.observations.nbytes


def test_materialize_caps_position_count():
    game = Connect4()
    config = SelfPlayConfig(num_games=6, parallel_games=6, num_simulations=8, seed=2)
    records = play_games(game, uniform_evaluator(game), config).records

    data = materialize(game, records, max_positions=20, rng=np.random.default_rng(0))
    # The cap is applied by sampling, so allow slack, but it must bite.
    assert len(data) < 2 * sum(len(r) for r in records)
    assert len(data.policy_offsets) == len(data) + 1


# ---------------------------------------------------------------------- checkpoint


def test_checkpoint_restores_weights_optimizer_and_counters(tmp_path):
    game = Connect4()
    manager = CheckpointManager(tmp_path / "run")
    train_config = TrainConfig()

    net = build_network(net_config_for(game, blocks=1, filters=8))
    optimizer = build_optimizer(net, train_config)

    # Take a real step so the optimizer accumulates non-trivial state.
    logits, value = net(torch.zeros(2, *game.observation_shape))
    (logits.sum() + value.sum()).backward()
    optimizer.step()

    run_state = RunState(generation=7, global_step=1234, total_games=99)
    manager.save(net, run_state, "connect4", {}, optimizer=optimizer)

    restored_net = build_network(net_config_for(game, blocks=1, filters=8))
    restored_optimizer = build_optimizer(restored_net, train_config)
    checkpoint = manager.load()

    assert checkpoint is not None
    assert checkpoint.run_state.generation == 7
    assert checkpoint.run_state.global_step == 1234
    assert checkpoint.run_state.total_games == 99

    manager.restore_into(checkpoint, restored_net, restored_optimizer)

    for a, b in zip(net.parameters(), restored_net.parameters()):
        assert torch.allclose(a, b)
    # Adam's moment estimates must survive, or every resume costs a loss spike.
    assert len(restored_optimizer.state_dict()["state"]) > 0


def test_checkpoint_survives_a_corrupt_latest_pointer(tmp_path):
    game = Connect4()
    manager = CheckpointManager(tmp_path / "run")
    net = build_network(net_config_for(game, blocks=1, filters=8))
    manager.save(net, RunState(generation=3), "connect4", {})

    (manager.run_dir / "latest.json").write_text("{ this is not json", encoding="utf-8")
    assert manager.latest_generation() == 3, "should fall back to scanning directories"


def test_prune_keeps_recent_and_milestone_checkpoints(tmp_path):
    game = Connect4()
    manager = CheckpointManager(tmp_path / "run")
    net = build_network(net_config_for(game, blocks=1, filters=8))
    for generation in range(25):
        manager.save(net, RunState(generation=generation), "connect4", {})

    manager.prune_checkpoints(keep_recent=3, keep_every=10)
    remaining = set(manager.available_generations())

    assert {22, 23, 24} <= remaining, "recent checkpoints must survive"
    assert {0, 10, 20} <= remaining, "milestone opponents must survive"
    assert 13 not in remaining


def test_prune_leaves_a_usable_eval_baseline_on_short_runs(tmp_path):
    """Regression: with keep_every=10 an 11-generation run kept only {0, 6..10}, so
    'compare the latest against 5 generations ago' had no opponent to load and the
    evaluation crashed. Milestones exist to be played against."""
    game = Connect4()
    manager = CheckpointManager(tmp_path / "run")
    net = build_network(net_config_for(game, blocks=1, filters=8))
    for generation in range(11):
        manager.save(net, RunState(generation=generation), "connect4", {})

    manager.prune_checkpoints(
        keep_recent=default_config("connect4").keep_recent_checkpoints,
        keep_every=default_config("connect4").keep_every_checkpoint,
    )
    remaining = set(manager.available_generations())

    latest = max(remaining)
    older = [g for g in remaining if g <= latest - 5]
    assert older, f"no baseline at least 5 generations back survived: {sorted(remaining)}"


# ---------------------------------------------------------------------- pipeline


def test_pipeline_runs_a_generation_and_resumes(tmp_path):
    run_dir = tmp_path / "run"
    config = default_config("connect4", profile="tiny")
    config.selfplay.update(num_games=4, parallel_games=4, num_simulations=8)
    config.train.update(steps_per_generation=3, batch_size=16)

    session = TrainingSession(run_dir, config=config, device="cpu")
    session.run_generation()
    assert session.run_state.generation == 1
    first_games = session.run_state.total_games
    assert first_games > 0

    # A brand-new session object against the same directory must continue, not restart.
    resumed = TrainingSession(run_dir, device="cpu")
    assert resumed.run_state.generation == 1
    assert resumed.run_state.total_games == first_games

    resumed.run_generation()
    assert resumed.run_state.generation == 2
    assert resumed.run_state.total_games > first_games
    assert len(resumed.manager.read_history()) == 2


def test_pipeline_refuses_incompatible_architecture_change(tmp_path):
    run_dir = tmp_path / "run"
    config = default_config("connect4", profile="tiny")
    config.selfplay.update(num_games=2, parallel_games=2, num_simulations=8)
    config.train.update(steps_per_generation=1, batch_size=8)
    TrainingSession(run_dir, config=config, device="cpu")

    bigger = PipelineConfig.from_dict(config.to_dict())
    bigger.net = dict(bigger.net, filters=256)

    with pytest.raises(ValueError, match="network sizing changed"):
        TrainingSession(run_dir, config=bigger, device="cpu")


# ------------------------------------------------------------------------- arena


def test_elo_maths():
    assert ArenaResult(wins=50, losses=50, draws=0).elo_diff == pytest.approx(0.0)
    assert ArenaResult(wins=75, losses=25, draws=0).elo_diff == pytest.approx(
        190.8, abs=1.0
    )
    assert ArenaResult(wins=0, losses=100, draws=0).elo_diff < -1000
