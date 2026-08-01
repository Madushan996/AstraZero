"""Held-out evaluation.

Training loss falls whether or not the network is generalising. With a small replay
buffer it will happily memorise. The holdout split is the cheapest way to see that
happening, so it needs to actually be held out -- never trained on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from az.core.network import build_network, net_config_for
from az.core.replay import materialize
from az.core.selfplay import SelfPlayConfig, play_games
from az.core.mcts import uniform_evaluator
from az.core.trainer import (
    TrainConfig,
    build_optimizer,
    evaluate_loss,
    train_on_data,
)
from az.games.connect4 import Connect4


def make_data(games: int = 90, seed: int = 11):
    game = Connect4()
    config = SelfPlayConfig(
        num_games=games, parallel_games=20, num_simulations=8, seed=seed
    )
    records = play_games(game, uniform_evaluator(game), config).records
    return game, materialize(game, records)


def test_holdout_metrics_are_reported():
    game, data = make_data()
    assert len(data) >= 2000, f"need enough data to trigger the split, got {len(data)}"

    net = build_network(net_config_for(game, blocks=1, filters=16))
    config = TrainConfig(steps_per_generation=5, batch_size=32, use_amp=False)
    optimizer = build_optimizer(net, config)

    metrics = train_on_data(
        net, optimizer, data, config, torch.device("cpu"),
        generation=0, global_step=0, rng=np.random.default_rng(0),
    )

    assert metrics.holdout is not None
    for key in ("policy_loss", "value_loss", "value_mae"):
        assert key in metrics.holdout
    assert "holdout" in metrics.to_dict()

    # The comparison must be like-for-like: a matched sample of *training* rows scored
    # at the same moment, so the gap between them actually means something.
    for key in ("train_policy_loss", "train_value_loss", "policy_gap", "value_gap"):
        assert key in metrics.holdout

    gap = metrics.holdout["policy_gap"]
    assert gap == pytest.approx(
        metrics.holdout["policy_loss"] - metrics.holdout["train_policy_loss"], abs=1e-3
    )


def test_holdout_gap_is_near_zero_when_not_overfitting():
    """A network trained for only a handful of steps cannot have memorised anything,
    so held-out and seen data must score about the same."""
    game, data = make_data(games=120, seed=21)
    net = build_network(net_config_for(game, blocks=1, filters=16))
    config = TrainConfig(steps_per_generation=3, batch_size=32, use_amp=False)
    optimizer = build_optimizer(net, config)

    metrics = train_on_data(
        net, optimizer, data, config, torch.device("cpu"),
        generation=0, global_step=0, rng=np.random.default_rng(5),
    )

    assert abs(metrics.holdout["policy_gap"]) < 0.5, (
        f"unexpected generalisation gap after 3 steps: {metrics.holdout}"
    )


def test_holdout_is_skipped_on_tiny_datasets():
    """Splitting 40 positions would make the holdout number meaningless noise."""
    game, data = make_data(games=2, seed=3)
    net = build_network(net_config_for(game, blocks=1, filters=16))
    config = TrainConfig(steps_per_generation=2, batch_size=8, use_amp=False)
    optimizer = build_optimizer(net, config)

    metrics = train_on_data(
        net, optimizer, data, config, torch.device("cpu"),
        generation=0, global_step=0, rng=np.random.default_rng(0),
    )
    assert metrics.holdout is None
    assert "holdout" not in metrics.to_dict()


def test_holdout_fraction_zero_disables_it():
    game, data = make_data()
    net = build_network(net_config_for(game, blocks=1, filters=16))
    config = TrainConfig(
        steps_per_generation=2, batch_size=32, use_amp=False, holdout_fraction=0.0
    )
    optimizer = build_optimizer(net, config)

    metrics = train_on_data(
        net, optimizer, data, config, torch.device("cpu"),
        generation=0, global_step=0, rng=np.random.default_rng(0),
    )
    assert metrics.holdout is None


def test_evaluate_loss_respects_the_row_subset():
    game, data = make_data()
    net = build_network(net_config_for(game, blocks=1, filters=16))
    device = torch.device("cpu")

    everything = evaluate_loss(net, data, device)
    subset = evaluate_loss(net, data, device, rows=np.arange(0, 64))

    assert everything["policy_loss"] > 0
    assert subset["policy_loss"] > 0
    # Different sample, so the numbers should not be identical.
    assert everything != subset


def test_evaluate_loss_on_empty_rows_is_safe():
    game, data = make_data(games=4)
    net = build_network(net_config_for(game, blocks=1, filters=16))
    result = evaluate_loss(
        net, data, torch.device("cpu"), rows=np.zeros(0, dtype=np.int64)
    )
    assert result == {"policy_loss": 0.0, "value_loss": 0.0, "value_mae": 0.0}
