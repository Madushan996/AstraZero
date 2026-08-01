"""Tests for the replay window's memory bounds.

Loading the whole window is the obvious implementation and it OOMs at chess scale, so
the bounding behaviour is pinned down here rather than left to be rediscovered on a
crashed container.
"""

from __future__ import annotations

import numpy as np

from az.core.replay import GameRecord, ReplayBuffer


def make_record(plies: int) -> GameRecord:
    return GameRecord(
        moves=list(range(plies)),
        policy_indices=[[i] for i in range(plies)],
        policy_probs=[[1.0] for _ in range(plies)],
        values=[1.0 if i % 2 == 0 else -1.0 for i in range(plies)],
        result=1.0,
    )


def test_load_window_respects_max_games(tmp_path):
    buffer = ReplayBuffer(tmp_path, window_games=1000)
    for worker in range(10):
        buffer.add_games([make_record(10) for _ in range(20)], generation=0,
                         worker=worker)

    records = buffer.load_window(max_games=50, rng=np.random.default_rng(0))
    assert len(records) == 50


def test_load_window_respects_max_plies(tmp_path):
    buffer = ReplayBuffer(tmp_path, window_games=10_000)
    for worker in range(20):
        buffer.add_games([make_record(50) for _ in range(10)], generation=0,
                         worker=worker)

    records = buffer.load_window(max_plies=600, rng=np.random.default_rng(0))
    total_plies = sum(len(r) for r in records)

    # It stops as soon as the budget is met, so it may overshoot by up to one shard
    # (10 games x 50 plies = 500), but must not load anything like all 10,000 plies.
    assert total_plies >= 600
    assert total_plies <= 600 + 500
    assert len(records) < 200


def test_load_window_respects_the_configured_window(tmp_path):
    buffer = ReplayBuffer(tmp_path, window_games=25)
    for worker in range(10):
        buffer.add_games([make_record(5) for _ in range(20)], generation=0,
                         worker=worker)

    assert len(buffer.load_window()) == 25


def test_load_window_samples_different_slices(tmp_path):
    """Successive generations should not train on an identical subset."""
    buffer = ReplayBuffer(tmp_path, window_games=10_000)
    for worker in range(30):
        record = make_record(10)
        record.metadata["worker"] = worker
        buffer.add_games([record], generation=0, worker=worker)

    first = buffer.load_window(max_plies=50, rng=np.random.default_rng(1))
    second = buffer.load_window(max_plies=50, rng=np.random.default_rng(2))

    workers_a = {r.metadata.get("worker") for r in first}
    workers_b = {r.metadata.get("worker") for r in second}
    assert workers_a != workers_b


def test_load_window_skips_a_corrupt_shard(tmp_path):
    buffer = ReplayBuffer(tmp_path, window_games=100)
    buffer.add_games([make_record(5) for _ in range(3)], generation=0, worker=0)

    corrupt = buffer.directory / "gen00000_w9.jsonl.gz"
    corrupt.write_bytes(b"this is not gzip data")

    records = buffer.load_window()
    assert len(records) == 3, "a corrupt shard must not take down training"


def test_prune_keeps_the_newest_shards(tmp_path):
    buffer = ReplayBuffer(tmp_path)
    for generation in range(12):
        buffer.add_games([make_record(4)], generation=generation, worker=0)

    removed = buffer.prune(keep_shards=5)
    assert removed == 7
    assert len(buffer.shards()) == 5
    assert "gen00011" in buffer.shards()[-1].name
