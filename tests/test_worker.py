"""Multi-process self-play.

Self-play is ~97% of the cost of running this project, and MCTS tree descent is
single-threaded Python, so one process per container leaves most of a paid container
idle. These tests cover the fan-out that fixes that.

Subprocesses rather than `multiprocessing` is a deliberate choice: serverless runtimes
hold gRPC file descriptors, and mp spawn tries to hand the parent's fd table to each
child. On Beam that fails with "bad value(s) in fds_to_keep". A fresh interpreter
inherits nothing and behaves the same everywhere.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from az.core.checkpoint import CheckpointManager, RunState
from az.core.game import make_game
from az.core.network import build_network, net_config_for
from az.core.pipeline import default_config
from az.core.worker import (
    SUMMARY_PREFIX,
    _aggregate,
    _describe_failure,
    _parse_summary,
    run_parallel_selfplay,
    run_selfplay_shard,
)


@pytest.fixture
def connect4_run(tmp_path):
    """A minimal Connect 4 run directory with one tiny checkpoint."""
    run_dir = tmp_path / "run"
    config = default_config("connect4", profile="tiny")
    config.net = dict(blocks=1, filters=16, value_hidden=16)
    config.selfplay.update(num_simulations=8, parallel_games=4)

    manager = CheckpointManager(run_dir)
    manager.write_config(config.to_dict())

    game = make_game("connect4")
    net = build_network(net_config_for(game, **config.net))
    manager.save(net, RunState(generation=0), "connect4", {})
    return run_dir


def test_single_shard_writes_games(connect4_run):
    summary = run_selfplay_shard(
        run_dir=str(connect4_run),
        worker_tag="test0",
        seconds=20.0,
        num_games=4,
        simulations=8,
        parallel=4,
        seed=1,
        device_preference="cpu",
    )

    assert summary["games"] > 0
    shards = list((connect4_run / "games").glob("*test0*.jsonl.gz"))
    assert len(shards) == 1, "the shard must be named after the worker tag"


def test_parallel_selfplay_writes_one_shard_per_process(connect4_run):
    result = run_parallel_selfplay(
        run_dir=str(connect4_run),
        worker_id=7,
        processes=2,
        seconds=25.0,
        num_games=8,
        simulations=8,
        parallel=4,
        device_preference="cpu",
    )

    assert not result.get("failures"), f"child failed: {result.get('failures')}"
    assert result["processes"] == 2
    assert result["games"] > 0

    shards = sorted(p.name for p in (connect4_run / "games").glob("*.jsonl.gz"))
    assert len(shards) == 2, f"expected one shard per process, got {shards}"
    # Distinct names are what makes concurrent writes to a shared volume safe.
    assert len({s for s in shards}) == 2
    assert any("w007p0" in s for s in shards)
    assert any("w007p1" in s for s in shards)


def test_single_process_path_skips_subprocesses(connect4_run):
    """processes=1 must run inline -- spawning a child would be pure overhead."""
    result = run_parallel_selfplay(
        run_dir=str(connect4_run),
        worker_id=1,
        processes=1,
        seconds=20.0,
        num_games=3,
        simulations=8,
        parallel=3,
        device_preference="cpu",
    )
    assert result["processes"] == 1
    assert result["games"] > 0
    assert "wall_seconds" not in result


def test_child_failure_is_reported_not_swallowed(tmp_path):
    """A run directory with no checkpoint must surface an error, not silently
    report zero games as if self-play merely went slowly."""
    empty = tmp_path / "empty"
    (empty / "games").mkdir(parents=True)
    CheckpointManager(empty).write_config(default_config("connect4").to_dict())

    result = run_parallel_selfplay(
        run_dir=str(empty),
        worker_id=0,
        processes=2,
        seconds=5.0,
        num_games=2,
        device_preference="cpu",
    )

    assert result["games"] == 0
    assert result.get("failures"), "a dead child must be reported"
    assert any("checkpoint" in f.lower() for f in result["failures"])


def test_parse_summary_ignores_library_chatter():
    payload = {"games": 3, "positions": 40}
    stdout = (
        "some torch warning\n"
        "another line\n"
        + SUMMARY_PREFIX
        + json.dumps(payload)
        + "\n"
    )
    assert _parse_summary(stdout) == payload


def test_parse_summary_returns_none_without_a_summary():
    assert _parse_summary("just noise\n") is None
    assert _parse_summary("") is None
    assert _parse_summary(SUMMARY_PREFIX + "{not json") is None


def test_aggregate_sums_and_collects_failures():
    summaries = [
        {"games": 5, "positions": 100},
        {"games": 0, "positions": 0, "error": "boom"},
        {"games": 3, "positions": 60},
    ]
    aggregated = _aggregate(summaries)
    assert aggregated["games"] == 8
    assert aggregated["positions"] == 160
    assert aggregated["failures"] == ["boom"]


def test_aggregate_omits_failures_when_all_succeed():
    aggregated = _aggregate([{"games": 2, "positions": 10}])
    assert "failures" not in aggregated


def test_failure_description_surfaces_the_exception_line():
    """Regression: a generation lost half its workers and the log contained no
    exception at all, because the traceback was truncated from the wrong end."""
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/mnt/code/az/core/worker.py", line 247, in <module>\n'
        "    raise SystemExit(_main())\n"
        '  File "/mnt/code/az/core/worker.py", line 232, in _main\n'
        "    summary = run_selfplay_shard(\n"
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
    )
    described = _describe_failure(stderr, 1)
    assert "OutOfMemoryError" in described
    assert "CUDA out of memory" in described


def test_failure_description_handles_a_killed_child():
    described = _describe_failure("", -9)
    assert "signal 9" in described
    assert "OOM" in described


def test_failure_description_handles_silence():
    described = _describe_failure("", 3)
    assert "exit code 3" in described


def test_failure_description_falls_back_to_last_line():
    described = _describe_failure("something went wrong\nand then stopped\n", 1)
    assert "and then stopped" in described
