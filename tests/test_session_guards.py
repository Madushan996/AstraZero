"""Guards that stop a session from burning compute on a broken configuration.

Both of these come from a real Kaggle run that failed every self-play child with
cudaErrorNoKernelImageForDevice, produced zero games, and then iterated 76 empty
generations at four seconds each before anyone noticed.
"""

from __future__ import annotations

import torch

from az.core.checkpoint import CheckpointManager, RunState
from az.core.game import make_game
from az.core.network import build_network, net_config_for
from az.core.pipeline import TrainingSession, default_config
from az.core.worker import run_selfplay_shard


def build_run(tmp_path):
    run_dir = tmp_path / "run"
    config = default_config("connect4", profile="tiny")
    config.net = dict(blocks=1, filters=16, value_hidden=16)
    config.selfplay.update(num_games=2, parallel_games=2, num_simulations=8)
    config.train.update(steps_per_generation=1, batch_size=8)
    config.selfplay_processes = 1

    manager = CheckpointManager(run_dir)
    manager.write_config(config.to_dict())
    game = make_game("connect4")
    net = build_network(net_config_for(game, **config.net))
    manager.save(net, RunState(generation=0), "connect4", {})
    return run_dir, config


def test_session_stops_after_two_empty_generations(tmp_path, monkeypatch):
    """Zero games means zero learning; repeating it just burns the clock."""
    run_dir, config = build_run(tmp_path)
    session = TrainingSession(run_dir, config=config, device="cpu")

    calls = {"n": 0}

    def empty_generation(*args, **kwargs):
        calls["n"] += 1
        return {
            "generation": calls["n"],
            "selfplay": {"games": 0, "positions": 0, "seconds": 0.1},
            "train": {"steps": 0, "loss": 0.0},
        }

    monkeypatch.setattr(session, "run_generation", empty_generation)
    session.run_session(generations=25, verbose=False)

    assert calls["n"] == 2, f"ran {calls['n']} empty generations before stopping"


def test_a_productive_generation_resets_the_counter(tmp_path, monkeypatch):
    """One bad generation among good ones must not end the session."""
    run_dir, config = build_run(tmp_path)
    session = TrainingSession(run_dir, config=config, device="cpu")

    pattern = [5, 0, 7, 0, 3]
    calls = {"n": 0}

    def generation(*args, **kwargs):
        games = pattern[calls["n"]] if calls["n"] < len(pattern) else 4
        calls["n"] += 1
        return {
            "generation": calls["n"],
            "selfplay": {"games": games, "positions": games * 10, "seconds": 0.1},
            "train": {"steps": 1, "loss": 1.0},
        }

    monkeypatch.setattr(session, "run_generation", generation)
    session.run_session(generations=len(pattern), verbose=False)

    assert calls["n"] == len(pattern), "an isolated empty generation ended the session"


def test_worker_falls_back_to_cpu_when_the_gpu_is_unusable(tmp_path, monkeypatch):
    """torch.cuda.is_available() can be True on a GPU this build has no kernels for.
    The failure only appears when a kernel runs, so the worker probes with a real
    forward pass and drops to CPU rather than losing the whole shard."""
    run_dir, _ = build_run(tmp_path)

    # Pretend CUDA exists but every use of it explodes, as on the failing Kaggle run.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original_to = torch.nn.Module.to

    def exploding_to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if str(target).startswith("cuda"):
            raise RuntimeError("cudaErrorNoKernelImageForDevice")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", exploding_to)

    summary = run_selfplay_shard(
        run_dir=str(run_dir),
        worker_tag="fallback",
        seconds=20.0,
        num_games=2,
        simulations=8,
        parallel=2,
        seed=3,
        device_preference="cuda",
    )

    assert summary["games"] > 0, "fell over instead of falling back to CPU"
