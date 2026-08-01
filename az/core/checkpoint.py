"""Everything needed to stop training at any moment and resume days later.

This is the piece that makes the intended workflow work: train for a few hours, close
the laptop, come back next week, keep going with nothing lost.

A checkpoint deliberately stores more than model weights:

  * model weights            -- obvious
  * optimizer state          -- Adam's moment estimates. Dropping these makes the loss
                                spike for hundreds of steps after every single resume,
                                which quietly wastes a chunk of each session.
  * AMP scaler state         -- same reasoning for mixed-precision runs.
  * generation / global step -- so the LR schedule continues instead of restarting.
  * RNG state                -- makes a resumed run reproducible.
  * net + game config        -- a checkpoint knows how to rebuild its own model, so
                                `uci.py` can load any checkpoint with no side channel.

The replay buffer lives beside the checkpoints in the same run directory (see
replay.py) and is versioned by generation, so the whole run directory is the single
unit you back up or sync down from Modal.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from az.core.network import AlphaZeroNet, NetConfig, build_network

LATEST_POINTER = "latest.json"


@dataclass
class RunState:
    """Mutable bookkeeping that advances across sessions."""

    generation: int = 0
    global_step: int = 0
    total_games: int = 0
    total_positions: int = 0
    total_train_seconds: float = 0.0
    total_selfplay_seconds: float = 0.0
    sessions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass
class Checkpoint:
    net_config: NetConfig
    game_name: str
    game_kwargs: dict[str, Any]
    run_state: RunState
    model_state: dict[str, Any]
    optimizer_state: Optional[dict[str, Any]] = None
    scaler_state: Optional[dict[str, Any]] = None
    rng_state: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Owns the on-disk layout of a training run.

        run_dir/
            config.json           immutable run definition (game, net shape)
            latest.json           pointer to the newest generation
            history.jsonl         append-only log of every generation + eval
            checkpoints/gen00007/{model.pt, meta.json}
            games/gen00007_w3.jsonl.gz
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.games_dir = self.run_dir / "games"
        self.history_path = self.run_dir / "history.jsonl"
        self.config_path = self.run_dir / "config.json"
        for directory in (self.run_dir, self.checkpoint_dir, self.games_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # --- run definition -----------------------------------------------------

    def write_config(self, config: dict[str, Any]) -> None:
        _atomic_write_json(self.config_path, config)

    def read_config(self) -> Optional[dict[str, Any]]:
        if not self.config_path.exists():
            return None
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    # --- saving -------------------------------------------------------------

    def generation_dir(self, generation: int) -> Path:
        return self.checkpoint_dir / f"gen{generation:05d}"

    def save(
        self,
        net: AlphaZeroNet,
        run_state: RunState,
        game_name: str,
        game_kwargs: dict[str, Any],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Path:
        target = self.generation_dir(run_state.generation)
        target.mkdir(parents=True, exist_ok=True)

        payload = {
            "net_config": net.config.to_dict(),
            "game_name": game_name,
            "game_kwargs": game_kwargs,
            "run_state": run_state.to_dict(),
            "model_state": net.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
            },
            "extra": extra or {},
            "saved_at": time.time(),
        }

        # Write to a temp file first: a container killed mid-write must not be able to
        # corrupt the checkpoint you were going to resume from.
        tmp = target / "model.pt.partial"
        torch.save(payload, tmp)
        tmp.replace(target / "model.pt")

        _atomic_write_json(
            target / "meta.json",
            {
                "generation": run_state.generation,
                "game": game_name,
                "run_state": run_state.to_dict(),
                "net_config": net.config.to_dict(),
                "parameters": net.parameter_count(),
                "saved_at": time.time(),
                **(extra or {}),
            },
        )
        _atomic_write_json(
            self.run_dir / LATEST_POINTER,
            {"generation": run_state.generation, "path": target.name},
        )
        return target

    # --- loading ------------------------------------------------------------

    def latest_generation(self) -> Optional[int]:
        pointer = self.run_dir / LATEST_POINTER
        if pointer.exists():
            try:
                return int(json.loads(pointer.read_text(encoding="utf-8"))["generation"])
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        # Pointer missing or corrupt: fall back to scanning directories.
        generations = self.available_generations()
        return generations[-1] if generations else None

    def available_generations(self) -> list[int]:
        found = []
        for path in self.checkpoint_dir.glob("gen*"):
            if (path / "model.pt").exists():
                try:
                    found.append(int(path.name[3:]))
                except ValueError:
                    continue
        return sorted(found)

    def load(
        self, generation: Optional[int] = None, map_location: str = "cpu"
    ) -> Optional[Checkpoint]:
        if generation is None:
            generation = self.latest_generation()
        if generation is None:
            return None

        path = self.generation_dir(generation) / "model.pt"
        if not path.exists():
            return None

        payload = torch.load(path, map_location=map_location, weights_only=False)
        return Checkpoint(
            net_config=NetConfig.from_dict(payload["net_config"]),
            game_name=payload["game_name"],
            game_kwargs=payload.get("game_kwargs", {}),
            run_state=RunState.from_dict(payload["run_state"]),
            model_state=payload["model_state"],
            optimizer_state=payload.get("optimizer_state"),
            scaler_state=payload.get("scaler_state"),
            rng_state=payload.get("rng_state", {}),
            extra=payload.get("extra", {}),
        )

    def restore_into(
        self,
        checkpoint: Checkpoint,
        net: AlphaZeroNet,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        restore_rng: bool = True,
    ) -> None:
        net.load_state_dict(checkpoint.model_state)
        if optimizer is not None and checkpoint.optimizer_state is not None:
            optimizer.load_state_dict(checkpoint.optimizer_state)
        if scaler is not None and checkpoint.scaler_state is not None:
            scaler.load_state_dict(checkpoint.scaler_state)
        if restore_rng and checkpoint.rng_state:
            try:
                torch.set_rng_state(
                    checkpoint.rng_state["torch"].cpu().to(torch.uint8)
                )
                np.random.set_state(checkpoint.rng_state["numpy"])
            except (KeyError, TypeError, ValueError):
                pass  # RNG continuity is a nicety, never a reason to fail a resume.

    # --- history ------------------------------------------------------------

    def append_history(self, entry: dict[str, Any]) -> None:
        entry = {"time": time.time(), **entry}
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def read_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        entries = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    # --- housekeeping -------------------------------------------------------

    def prune_checkpoints(self, keep_recent: int = 5, keep_every: int = 10) -> list[int]:
        """Keep the newest `keep_recent`, plus every `keep_every`-th generation as a
        permanent milestone (those are your benchmark opponents). Delete the rest."""
        generations = self.available_generations()
        if not generations:
            return []

        keep = set(generations[-keep_recent:])
        keep.update(g for g in generations if keep_every > 0 and g % keep_every == 0)
        keep.add(generations[0])

        removed = []
        for generation in generations:
            if generation not in keep:
                shutil.rmtree(self.generation_dir(generation), ignore_errors=True)
                removed.append(generation)
        return removed

    def summary(self) -> dict[str, Any]:
        generations = self.available_generations()
        checkpoint = self.load() if generations else None
        return {
            "run_dir": str(self.run_dir),
            "generations": len(generations),
            "latest": generations[-1] if generations else None,
            "milestones": generations,
            "run_state": checkpoint.run_state.to_dict() if checkpoint else None,
        }


def load_net_from_checkpoint(
    checkpoint: Checkpoint, device: torch.device | str = "cpu"
) -> AlphaZeroNet:
    net = build_network(checkpoint.net_config, device)
    net.load_state_dict(checkpoint.model_state)
    net.eval()
    return net


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
