"""Replay buffer stored as compact, append-only game records on disk.

Two decisions worth explaining, because they are what make the "train an hour whenever
I feel like it, forever" workflow actually work:

1. We store GAMES, not encoded positions. A chess observation is 119x8x8 floats
   (~30 KB); the move list that generates it is a few hundred bytes. Games are replayed
   into tensors at training time. This is ~100x smaller on disk and means the buffer
   survives changes to the encoding or the network architecture -- if you later decide
   on a bigger net, you retrain it on every game you have ever generated instead of
   starting over.

2. Each producer writes its OWN shard file, never a shared one. Modal fans self-play out
   across many containers writing to one Volume; unique paths per worker means those
   writes can never conflict.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np

from az.core.game import Game

SHARD_SUFFIX = ".jsonl.gz"


@dataclass
class GameRecord:
    """One completed self-play game."""

    moves: list[int]
    policy_indices: list[list[int]]
    policy_probs: list[list[float]]
    values: list[float]  # target value for the player to move at each ply
    result: float  # final result from the first player's perspective
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.moves)

    def to_json(self) -> str:
        return json.dumps(
            {
                "m": self.moves,
                "pi": self.policy_indices,
                "pp": [[round(p, 5) for p in row] for row in self.policy_probs],
                "v": [round(v, 4) for v in self.values],
                "r": self.result,
                "meta": self.metadata,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> "GameRecord":
        data = json.loads(line)
        return cls(
            moves=data["m"],
            policy_indices=data["pi"],
            policy_probs=data["pp"],
            values=data["v"],
            result=data["r"],
            metadata=data.get("meta", {}),
        )


def write_shard(path: Path, records: Iterable[GameRecord]) -> int:
    """Write records to a gzipped JSONL shard. Returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = path.with_suffix(path.suffix + ".partial")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
            count += 1
    # Atomic-ish rename so a crashed writer never leaves a half-file that a reader
    # would choke on.
    tmp.replace(path)
    return count


def read_shard(path: Path) -> list[GameRecord]:
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(GameRecord.from_json(line))
    return records


class ReplayBuffer:
    """A sliding window over the most recent game shards in a directory."""

    def __init__(self, directory: Path, window_games: int = 50_000) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.window_games = window_games

    # --- writing ------------------------------------------------------------

    def shard_path(self, generation: int, worker: int | str) -> Path:
        return self.directory / f"gen{generation:05d}_w{worker}{SHARD_SUFFIX}"

    def add_games(
        self, records: Sequence[GameRecord], generation: int, worker: int | str = 0
    ) -> Path:
        path = self.shard_path(generation, worker)
        # If a worker id somehow repeats, don't clobber the earlier shard.
        if path.exists():
            path = self.directory / (
                f"gen{generation:05d}_w{worker}_{int(time.time()*1000)}{SHARD_SUFFIX}"
            )
        write_shard(path, records)
        return path

    # --- reading ------------------------------------------------------------

    def shards(self) -> list[Path]:
        """Shards sorted oldest-first by generation, then by name."""
        return sorted(
            (p for p in self.directory.glob(f"*{SHARD_SUFFIX}") if p.is_file()),
            key=lambda p: p.name,
        )

    def load_window(
        self,
        max_games: Optional[int] = None,
        max_plies: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> list[GameRecord]:
        """Load a training sample from the newest `window_games` games.

        Loading the entire window is not viable at scale: 40k chess games held as Python
        lists is several GB of object overhead, which OOMs a container long before the
        tensors do. So we pick shards at random from within the window and stop as soon
        as we have enough data for one generation of training.

        Sampling is uniform *within* the window, which preserves AlphaZero's
        "train on the most recent N games" semantics while bounding memory.
        """
        shards = self.shards()
        if not shards:
            return []

        # Restrict to the newest shards that could plausibly hold the window, then
        # shuffle so successive generations see different slices of it.
        candidates = shards[-max(1, len(shards)) :]
        order = list(range(len(candidates)))
        if rng is not None or max_plies is not None or max_games is not None:
            rng = rng or np.random.default_rng()
            rng.shuffle(order)
        else:
            order.reverse()  # newest first when taking everything

        records: list[GameRecord] = []
        plies = 0
        for index in order:
            try:
                loaded = read_shard(candidates[index])
            except (OSError, EOFError, json.JSONDecodeError):
                # A shard truncated by a killed container must not poison training.
                continue

            records.extend(loaded)
            plies += sum(len(r) for r in loaded)

            if len(records) >= self.window_games:
                break
            if max_games is not None and len(records) >= max_games:
                break
            if max_plies is not None and plies >= max_plies:
                break

        limit = min(self.window_games, max_games or self.window_games)
        return records[:limit]

    def stats(self) -> dict[str, Any]:
        shards = self.shards()
        total_bytes = sum(p.stat().st_size for p in shards)
        return {
            "shards": len(shards),
            "disk_mb": round(total_bytes / 1e6, 2),
            "newest": shards[-1].name if shards else None,
        }

    def prune(self, keep_shards: int) -> int:
        """Delete all but the newest `keep_shards` shards. Returns count removed."""
        shards = self.shards()
        doomed = shards[:-keep_shards] if keep_shards > 0 else shards
        for path in doomed:
            path.unlink(missing_ok=True)
        return len(doomed)


@dataclass
class TrainingData:
    """Materialised training set, with policy targets kept SPARSE.

    Dense policy targets are what break chess at scale: 250k positions x 4672 moves x
    float32 is 4.7 GB, and it is ~99% zeros because a chess position has about 35 legal
    moves. Storing (indices, probs) runs of only the legal moves is a ~130x saving, and
    the dense (batch, action_size) tensor the loss needs is cheap to build for one
    batch at a time -- 512 x 4672 floats is under 10 MB.
    """

    observations: np.ndarray  # (N, C, H, W) uint8
    policy_offsets: np.ndarray  # (N + 1,) int64, CSR-style row starts
    policy_indices: np.ndarray  # (nnz,) int32
    policy_probs: np.ndarray  # (nnz,) float32
    values: np.ndarray  # (N,) float32
    action_size: int

    def __len__(self) -> int:
        return len(self.observations)

    def policy_batch(self, rows: np.ndarray) -> np.ndarray:
        """Expand the sparse policy targets for `rows` into a dense array."""
        dense = np.zeros((len(rows), self.action_size), dtype=np.float32)
        for position, row in enumerate(rows):
            start, stop = self.policy_offsets[row], self.policy_offsets[row + 1]
            dense[position, self.policy_indices[start:stop]] = self.policy_probs[
                start:stop
            ]
        return dense

    def nbytes(self) -> int:
        return int(
            self.observations.nbytes
            + self.policy_offsets.nbytes
            + self.policy_indices.nbytes
            + self.policy_probs.nbytes
            + self.values.nbytes
        )


def materialize(
    game: Game,
    records: Sequence[GameRecord],
    max_positions: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> TrainingData:
    """Replay games into training tensors.

    Observations are stored as uint8: AlphaZero planes are overwhelmingly binary, and
    the handful of scalar planes are quantised to 0..255 by the game's encoder. That is
    a 4x memory saving over float32. Each game is replayed exactly once, so this is
    linear in total plies.
    """
    rng = rng or np.random.default_rng()

    channels, height, width = game.observation_shape
    total_plies = sum(len(r) for r in records)
    if total_plies == 0:
        return TrainingData(
            observations=np.zeros((0, channels, height, width), dtype=np.uint8),
            policy_offsets=np.zeros(1, dtype=np.int64),
            policy_indices=np.zeros(0, dtype=np.int32),
            policy_probs=np.zeros(0, dtype=np.float32),
            values=np.zeros(0, dtype=np.float32),
            action_size=game.action_size,
        )

    keep_probability = 1.0
    if max_positions is not None and total_plies > max_positions:
        keep_probability = max_positions / total_plies

    observations: list[np.ndarray] = []
    offsets: list[int] = [0]
    indices: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    values: list[float] = []
    running = 0

    for record in records:
        state = game.initial_state()
        for ply, action in enumerate(record.moves):
            if keep_probability >= 1.0 or rng.random() < keep_probability:
                encoded = game.encode(state)
                row_indices = np.asarray(record.policy_indices[ply], dtype=np.int32)
                row_probs = np.asarray(record.policy_probs[ply], dtype=np.float32)

                # Symmetries need the dense form to permute, but only one position's
                # worth exists at a time, so this stays small.
                if len(game.symmetries(encoded, row_probs)) > 1:
                    dense = np.zeros(game.action_size, dtype=np.float32)
                    dense[row_indices] = row_probs
                    variants = [
                        (obs, np.flatnonzero(policy), policy[np.flatnonzero(policy)])
                        for obs, policy in game.symmetries(encoded, dense)
                    ]
                else:
                    variants = [(encoded, row_indices, row_probs)]

                for aug_obs, aug_indices, aug_probs in variants:
                    observations.append(_quantize(aug_obs))
                    indices.append(np.asarray(aug_indices, dtype=np.int32))
                    probs.append(np.asarray(aug_probs, dtype=np.float32))
                    values.append(record.values[ply])
                    running += len(aug_indices)
                    offsets.append(running)

            state = game.next_state(state, action)

    return TrainingData(
        observations=np.stack(observations),
        policy_offsets=np.asarray(offsets, dtype=np.int64),
        policy_indices=(
            np.concatenate(indices) if indices else np.zeros(0, dtype=np.int32)
        ),
        policy_probs=(
            np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)
        ),
        values=np.asarray(values, dtype=np.float32),
        action_size=game.action_size,
    )


def _quantize(encoded: np.ndarray) -> np.ndarray:
    """float32 planes in [0, 1] -> uint8 in [0, 255]."""
    return np.clip(encoded * 255.0, 0, 255).astype(np.uint8)


def dequantize(batch: np.ndarray) -> np.ndarray:
    return batch.astype(np.float32) / 255.0


def iter_records(directory: Path) -> Iterator[GameRecord]:
    """Stream every record in a directory, oldest shard first."""
    for path in sorted(Path(directory).glob(f"*{SHARD_SUFFIX}")):
        yield from read_shard(path)
