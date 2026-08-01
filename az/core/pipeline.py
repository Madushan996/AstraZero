"""The generation loop, and the session semantics that make it resumable.

One *generation* is: play games -> append to buffer -> train -> checkpoint.
One *session* is: as many generations as fit in your time budget.

Sessions are the unit you actually run. A session picks up at whatever generation the
last one left off at, and the only durable state is the run directory -- so the run
directory is the whole thing you back up, sync down from Modal, or hand to the UCI
engine.
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from az.core.checkpoint import CheckpointManager, RunState
from az.core.game import Game, make_game
from az.core.mcts import torch_evaluator
from az.core.network import AlphaZeroNet, NetConfig, build_network, net_config_for
from az.core.replay import ReplayBuffer, materialize
from az.core.selfplay import SelfPlayConfig, play_games
from az.core.trainer import TrainConfig, build_optimizer, train_on_data


@dataclass
class PipelineConfig:
    """The complete definition of a training run, persisted to config.json."""

    game_name: str = "chess"
    game_kwargs: dict[str, Any] = field(default_factory=dict)
    net: dict[str, Any] = field(default_factory=dict)  # NetConfig sizing overrides
    selfplay: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    buffer_window_games: int = 50_000
    keep_recent_checkpoints: int = 5
    # Milestone checkpoints are your Elo baselines, and a checkpoint is only ~150 MB
    # while a deleted one is unrecoverable. Every 5th generation rather than every 10th:
    # with a short run, keep_every=10 left only generations 0 and 10 alive, so the
    # obvious comparison "latest vs 10 generations ago" had nothing to play against.
    keep_every_checkpoint: int = 5
    keep_shards: int = 400

    def selfplay_config(self) -> SelfPlayConfig:
        return SelfPlayConfig(**self.selfplay)

    def train_config(self) -> TrainConfig:
        return TrainConfig(**self.train)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class InterruptGuard:
    """Turn Ctrl+C into a cooperative stop instead of a hard kill.

    The first Ctrl+C asks the pipeline to finish what it is doing and checkpoint.
    A second one aborts immediately, for when you really mean it.
    """

    def __init__(self) -> None:
        self.stop_requested = False
        self._previous: Any = None
        self._hits = 0

    def __enter__(self) -> "InterruptGuard":
        def handler(signum, frame):  # noqa: ANN001 - signal API
            self._hits += 1
            self.stop_requested = True
            if self._hits == 1:
                print(
                    "\n[interrupt] finishing current step and saving a checkpoint... "
                    "(press Ctrl+C again to abort immediately)",
                    flush=True,
                )
            else:
                raise KeyboardInterrupt

        try:
            self._previous = signal.signal(signal.SIGINT, handler)
        except ValueError:
            self._previous = None  # not on the main thread; guard becomes a no-op
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)

    def __call__(self) -> bool:
        return self.stop_requested


class TrainingSession:
    """Owns a run directory and advances it by whole generations."""

    def __init__(
        self,
        run_dir: Path,
        config: Optional[PipelineConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.manager = CheckpointManager(run_dir)

        stored = self.manager.read_config()
        if config is None:
            if stored is None:
                raise FileNotFoundError(
                    f"no config.json in {run_dir}; pass a PipelineConfig to start a "
                    f"new run"
                )
            self.config = PipelineConfig.from_dict(stored)
        else:
            self.config = config
            if stored is None:
                self.manager.write_config(config.to_dict())
            else:
                # Sizing must not change mid-run or the checkpoint stops loading.
                self._assert_compatible(stored, config)
                self.manager.write_config(config.to_dict())

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.game: Game = make_game(self.config.game_name, **self.config.game_kwargs)
        self.buffer = ReplayBuffer(
            self.manager.games_dir, window_games=self.config.buffer_window_games
        )

        self.net, self.optimizer, self.scaler, self.run_state = self._bootstrap()

    # --- setup --------------------------------------------------------------

    # Game settings that change the OBSERVATION or ACTION space. Changing one of these
    # makes every stored game and every trained weight incompatible, so it is refused.
    # Anything else (game length caps, how a shuffled draw is scored) only affects how
    # future games are labelled and may be tuned mid-run.
    ENCODING_CRITICAL_KWARGS = ("history_length",)

    def _assert_compatible(
        self, stored: dict[str, Any], incoming: PipelineConfig
    ) -> None:
        if stored.get("game_name") != incoming.game_name:
            raise ValueError(
                f"cannot change the game on an existing run "
                f"({stored.get('game_name')!r} -> {incoming.game_name!r})."
            )

        stored_kwargs = stored.get("game_kwargs", {}) or {}
        for key in self.ENCODING_CRITICAL_KWARGS:
            if stored_kwargs.get(key) != incoming.game_kwargs.get(key):
                raise ValueError(
                    f"cannot change {key!r} on an existing run "
                    f"({stored_kwargs.get(key)!r} -> {incoming.game_kwargs.get(key)!r}) "
                    f"-- it changes the observation encoding, so stored games and "
                    f"weights would no longer match. Start a new run directory."
                )
        stored_net, incoming_net = stored.get("net", {}), incoming.net
        if stored_net != incoming_net:
            raise ValueError(
                f"network sizing changed ({stored_net} -> {incoming_net}). Existing "
                f"weights will not load. Start a new run directory -- you can point it "
                f"at this run's games/ to reuse every game already generated."
            )

    def _bootstrap(self):
        train_config = self.config.train_config()
        checkpoint = self.manager.load(map_location=str(self.device))

        if checkpoint is not None:
            net = build_network(checkpoint.net_config, self.device)
            optimizer = build_optimizer(net, train_config)
            scaler = self._make_scaler(train_config)
            self.manager.restore_into(checkpoint, net, optimizer, scaler)
            run_state = checkpoint.run_state
            print(
                f"[resume] generation {run_state.generation}, "
                f"{run_state.total_games} games, {run_state.global_step} steps, "
                f"{run_state.total_train_seconds / 3600:.1f}h training so far"
            )
        else:
            net_config = net_config_for(self.game, **self.config.net)
            net = build_network(net_config, self.device)
            optimizer = build_optimizer(net, train_config)
            scaler = self._make_scaler(train_config)
            run_state = RunState()
            print(
                f"[new run] {self.config.game_name}, "
                f"{net.parameter_count() / 1e6:.2f}M parameters, device={self.device}"
            )

        return net, optimizer, scaler, run_state

    def _make_scaler(self, train_config: TrainConfig):
        if self.device.type != "cuda" or not train_config.use_amp:
            return None
        return torch.amp.GradScaler("cuda")

    # --- one generation -----------------------------------------------------

    def run_generation(
        self,
        selfplay_seconds: Optional[float] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        selfplay_config = self.config.selfplay_config()
        if selfplay_seconds is not None:
            selfplay_config.max_seconds = selfplay_seconds
        train_config = self.config.train_config()

        generation = self.run_state.generation

        # --- self-play ---
        evaluator = torch_evaluator(
            self.net, self.device, use_amp=train_config.use_amp
        )
        result = play_games(
            self.game,
            evaluator,
            selfplay_config,
            generation=generation,
            should_stop=should_stop,
        )

        if result.records:
            self.buffer.add_games(result.records, generation=generation)

        self.run_state.total_games += result.games
        self.run_state.total_positions += result.positions
        self.run_state.total_selfplay_seconds += result.seconds

        # --- train ---
        # Pull ~2x the positions we intend to train on, so materialize() has room to
        # subsample rather than being forced to take every ply of every game loaded.
        records = self.buffer.load_window(max_plies=train_config.max_positions * 2)
        data = materialize(
            self.game, records, max_positions=train_config.max_positions
        )

        metrics = train_on_data(
            self.net,
            self.optimizer,
            data,
            train_config,
            self.device,
            generation=generation,
            global_step=self.run_state.global_step,
            scaler=self.scaler,
        )

        self.run_state.global_step += metrics.steps
        self.run_state.total_train_seconds += metrics.seconds
        self.run_state.generation += 1

        # --- persist ---
        self.manager.save(
            self.net,
            self.run_state,
            self.config.game_name,
            self.config.game_kwargs,
            optimizer=self.optimizer,
            scaler=self.scaler,
        )

        entry = {
            "generation": generation,
            "selfplay": result.summary(),
            "train": metrics.to_dict(),
            "buffer": self.buffer.stats(),
            "window_games": len(records),
            "train_data_mb": round(data.nbytes() / 1e6, 1),
        }
        self.manager.append_history(entry)
        return entry

    # --- a whole session ----------------------------------------------------

    def run_session(
        self,
        minutes: Optional[float] = None,
        generations: Optional[int] = None,
        selfplay_fraction: float = 0.8,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Run until the time budget or generation count is exhausted.

        `selfplay_fraction` splits each generation's budget between generating games
        and training on them. Self-play dominates because it is the bottleneck; the
        network can only be as good as the games it has seen.
        """
        if minutes is None and generations is None:
            raise ValueError("specify minutes, generations, or both")

        deadline = time.time() + minutes * 60 if minutes else None
        self.run_state.sessions += 1
        entries: list[dict[str, Any]] = []
        completed = 0

        with InterruptGuard() as guard:
            while True:
                if generations is not None and completed >= generations:
                    break
                if deadline is not None and time.time() >= deadline:
                    break
                if guard.stop_requested:
                    break

                if deadline is not None:
                    remaining = deadline - time.time()
                    # Aim for ~8 generations per session so progress is checkpointed
                    # often, but never let one be so short it produces no games.
                    target = max(60.0, min(remaining, remaining / 4 + 60))
                    selfplay_seconds = target * selfplay_fraction
                else:
                    selfplay_seconds = None

                entry = self.run_generation(
                    selfplay_seconds=selfplay_seconds, should_stop=guard
                )
                entries.append(entry)
                completed += 1

                if verbose:
                    _print_generation(entry, self.run_state)

        self.housekeeping()
        if verbose:
            print(
                f"\n[session done] generation now {self.run_state.generation}, "
                f"{self.run_state.total_games} games lifetime. "
                f"Resume any time with the same run directory."
            )
        return entries

    def housekeeping(self) -> None:
        """Drop stale checkpoints and shards. Called at the end of a session."""
        self.manager.prune_checkpoints(
            keep_recent=self.config.keep_recent_checkpoints,
            keep_every=self.config.keep_every_checkpoint,
        )
        if self.config.keep_shards > 0:
            self.buffer.prune(keep_shards=self.config.keep_shards)


def _print_generation(entry: dict[str, Any], run_state: RunState) -> None:
    selfplay = entry["selfplay"]
    train = entry["train"]
    print(
        f"gen {entry['generation']:>4} | "
        f"games {selfplay.get('games', 0):>4} "
        f"({selfplay.get('sec_per_game', 0):.1f}s each, "
        f"{selfplay.get('avg_plies', 0):.0f} plies) | "
        f"loss {train.get('loss', 0):.3f} "
        f"(p {train.get('policy_loss', 0):.3f} v {train.get('value_loss', 0):.3f}) | "
        f"lr {train.get('lr', 0):.2e} | "
        f"lifetime {run_state.total_games} games",
        flush=True,
    )


def default_config(game_name: str, profile: str = "balanced") -> PipelineConfig:
    """Sensible starting points. `profile` trades speed against eventual strength."""
    profiles = {
        # For proving the pipeline works on a laptop CPU in minutes.
        "tiny": dict(
            net=dict(blocks=3, filters=48, value_hidden=64),
            selfplay=dict(
                num_games=48, parallel_games=24, num_simulations=60,
                temperature_moves=12,
            ),
            train=dict(
                batch_size=128, steps_per_generation=100, learning_rate=2e-3,
                use_amp=False, max_positions=60_000,
            ),
            buffer_window_games=4_000,
        ),
        # A real but affordable run on a mid-range cloud GPU.
        "balanced": dict(
            net=dict(blocks=8, filters=128, value_hidden=128),
            selfplay=dict(
                num_games=512, parallel_games=96, num_simulations=200,
                temperature_moves=20,
            ),
            train=dict(
                batch_size=512, steps_per_generation=400, learning_rate=2e-3,
                max_positions=250_000,
            ),
            buffer_window_games=40_000,
        ),
        # Closer to the paper's shape; needs a big GPU and real patience.
        "strong": dict(
            net=dict(blocks=14, filters=192, value_hidden=256),
            selfplay=dict(
                num_games=1024, parallel_games=128, num_simulations=400,
                temperature_moves=30,
            ),
            train=dict(
                batch_size=1024, steps_per_generation=600, learning_rate=2e-3,
                max_positions=500_000,
            ),
            buffer_window_games=100_000,
        ),
    }
    if profile not in profiles:
        raise KeyError(f"unknown profile {profile!r}; try {sorted(profiles)}")

    settings = profiles[profile]
    game_kwargs: dict[str, Any] = {}
    if game_name == "chess":
        # History length 8 is the paper's; 4 roughly halves encoding cost for a small
        # strength loss, which is a good trade at the tiny profile.
        game_kwargs = {"history_length": 4 if profile == "tiny" else 8}
        settings = dict(settings)
        settings["selfplay"] = dict(settings["selfplay"], dirichlet_alpha=0.3)
    elif game_name == "connect4":
        settings = dict(settings)
        settings["selfplay"] = dict(
            settings["selfplay"], dirichlet_alpha=1.0, temperature_moves=8
        )

    return PipelineConfig(game_name=game_name, game_kwargs=game_kwargs, **settings)
