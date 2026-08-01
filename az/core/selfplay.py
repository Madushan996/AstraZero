"""Self-play game generation.

Games are played in a batch that stays full: whenever one finishes, a fresh game takes
its slot, so the network always sees a full-width batch and the GPU is never waiting on
a single straggler game.

Everything is bounded by BOTH a game count and a wall-clock budget. The time budget is
what makes "train for about three hours today" a real instruction rather than a guess --
generation stops cleanly on the deadline and whatever was completed is kept.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from az.core.game import Game
from az.core.mcts import BatchedMCTS, Evaluator, MCTSConfig
from az.core.replay import GameRecord


@dataclass
class SelfPlayConfig:
    num_games: int = 64
    parallel_games: int = 32
    num_simulations: int = 200
    temperature: float = 1.0
    temperature_moves: int = 20  # plies of exploratory sampling before greedy play
    final_temperature: float = 0.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    c_puct_init: float = 1.25
    resign_threshold: Optional[float] = None  # e.g. -0.92; None disables resignation
    resign_min_ply: int = 30
    max_seconds: Optional[float] = None
    # `max_seconds` is a SOFT deadline: it stops new games from starting but lets
    # in-flight ones finish, because a partial game has no result to learn from. This
    # multiplier caps how far that overrun can go before in-flight games are abandoned,
    # so a slow generation cannot silently burn hours of paid GPU time.
    overrun_factor: float = 1.5
    seed: Optional[int] = None

    def mcts_config(self) -> MCTSConfig:
        return MCTSConfig(
            num_simulations=self.num_simulations,
            c_puct_init=self.c_puct_init,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
        )


@dataclass
class _LiveGame:
    """A game in progress occupying one batch slot."""

    state: object
    moves: list[int]
    policy_indices: list[list[int]]
    policy_probs: list[list[float]]
    root_values: list[float]

    @classmethod
    def start(cls, game: Game) -> "_LiveGame":
        return cls(
            state=game.initial_state(),
            moves=[],
            policy_indices=[],
            policy_probs=[],
            root_values=[],
        )


@dataclass
class SelfPlayResult:
    records: list[GameRecord]
    seconds: float
    positions: int

    @property
    def games(self) -> int:
        return len(self.records)

    def summary(self) -> dict:
        if not self.records:
            return {"games": 0, "positions": 0, "seconds": round(self.seconds, 1)}
        lengths = [len(r) for r in self.records]
        results = [r.result for r in self.records]
        return {
            "games": len(self.records),
            "positions": self.positions,
            "seconds": round(self.seconds, 1),
            "sec_per_game": round(self.seconds / len(self.records), 2),
            "avg_plies": round(float(np.mean(lengths)), 1),
            "first_player_wins": sum(1 for r in results if r > 0),
            "draws": sum(1 for r in results if r == 0),
            "second_player_wins": sum(1 for r in results if r < 0),
        }


def play_games(
    game: Game,
    evaluator: Evaluator,
    config: SelfPlayConfig,
    generation: int = 0,
    on_game_finished: Optional[Callable[[int, GameRecord], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> SelfPlayResult:
    """Generate up to `config.num_games` games, stopping early on the time budget.

    `should_stop` is polled between simulation rounds so Ctrl+C returns everything
    completed so far instead of throwing away the session's work.
    """
    rng = np.random.default_rng(config.seed)
    mcts = BatchedMCTS(game, evaluator, config.mcts_config())

    started = time.time()
    deadline = started + config.max_seconds if config.max_seconds else None
    hard_deadline = (
        started + config.max_seconds * max(1.0, config.overrun_factor)
        if config.max_seconds
        else None
    )

    width = max(1, min(config.parallel_games, config.num_games))
    live = [_LiveGame.start(game) for _ in range(width)]
    finished: list[GameRecord] = []
    launched = width

    while live:
        results = mcts.search([g.state for g in live], add_root_noise=True, rng=rng)

        still_live: list[_LiveGame] = []
        for live_game, result in zip(live, results):
            visits = result.visit_counts
            total = float(visits.sum())
            if total <= 0:
                # No legal moves were searched: the position is already over.
                finished.append(_finalize(game, live_game, generation, config))
                continue

            # The training target is always the raw visit distribution. Temperature
            # only affects which move we actually play -- mixing the two would teach
            # the network to imitate its own exploration noise.
            target = visits / total
            support = np.flatnonzero(visits > 0)

            ply = len(live_game.moves)
            temperature = (
                config.temperature
                if ply < config.temperature_moves
                else config.final_temperature
            )
            action = _sample_action(visits, temperature, rng)

            live_game.policy_indices.append([int(a) for a in support])
            live_game.policy_probs.append([float(p) for p in target[support]])
            live_game.root_values.append(float(result.root_value))
            live_game.moves.append(int(action))
            live_game.state = game.next_state(live_game.state, action)

            resigned = (
                config.resign_threshold is not None
                and ply >= config.resign_min_ply
                and result.root_value < config.resign_threshold
            )

            if resigned or game.is_terminal(live_game.state):
                record = _finalize(
                    game, live_game, generation, config, resigned=resigned
                )
                finished.append(record)
                if on_game_finished:
                    on_game_finished(len(finished), record)
            else:
                still_live.append(live_game)

        live = still_live

        if should_stop is not None and should_stop():
            # Abandon in-flight games (they have no result to learn from) and keep
            # everything already completed.
            break

        now = time.time()
        if hard_deadline is not None and now >= hard_deadline:
            # Overran the soft deadline by too much: cut losses rather than keep
            # paying for games that may not finish.
            break

        out_of_time = deadline is not None and now >= deadline
        while len(live) < width and launched < config.num_games and not out_of_time:
            live.append(_LiveGame.start(game))
            launched += 1

    seconds = time.time() - started
    return SelfPlayResult(
        records=finished,
        seconds=seconds,
        positions=sum(len(r) for r in finished),
    )


def _sample_action(
    visits: np.ndarray, temperature: float, rng: np.random.Generator
) -> int:
    if temperature <= 1e-6:
        return int(visits.argmax())
    scaled = np.power(visits, 1.0 / temperature)
    total = scaled.sum()
    if not np.isfinite(total) or total <= 0:
        return int(visits.argmax())
    return int(rng.choice(len(visits), p=scaled / total))


def _finalize(
    game: Game,
    live_game: _LiveGame,
    generation: int,
    config: SelfPlayConfig,
    resigned: bool = False,
) -> GameRecord:
    """Assign value targets by walking the result backwards, flipping sign each ply."""
    if resigned:
        # The resigning player was the mover at the position BEFORE the last move,
        # and that move has already been applied. So the player to move at the final
        # state is the resigner's opponent -- the winner. Getting this sign backwards
        # inverts the value target of every resigned game, which is silent and fatal.
        final_value = 1.0
    else:
        final_value = game.terminal_value(live_game.state)
        if final_value is None:
            final_value = 0.0  # hit a length cap: score as a draw

    num_plies = len(live_game.moves)
    values = [0.0] * num_plies
    value = float(final_value)
    for ply in range(num_plies - 1, -1, -1):
        value = -value  # perspective flips going back one ply
        values[ply] = value

    # values[0] is from the first player's perspective, which is the game result.
    result = values[0] if num_plies else 0.0

    return GameRecord(
        moves=live_game.moves,
        policy_indices=live_game.policy_indices,
        policy_probs=live_game.policy_probs,
        values=values,
        result=float(result),
        metadata={
            "generation": generation,
            "sims": config.num_simulations,
            "resigned": resigned,
            "plies": num_plies,
        },
    )
