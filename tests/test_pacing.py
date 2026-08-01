"""Generation pacing.

A worker plays `parallel` games concurrently and they finish together, so a generation
shorter than one batch produces ZERO games at full cost. This has caused two real
failures: `--parallel 96` needing 25 minutes per batch, and a short probe forced onto
`--parallel 4` that made the platform look four times slower than it was.
"""

from __future__ import annotations

import pytest

from modal_app import (
    MAX_GENERATION_SECONDS,
    MIN_GENERATION_SECONDS,
    SECONDS_PER_PLY_SIM_GAME,
    SELFPLAY_OVERRUN_FACTOR,
    SELFPLAY_STARTUP_MARGIN,
    SELFPLAY_WORKER_TIMEOUT,
    TYPICAL_PLIES,
    max_safe_selfplay_seconds,
    minimum_generation_seconds,
)


def batch_seconds(simulations: int, parallel: int) -> float:
    return SECONDS_PER_PLY_SIM_GAME * TYPICAL_PLIES * simulations * parallel


@pytest.mark.parametrize(
    "simulations,parallel",
    [(100, 16), (200, 16), (400, 8), (800, 4), (200, 32)],
)
def test_floor_always_allows_a_batch_to_finish(simulations, parallel):
    floor = minimum_generation_seconds(simulations, parallel, selfplay_fraction=0.85)
    selfplay_window = floor * 0.85
    assert selfplay_window >= batch_seconds(simulations, parallel), (
        f"{simulations} sims x {parallel} parallel needs "
        f"{batch_seconds(simulations, parallel):.0f}s but the floor only allows "
        f"{selfplay_window:.0f}s of self-play"
    )


def test_floor_never_drops_below_the_startup_floor():
    """Tiny settings must still not produce generations that are mostly container boot."""
    assert minimum_generation_seconds(8, 1) == MIN_GENERATION_SECONDS


def test_floor_grows_with_simulations_and_width():
    base = minimum_generation_seconds(200, 16)
    assert minimum_generation_seconds(400, 16) > base
    assert minimum_generation_seconds(200, 32) > base


def test_the_parallel_96_trap_is_caught():
    """The original failure: 96 concurrent games at 100 simulations takes ~25 minutes
    per batch, so any short generation yields nothing."""
    floor = minimum_generation_seconds(100, 96)
    assert floor > 1500, f"floor {floor:.0f}s would still allow an empty generation"


def test_selfplay_budget_cannot_outlive_the_worker_timeout():
    """Regression, and an expensive one.

    A worker writes its shard only when self-play RETURNS, so a container killed by its
    own timeout loses every game it played -- then `retries` runs it again to be killed
    again. A 3-hour session gave workers a 52-minute self-play budget which, with the
    1.5x overrun that lets in-flight games finish, could reach 78 minutes against a
    60-minute timeout.
    """
    budget = max_safe_selfplay_seconds()
    worst_case = budget * SELFPLAY_OVERRUN_FACTOR + SELFPLAY_STARTUP_MARGIN
    assert worst_case <= SELFPLAY_WORKER_TIMEOUT, (
        f"worst case {worst_case:.0f}s exceeds the {SELFPLAY_WORKER_TIMEOUT}s timeout"
    )


def test_any_generation_length_stays_inside_the_worker_timeout():
    """Whatever the session length, the derived self-play budget must be survivable."""
    for minutes in (30, 90, 180, 360, 720):
        remaining = minutes * 60.0
        floor = minimum_generation_seconds(200, 16)
        generation = min(remaining, MAX_GENERATION_SECONDS, max(floor, remaining / 3 + 60))
        selfplay = min(generation * 0.85, max_safe_selfplay_seconds())
        worst_case = selfplay * SELFPLAY_OVERRUN_FACTOR + SELFPLAY_STARTUP_MARGIN
        assert worst_case <= SELFPLAY_WORKER_TIMEOUT, (
            f"{minutes} min session produces a {selfplay:.0f}s budget, worst case "
            f"{worst_case:.0f}s > {SELFPLAY_WORKER_TIMEOUT}s"
        )


def test_generation_length_is_capped_to_bound_the_blast_radius():
    """No single generation should hold hours of unwritten work."""
    remaining = 6 * 3600.0
    floor = minimum_generation_seconds(200, 16)
    generation = min(remaining, MAX_GENERATION_SECONDS, max(floor, remaining / 3 + 60))
    assert generation <= MAX_GENERATION_SECONDS
    assert generation >= floor, "the cap must never fall below the batch-completion floor"


def test_headroom_is_applied():
    """Batches vary in length; the floor should not sit exactly on the boundary."""
    simulations, parallel = 200, 16
    floor = minimum_generation_seconds(simulations, parallel, selfplay_fraction=1.0)
    assert floor > batch_seconds(simulations, parallel) * 1.2
