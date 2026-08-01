"""Recompute value targets on already-recorded games.

Game records store the move list, so the outcome can be re-derived at any time under
different adjudication rules. That makes a labelling change retroactive: instead of
waiting for tens of thousands of mislabelled games to age out of the replay window, the
whole buffer is corrected in one pass.

This is a second payoff from storing games rather than encoded tensors. Tensors would
have baked the wrong value target in permanently.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from az.core.game import Game
from az.core.replay import GameRecord, SHARD_SUFFIX, read_shard, write_shard


def relabel_record(game: Game, record: GameRecord) -> tuple[GameRecord, bool]:
    """Replay a game under the game's current rules and rewrite its value targets.

    Returns the record and whether anything changed. Policy targets are untouched --
    they came from search and remain valid; only the outcome labels are re-derived.
    """
    # Self-play always starts from the standard position, but a record may record its
    # own origin -- honour it so replay cannot silently diverge from the real game.
    start_fen = record.metadata.get("fen")
    try:
        state = game.initial_state(start_fen) if start_fen else game.initial_state()
        for action in record.moves:
            state = game.next_state(state, action)
    except (ValueError, KeyError, TypeError):
        return record, False  # undecodable; leave it exactly as it was

    final_value = game.terminal_value(state)
    if final_value is None:
        final_value = 0.0

    num_plies = len(record.moves)
    values = [0.0] * num_plies
    value = float(final_value)
    for ply in range(num_plies - 1, -1, -1):
        value = -value
        values[ply] = value

    result = values[0] if num_plies else 0.0
    if values == record.values:
        return record, False

    updated = replace(
        record,
        values=values,
        result=float(result),
        metadata={**record.metadata, "relabelled": True},
    )
    return updated, True


def relabel_directory(
    game: Game, directory: Path, limit: Optional[int] = None
) -> dict[str, Any]:
    """Rewrite every shard in place. Each shard is written atomically, so an
    interrupted run leaves the buffer readable rather than half-corrupted."""
    directory = Path(directory)
    shards = sorted(directory.glob(f"*{SHARD_SUFFIX}"))
    if limit:
        shards = shards[:limit]

    changed_games = 0
    total_games = 0
    changed_shards = 0
    outcome_shift = {"to_win": 0, "to_loss": 0, "to_draw": 0}

    for path in shards:
        try:
            records = read_shard(path)
        except (OSError, EOFError, ValueError):
            continue

        updated_records = []
        shard_changed = False
        for record in records:
            before = record.result
            updated, changed = relabel_record(game, record)
            updated_records.append(updated)
            total_games += 1
            if changed:
                changed_games += 1
                shard_changed = True
                if before == 0.0 and updated.result > 0:
                    outcome_shift["to_win"] += 1
                elif before == 0.0 and updated.result < 0:
                    outcome_shift["to_loss"] += 1
                elif before != 0.0 and updated.result == 0.0:
                    outcome_shift["to_draw"] += 1

        if shard_changed:
            write_shard(path, updated_records)
            changed_shards += 1

    return {
        "shards": len(shards),
        "shards_rewritten": changed_shards,
        "games": total_games,
        "games_relabelled": changed_games,
        "share": round(100 * changed_games / max(total_games, 1), 1),
        **outcome_shift,
    }
