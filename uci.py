"""UCI engine wrapper -- point Arena, Cute Chess or BanksiaGUI at this file.

Usage from a GUI: register the engine command as

    python C:\\path\\to\\Alphazeroclone\\uci.py --run C:\\path\\to\\runs\\chess

Register the same script twice with different --generation values to make two of your
own checkpoints play each other inside the GUI, which is the most convincing way to see
that the thing is actually improving.

Search runs on a background thread so `stop` and `quit` stay responsive, as the protocol
requires. Everything written to stdout is flushed immediately; GUIs will hang otherwise.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import chess
import torch

from az.core.checkpoint import CheckpointManager, load_net_from_checkpoint
from az.core.game import make_game
from az.core.mcts import BatchedMCTS, MCTSConfig, torch_evaluator
from az.games.chess_encoding import move_to_index

ENGINE_NAME = "AstraZero"
# Shown in GUIs and tournament cross-tables. Override with --author, or edit here.
ENGINE_AUTHOR = "AstraZero project"

DEFAULT_SIMULATIONS = 400
DEFAULT_MOVE_OVERHEAD_MS = 50


def value_to_centipawns(value: float) -> int:
    """Map a tanh value in (-1, 1) to a UCI centipawn score.

    Standard logistic conversion: treat (v+1)/2 as an expected score and invert the
    Elo formula. Purely cosmetic -- it only affects what the GUI displays.
    """
    win_probability = min(max((value + 1.0) / 2.0, 1e-4), 1 - 1e-4)
    return int(round(400.0 * math.log10(win_probability / (1 - win_probability))))


class UCIEngine:
    def __init__(
        self,
        run_dir: Path,
        generation: Optional[int] = None,
        simulations: int = DEFAULT_SIMULATIONS,
        device: Optional[str] = None,
        author: str = ENGINE_AUTHOR,
        name: Optional[str] = None,
    ) -> None:
        self.author = author
        self.name = name  # resolved after the checkpoint is loaded
        self.manager = CheckpointManager(run_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = self.manager.load(generation, map_location=str(self.device))
        if checkpoint is None:
            raise SystemExit(
                f"no checkpoint found in {run_dir}. Train at least one generation "
                f"first."
            )
        if checkpoint.game_name != "chess":
            raise SystemExit(
                f"this run is '{checkpoint.game_name}', not chess -- UCI does not "
                f"apply."
            )

        self.generation = checkpoint.run_state.generation
        # A distinct name per generation matters: engine-vs-engine tournaments key their
        # cross-tables on it, and two entrants sharing a name get merged into one row.
        if not self.name:
            self.name = f"{ENGINE_NAME}_Gen{self.generation}"
        self.game = make_game(checkpoint.game_name, **checkpoint.game_kwargs)
        self.net = load_net_from_checkpoint(checkpoint, self.device)
        self.simulations = simulations
        self.move_overhead_ms = DEFAULT_MOVE_OVERHEAD_MS

        self.state = self.game.initial_state()
        self._stop_event = threading.Event()
        self._search_thread: Optional[threading.Thread] = None

    # --- protocol -----------------------------------------------------------

    def run(self) -> None:
        for line in sys.stdin:
            command = line.strip()
            if not command:
                continue
            if not self.handle(command):
                break

    def handle(self, command: str) -> bool:
        """Returns False when the engine should exit."""
        # Strip a UTF-8 BOM: some shells and GUIs prepend one to the first line, and a
        # BOM'd "uci" silently matches nothing, so the handshake appears to hang.
        parts = command.lstrip("﻿").split()
        if not parts:
            return True
        verb = parts[0]

        if verb == "uci":
            self._identify()
        elif verb == "isready":
            respond("readyok")
        elif verb == "setoption":
            self._set_option(parts)
        elif verb == "ucinewgame":
            self.state = self.game.initial_state()
        elif verb == "position":
            self._set_position(parts)
        elif verb == "go":
            self._go(parts)
        elif verb == "stop":
            self._stop_event.set()
        elif verb in {"quit", "exit"}:
            self._stop_event.set()
            self._join_search()
            return False
        # Unknown commands are ignored, per the protocol.
        return True

    def _identify(self) -> None:
        respond(f"id name {self.name}")
        respond(f"id author {self.author}")
        respond(
            f"option name Simulations type spin default {self.simulations} "
            f"min 8 max 1000000"
        )
        respond(
            f"option name MoveOverhead type spin default {self.move_overhead_ms} "
            f"min 0 max 5000"
        )
        respond("uciok")

    def _set_option(self, parts: list[str]) -> None:
        if "name" not in parts or "value" not in parts:
            return
        name = " ".join(parts[parts.index("name") + 1 : parts.index("value")]).lower()
        value = " ".join(parts[parts.index("value") + 1 :])
        try:
            if name == "simulations":
                self.simulations = max(8, int(value))
            elif name == "moveoverhead":
                self.move_overhead_ms = max(0, int(value))
        except ValueError:
            pass

    def _set_position(self, parts: list[str]) -> None:
        if "startpos" in parts:
            self.state = self.game.initial_state()
            move_index = parts.index("startpos") + 1
        elif "fen" in parts:
            fen_start = parts.index("fen") + 1
            fen_end = parts.index("moves") if "moves" in parts else len(parts)
            fen = " ".join(parts[fen_start:fen_end])
            self.state = self.game.initial_state(fen)
            move_index = fen_end
        else:
            return

        if move_index < len(parts) and parts[move_index] == "moves":
            for token in parts[move_index + 1 :]:
                action = self.game.string_to_action(self.state, token)
                if action is None:
                    respond(f"info string could not apply move {token}")
                    break
                self.state = self.game.next_state(self.state, action)

    # --- search -------------------------------------------------------------

    def _go(self, parts: list[str]) -> None:
        self._join_search()
        self._stop_event.clear()

        options = _parse_go(parts)
        budget = self._time_budget(options)
        simulations = options.get("nodes") or self.simulations

        self._search_thread = threading.Thread(
            target=self._search, args=(budget, simulations), daemon=True
        )
        self._search_thread.start()

    def _time_budget(self, options: dict[str, int]) -> Optional[float]:
        """Seconds to think, or None for unlimited (until `stop`)."""
        if options.get("infinite"):
            return None
        if "movetime" in options:
            return max(0.01, (options["movetime"] - self.move_overhead_ms) / 1000.0)

        white_to_move = self.state.board.turn == chess.WHITE
        remaining = options.get("wtime" if white_to_move else "btime")
        if remaining is None:
            return None

        increment = options.get("winc" if white_to_move else "binc", 0)
        moves_to_go = options.get("movestogo", 30)
        # Spend a steady fraction of what is left, plus most of the increment. Simple
        # and safe: never risk flagging to squeeze out a slightly better move.
        allocation = remaining / max(moves_to_go, 1) + increment * 0.8
        allocation = min(allocation, remaining * 0.4)
        return max(0.02, (allocation - self.move_overhead_ms) / 1000.0)

    def _search(self, budget: Optional[float], simulations: int) -> None:
        started = time.monotonic()
        deadline = started + budget if budget is not None else None

        mcts = BatchedMCTS(
            self.game,
            torch_evaluator(self.net, self.device),
            MCTSConfig(num_simulations=simulations),
        )

        try:
            result = mcts.search(
                [self.state],
                add_root_noise=False,
                deadline=deadline,
                should_stop=self._stop_event.is_set,
            )[0]
        except Exception as error:  # a crash here would hang the GUI forever
            respond(f"info string search failed: {error}")
            respond(self._fallback_bestmove())
            return

        elapsed = max(time.monotonic() - started, 1e-3)

        if result.best_action < 0:
            respond(self._fallback_bestmove())
            return

        pv = self._pv_to_uci(result.principal_variation)
        respond(
            f"info depth {max(len(pv), 1)} "
            f"score cp {value_to_centipawns(result.root_value)} "
            f"nodes {result.nodes} "
            f"nps {int(result.nodes / elapsed)} "
            f"time {int(elapsed * 1000)} "
            f"pv {' '.join(pv)}"
        )

        move = self.game.action_to_move(self.state, result.best_action)
        respond(f"bestmove {move.uci() if move else '0000'}")

    def _pv_to_uci(self, actions: list[int]) -> list[str]:
        moves: list[str] = []
        board = self.state.board.copy()
        for action in actions:
            move = None
            for candidate in board.legal_moves:
                if move_to_index(candidate, board.turn) == action:
                    move = candidate
                    break
            if move is None:
                break
            moves.append(move.uci())
            board.push(move)
        return moves

    def _fallback_bestmove(self) -> str:
        """Never return nothing: a GUI that gets no bestmove waits forever."""
        legal = list(self.state.board.legal_moves)
        return f"bestmove {legal[0].uci()}" if legal else "bestmove 0000"

    def _join_search(self) -> None:
        if self._search_thread and self._search_thread.is_alive():
            self._search_thread.join(timeout=10.0)


def _parse_go(parts: list[str]) -> dict[str, int]:
    options: dict[str, int] = {}
    integer_keys = {
        "wtime", "btime", "winc", "binc", "movestogo", "movetime", "depth", "nodes",
    }
    index = 1
    while index < len(parts):
        token = parts[index]
        if token == "infinite":
            options["infinite"] = 1
        elif token in integer_keys and index + 1 < len(parts):
            try:
                options[token] = int(parts[index + 1])
            except ValueError:
                pass
            index += 1
        index += 1
    return options


def respond(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="UCI wrapper for an AstraZero run")
    parser.add_argument("--run", type=Path, required=True, help="run directory")
    parser.add_argument(
        "--generation", type=int, help="checkpoint to load (default: newest)"
    )
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--device", help="cuda / cpu")
    parser.add_argument(
        "--author", default=ENGINE_AUTHOR, help="name shown as the engine's author"
    )
    parser.add_argument(
        "--name",
        help="engine name reported to the GUI (default: AstraZero_GenN)",
    )
    args = parser.parse_args()

    engine = UCIEngine(
        args.run,
        generation=args.generation,
        simulations=args.simulations,
        device=args.device,
        author=args.author,
        name=args.name,
    )
    engine.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
