"""UCI protocol tests.

A GUI is unforgiving: if the engine ever fails to answer `bestmove`, or answers with an
illegal move, the GUI hangs or forfeits the game. These tests drive the engine the way
Arena and Cute Chess do, against a real (untrained) checkpoint.
"""

from __future__ import annotations

import time

import chess
import pytest

from az.core.checkpoint import CheckpointManager, RunState
from az.core.game import make_game
from az.core.network import build_network, net_config_for
from uci import UCIEngine, value_to_centipawns


@pytest.fixture(scope="module")
def chess_run(tmp_path_factory):
    """A run directory containing one tiny, untrained chess checkpoint."""
    run_dir = tmp_path_factory.mktemp("uci_run")
    game = make_game("chess", history_length=1)
    manager = CheckpointManager(run_dir)
    net = build_network(net_config_for(game, blocks=1, filters=16, value_hidden=16))
    manager.save(net, RunState(generation=1), "chess", {"history_length": 1})
    return run_dir


@pytest.fixture
def engine(chess_run):
    return UCIEngine(chess_run, simulations=24, device="cpu")


def drain(engine: UCIEngine, capsys, timeout: float = 60.0) -> list[str]:
    """Wait for the search thread, then return everything the engine printed."""
    deadline = time.time() + timeout
    while engine._search_thread and engine._search_thread.is_alive():
        if time.time() > deadline:
            pytest.fail("search thread did not finish")
        time.sleep(0.02)
    return [line for line in capsys.readouterr().out.splitlines() if line]


def test_uci_handshake(engine, capsys):
    engine.handle("uci")
    output = capsys.readouterr().out
    assert "id name" in output
    assert "id author" in output
    assert output.strip().endswith("uciok")


def test_isready(engine, capsys):
    engine.handle("isready")
    assert capsys.readouterr().out.strip() == "readyok"


def test_go_movetime_returns_a_legal_bestmove(engine, capsys):
    engine.handle("position startpos")
    engine.handle("go movetime 1000")
    lines = drain(engine, capsys)

    bestmove = [line for line in lines if line.startswith("bestmove")]
    assert len(bestmove) == 1

    uci_move = bestmove[0].split()[1]
    assert chess.Move.from_uci(uci_move) in chess.Board().legal_moves


def test_position_with_moves_is_applied(engine, capsys):
    engine.handle("position startpos moves e2e4 e7e5 g1f3")
    expected = chess.Board()
    for uci_move in ("e2e4", "e7e5", "g1f3"):
        expected.push(chess.Move.from_uci(uci_move))
    assert engine.state.board.fen() == expected.fen()

    engine.handle("go movetime 800")
    lines = drain(engine, capsys)
    uci_move = [l for l in lines if l.startswith("bestmove")][0].split()[1]
    assert chess.Move.from_uci(uci_move) in expected.legal_moves


def test_position_from_fen(engine, capsys):
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    engine.handle(f"position fen {fen}")
    assert engine.state.board.fen() == fen

    engine.handle("go movetime 800")
    lines = drain(engine, capsys)
    uci_move = [l for l in lines if l.startswith("bestmove")][0].split()[1]
    assert chess.Move.from_uci(uci_move) in chess.Board(fen).legal_moves


def test_info_line_is_well_formed(engine, capsys):
    engine.handle("position startpos")
    engine.handle("go nodes 32")
    lines = drain(engine, capsys)

    info = [line for line in lines if line.startswith("info depth")]
    assert info, "no info line emitted"
    tokens = info[0].split()
    for key in ("depth", "score", "nodes", "time", "pv"):
        assert key in tokens
    # The PV must be real, playable moves.
    board = chess.Board()
    for uci_move in tokens[tokens.index("pv") + 1 :]:
        move = chess.Move.from_uci(uci_move)
        assert move in board.legal_moves
        board.push(move)


def test_clock_based_go_respects_its_budget(engine, capsys):
    engine.handle("position startpos")
    started = time.time()
    engine.handle("go wtime 6000 btime 6000 winc 0 binc 0")
    lines = drain(engine, capsys)
    elapsed = time.time() - started

    assert any(line.startswith("bestmove") for line in lines)
    # 6s on the clock with movestogo defaulting to 30 means ~0.2s of thinking; allow
    # generous slack for a slow CPU but catch a runaway search.
    assert elapsed < 15.0, f"search overran its time budget ({elapsed:.1f}s)"


def test_stop_ends_an_infinite_search(engine, capsys):
    engine.handle("position startpos")
    engine.handle("go infinite")
    time.sleep(0.3)
    engine.handle("stop")
    lines = drain(engine, capsys, timeout=30.0)
    assert any(line.startswith("bestmove") for line in lines)


def test_immediate_stop_still_reports_a_searched_move(engine, capsys):
    """Regression: a GUI that sends `go` and `stop` back to back (or pipes commands in
    one go) used to abort before simulation 1, leaving the root unvisited. The engine
    then fell back to the first legal move python-chess happened to generate."""
    engine.handle("position startpos")
    engine.handle("go infinite")
    engine.handle("stop")  # no sleep: arrives while the search is still starting
    lines = drain(engine, capsys, timeout=30.0)

    bestmove = [line for line in lines if line.startswith("bestmove")]
    assert len(bestmove) == 1
    move = chess.Move.from_uci(bestmove[0].split()[1])
    assert move in chess.Board().legal_moves

    # It must be a real search/policy answer, not the emergency fallback.
    assert any(line.startswith("info depth") for line in lines), (
        "no info line: the engine took the arbitrary-move fallback path"
    )


def test_leading_bom_does_not_break_the_handshake(engine, capsys):
    """Some shells prepend a UTF-8 BOM to the first piped line."""
    engine.handle("﻿uci")
    output = capsys.readouterr().out
    assert "id name" in output
    assert output.strip().endswith("uciok")


def test_blank_command_is_safe(engine):
    assert engine.handle("   ") is True


def test_checkmate_position_still_answers(engine, capsys):
    """A mated position has no legal move; the engine must answer rather than hang."""
    engine.handle("position fen 7k/5QQ1/8/8/8/8/8/7K b - - 0 1")
    engine.handle("go movetime 500")
    lines = drain(engine, capsys)
    assert any(line.startswith("bestmove") for line in lines)


def test_ucinewgame_resets(engine, capsys):
    engine.handle("position startpos moves e2e4")
    engine.handle("ucinewgame")
    assert engine.state.board.fen() == chess.Board().fen()


def test_setoption_changes_simulations(engine, capsys):
    engine.handle("setoption name Simulations value 64")
    assert engine.simulations == 64
    engine.handle("setoption name MoveOverhead value 120")
    assert engine.move_overhead_ms == 120


def test_quit_returns_false(engine):
    assert engine.handle("quit") is False


def test_unknown_command_is_ignored(engine):
    assert engine.handle("this is not a uci command") is True


def test_centipawn_conversion_is_monotonic_and_signed():
    assert value_to_centipawns(0.0) == 0
    assert value_to_centipawns(0.5) > 0
    assert value_to_centipawns(-0.5) < 0
    assert value_to_centipawns(0.9) > value_to_centipawns(0.5)
    # Must stay finite at the extremes rather than overflowing.
    assert abs(value_to_centipawns(1.0)) < 10_000
    assert abs(value_to_centipawns(-1.0)) < 10_000
