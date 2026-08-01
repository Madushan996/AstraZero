<div align="center">

<img src="logo/logo.png" alt="AstraZero" width="200">

# AstraZero

**A chess engine that learned the game from nothing but its own games.**

No opening book. No endgame tablebases. No handcrafted evaluation. No human games.

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-124%20passing-3D7A57)](tests/)
[![License](https://img.shields.io/badge/license-MIT-A87A22)](LICENSE)

</div>

---

AlphaZero from scratch — game rules, move encoding, Monte-Carlo tree search, the network,
the training loop, cloud orchestration and a UCI interface. Built to be trained in
**sessions**: a few hours whenever you have credit, resuming exactly where the last one
stopped, playable in a real chess GUI in between.

It is currently around **200 Elo** and will lose to you. That is the honest state of a
project 1,700× short of AlphaZero's compute. What it does have is a working, measured
pipeline — and two findings that were worth more than any amount of extra training.

## Results

Training loss falls whether or not an engine improves, so the only evidence that counts
is games over the board.

```
generation 10 vs generation 0 (untrained)     +33  =24  -3     +191 Elo
```

| | value |
|---|---|
| Generation | 21 |
| Self-play games | 26,478 |
| Positions trained on | 4.2M |
| Network | 8 blocks × 128 filters, 12.1M parameters |
| Cost of all training so far | ~$58 |

## Two things I got wrong, and how they were found

Neither was visible in the loss curve. Both cost real money to discover, so they are
written up here in full.

### 1. Self-play was wasting 80% of what it paid for

MCTS tree descent is single-threaded Python, so one self-play process per container
saturates exactly **one CPU core** — no matter how many the container has — and leaves
the GPU mostly idle too.

Two symptoms give this away on any platform:

- throughput barely changes when you widen the search batch
- a fast GPU beats a slow one by only a few percent

Running one process per core fixed it. Same hardware, same bill:

| | games / worker-hour | cost / 1,000 games |
|---|---|---|
| one process | 215 | $6.74 |
| one process per core | **1,256** | **$1.22** |

### 2. A third of the training data taught the opposite of chess

The engine hung a piece almost every move. A probe of the value head explained why: given
a rook that could capture an undefended queen, it evaluated the position at `0.00` — even
after 800 simulations. It did not believe material mattered.

Sampling its own training data showed the cause. The engine wins material, cannot convert,
shuffles until the 50-move rule, and the game is labelled a draw:

| drawn games ending… | share |
|---|---|
| level | 10.7% |
| ≤ 2 pawns apart | 18.5% |
| 3–4 pawns apart | 22.2% |
| **a piece or more** | **24.5%** |
| **a queen or more** | **24.0%** |

**One game in three was teaching that a queen is worth nothing.** By the rules of chess
those really are draws — AlphaZero trained through this with 44 million games. At hobby
scale you cannot.

The fix scores a *shuffled* draw as a win for whoever is clearly ahead. Because the replay
buffer stores move lists rather than encoded tensors, the outcome could be re-derived for
every game already recorded — **7,115 of 21,733 games (32.7%) were relabelled in one
pass**, without regenerating anything.

Measured effect, asking the value head how it rates being a queen up versus a queen down,
with no search at all:

```
                      gen 14    gen 19    gen 21
value spread ±queen    0.072     0.261     0.426
tactics (200 sims)      3/5       4/5       4/5
capture a free queen   MISS       OK        OK
```

This is the only place human chess knowledge enters the system, and it enters purely as a
training label — never into the network, the encoding, or the search. Set
`adjudicate_material_at: 0` for a fully knowledge-free run, and expect to need far more
games.

## Play it

```bash
pip install -r requirements.txt
python package_engine.py --run runs/chess_modal --generation 21
```

That writes a self-contained engine:

```
engines/
    AstraZero_Gen21.bat      point your GUI at this
    AstraZero_Gen21.bmp      logo
    AstraZero_Gen21/         weights + config, 49 MB
```

In **Arena**: Engines → Install New Engine → select the `.bat` → choose **UCI**.
Also works with Cute Chess, BanksiaGUI, and anything else that speaks UCI.

Package two generations and they will report distinct names (`AstraZero_Gen21`,
`AstraZero_Gen14`), so you can run them against each other in a tournament. Watching one
generation beat an earlier one is the most convincing progress signal there is.

> **Give it time to think.** On a CPU-only machine this searches roughly 20–25 nodes per
> second. Use 30+ seconds per move; at blitz it barely searches at all.

## Train it

Start with Connect 4. It runs on the identical core, takes minutes on a laptop, and
catches the two classic AlphaZero bugs — flipped backup sign, wrong-perspective value
target — while they are still cheap to find:

```bash
python train.py train --run runs/c4 --game connect4 --profile tiny --generations 30
python train.py eval  --run runs/c4 --gap 20 --games 60
```

A positive Elo difference means the pipeline learns. Then chess:

```bash
python train.py train --run runs/chess --game chess --profile balanced --minutes 180
```

### In the cloud

Self-play parallelises perfectly and is ~97% of the cost, so fan-out matters more than a
fast GPU. Both [Modal](modal_app.py) and [Beam](beam_app.py) are supported:

```bash
modal run modal_app.py::init --game chess --profile balanced
modal run --detach modal_app.py::detached --minutes 180 \
    --workers 10 --processes 4 --simulations 200 --parallel 16
```

`--processes 4` is the setting from finding #1. Set it to the container's CPU count.

## How it works

Four pieces, none individually exotic:

1. **Game rules** — legal moves, terminal detection. For chess the fiddly part is the
   move encoding: 8×8×73 = 4,672 actions, exhaustively round-trip tested because a bug
   here does not crash, it silently caps strength.
2. **Network** — a residual tower with two heads: a policy over moves, and a value in
   [-1, 1] for who is winning.
3. **Search** — PUCT MCTS with no rollouts. Leaf values come from the value head.
4. **Self-play loop** — play games, store (position, search result, outcome), train the
   network to match, repeat.

The trick that makes it work: search is better than the raw network, because calculating
beats guessing. Training the network toward what search concluded makes the instinct
sharper — and search on top of a sharper instinct is better still.

### Design decisions worth knowing

**Search batches across games, not within one.** `BatchedMCTS` steps N independent trees
in lockstep and evaluates one leaf from each in a single forward pass, so the GPU sees
batches of N instead of 1. No virtual loss needed.

**The buffer stores games, not tensors.** A chess observation is ~30 KB; the move list
that generates it is a few hundred bytes. This paid off three separate times: migrating a
run between cloud providers cost 5 MB, a crashed training step lost nothing, and a
labelling fix reached retroactively across 21,733 games.

**Checkpoints include optimizer state.** Dropping Adam's moments makes the loss spike for
hundreds of steps after every resume — a real tax on a workflow built around stopping and
starting.

**Everything above the game rules is game-agnostic.** Connect 4 and chess share the same
search, trainer, buffer and checkpoint code.

## Project layout

| Path | What it is |
|---|---|
| [az/core/mcts.py](az/core/mcts.py) | Batched PUCT search |
| [az/core/network.py](az/core/network.py) | ResNet with policy + value heads |
| [az/core/selfplay.py](az/core/selfplay.py) | Game generation, Dirichlet noise, temperature |
| [az/core/trainer.py](az/core/trainer.py) | AlphaZero loss, LR schedule, held-out validation |
| [az/core/replay.py](az/core/replay.py) | Compact game-record buffer |
| [az/core/checkpoint.py](az/core/checkpoint.py) | Full resumable state |
| [az/core/worker.py](az/core/worker.py) | Multi-process self-play (finding #1) |
| [az/core/relabel.py](az/core/relabel.py) | Retroactive outcome relabelling (finding #2) |
| [az/games/chess_game.py](az/games/chess_game.py) | 119-plane chess encoding |
| [az/games/chess_encoding.py](az/games/chess_encoding.py) | The 4,672-move encoding |
| [uci.py](uci.py) | UCI engine |
| [tactics.py](tactics.py) | Diagnostics — is the value head learning material? |
| [modal_app.py](modal_app.py) / [beam_app.py](beam_app.py) | Cloud pipelines |

## Diagnostics

The most useful tool in the repo costs nothing to run:

```bash
python tactics.py --run runs/chess_modal --simulations 200
```

It queries the value head directly, with no search, and reports how strongly it separates
"up a queen" from "down a queen". A network can rank material correctly while compressing
every position to within 0.07 of a draw — which is materially blind in practice, because
search has nothing to steer on. That number found finding #2.

## Honest limitations

**It is weak.** ~200 Elo. It develops pieces and no longer gives material away for
nothing, but a casual club player beats it comfortably.

**The gap to AlphaZero is compute, not code.** 44 million games versus 26 thousand. At
measured rates roughly $200 reaches 100,000 games — the range where these engines
typically stop blundering outright.

**Measurement resolution is a real constraint.** Two similar networks draw ~77% of their
games, so a 60-game match resolves only about ±100 Elo. Small settings changes cannot be
A/B tested cheaply; only large ones.

## Tests

```bash
python -m pytest tests/ -q
```

124 tests. The chess move encoding is round-trip tested across thousands of moves from
random games, checking for index collisions — because that bug is silent.

## License

MIT — see [LICENSE](LICENSE).
