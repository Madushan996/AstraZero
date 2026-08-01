"""Supervised training step: fit the network to MCTS visit counts and game outcomes.

AlphaZero's loss, exactly as in the paper:

    L = (z - v)^2  -  pi^T log(p)  +  c ||theta||^2

The L2 term is delegated to the optimizer's weight_decay. Note the policy term is a
full cross-entropy against a *distribution* (the visit counts), not against a single
label -- the search's relative preference among good moves is the signal, and collapsing
it to argmax throws most of it away.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from az.core.network import AlphaZeroNet
from az.core.replay import TrainingData, dequantize


@dataclass
class TrainConfig:
    batch_size: int = 256
    steps_per_generation: int = 250
    learning_rate: float = 2e-3
    min_learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "adamw"  # "adamw" or "sgd"
    lr_decay_generations: int = 200  # cosine decay horizon, in generations
    warmup_steps: int = 100
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    grad_clip: float = 1.0
    use_amp: bool = True
    max_positions: int = 200_000  # cap on positions materialised per generation
    # Positions withheld from training and scored afterwards. Training loss always
    # falls; the gap between it and this number is what tells you whether the network
    # is learning chess or memorising a small replay buffer.
    holdout_fraction: float = 0.05


@dataclass
class TrainMetrics:
    steps: int
    total_loss: float
    policy_loss: float
    value_loss: float
    value_mae: float
    learning_rate: float
    seconds: float
    positions: int
    holdout: Optional[dict[str, float]] = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "steps": self.steps,
            "loss": round(self.total_loss, 4),
            "policy_loss": round(self.policy_loss, 4),
            "value_loss": round(self.value_loss, 4),
            "value_mae": round(self.value_mae, 4),
            "lr": self.learning_rate,
            "seconds": round(self.seconds, 1),
            "positions": self.positions,
        }
        if self.holdout:
            payload["holdout"] = self.holdout
        return payload


def build_optimizer(net: AlphaZeroNet, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            net.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    return torch.optim.AdamW(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def learning_rate_at(config: TrainConfig, generation: int, step: int) -> float:
    """Cosine decay across generations, with a short warmup on the very first steps.

    Driven by the persisted generation counter, so a resumed run continues along the
    schedule rather than jumping back to the initial learning rate.
    """
    if step < config.warmup_steps and generation == 0:
        return config.learning_rate * (step + 1) / config.warmup_steps

    horizon = max(1, config.lr_decay_generations)
    progress = min(generation / horizon, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (
        config.learning_rate - config.min_learning_rate
    ) * cosine


def train_on_data(
    net: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    data: TrainingData,
    config: TrainConfig,
    device: torch.device,
    generation: int,
    global_step: int,
    scaler: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
) -> TrainMetrics:
    """Run `config.steps_per_generation` optimisation steps over the given data."""
    rng = rng or np.random.default_rng()
    net.train()

    num_positions = len(data)
    if num_positions == 0:
        return TrainMetrics(0, 0.0, 0.0, 0.0, 0.0, config.learning_rate, 0.0, 0)

    # Withhold a slice so the holdout score afterwards measures generalisation rather
    # than recall. Skipped when there is too little data for the split to mean anything.
    holdout_rows = np.zeros(0, dtype=np.int64)
    train_rows = np.arange(num_positions)
    if config.holdout_fraction > 0 and num_positions >= 2000:
        shuffled = rng.permutation(num_positions)
        cut = max(1, int(num_positions * config.holdout_fraction))
        holdout_rows, train_rows = shuffled[:cut], shuffled[cut:]

    batch_size = min(config.batch_size, len(train_rows))
    started = time.time()

    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "mae": 0.0}
    steps = 0
    learning_rate = config.learning_rate

    for step in range(config.steps_per_generation):
        indices = train_rows[rng.integers(0, len(train_rows), size=batch_size)]

        obs = torch.from_numpy(dequantize(data.observations[indices])).to(device)
        # Dense policy targets are built one batch at a time; the full set would be
        # gigabytes for chess.
        target_policy = torch.from_numpy(data.policy_batch(indices)).to(device)
        target_value = torch.from_numpy(data.values[indices]).to(device)

        learning_rate = learning_rate_at(config, generation, global_step + step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)

        use_amp = bool(scaler) and device.type == "cuda" and config.use_amp
        with torch.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            policy_logits, value_pred = net(obs)
            policy_loss = _policy_cross_entropy(policy_logits, target_policy)
            value_loss = F.mse_loss(value_pred, target_value)
            loss = (
                config.policy_loss_weight * policy_loss
                + config.value_loss_weight * value_loss
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["policy"] += float(policy_loss.detach())
        totals["value"] += float(value_loss.detach())
        totals["mae"] += float((value_pred.detach() - target_value).abs().mean())
        steps += 1

    divisor = max(steps, 1)

    # Both numbers must be measured the SAME way to be comparable. The running average
    # above is taken across all steps, including early ones when the network was still
    # bad, so it is always higher than a post-training snapshot -- comparing it against
    # holdout made holdout look better than training, which is meaningless. So score a
    # matched sample of training rows here, after training, alongside the holdout.
    holdout = None
    if len(holdout_rows):
        sample = holdout_rows
        train_sample = rng.choice(
            train_rows, size=min(len(sample), len(train_rows)), replace=False
        )
        held = evaluate_loss(net, data, device, rows=sample)
        seen = evaluate_loss(net, data, device, rows=train_sample)
        holdout = {
            **held,
            "train_policy_loss": seen["policy_loss"],
            "train_value_loss": seen["value_loss"],
            # Positive means the network does worse on data it has not seen.
            "policy_gap": round(held["policy_loss"] - seen["policy_loss"], 4),
            "value_gap": round(held["value_loss"] - seen["value_loss"], 4),
        }

    return TrainMetrics(
        steps=steps,
        total_loss=totals["loss"] / divisor,
        policy_loss=totals["policy"] / divisor,
        value_loss=totals["value"] / divisor,
        value_mae=totals["mae"] / divisor,
        learning_rate=learning_rate,
        seconds=time.time() - started,
        positions=num_positions,
        holdout=holdout,
    )


def _policy_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """-sum(pi * log p), averaged over the batch.

    log_softmax over the raw logits keeps this numerically stable. Targets are zero on
    illegal moves, so those terms drop out of the sum without needing an explicit mask.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


@torch.no_grad()
def evaluate_loss(
    net: AlphaZeroNet,
    data: TrainingData,
    device: torch.device,
    rows: Optional[np.ndarray] = None,
    batch_size: int = 512,
    max_batches: int = 20,
) -> dict[str, float]:
    """Loss on held-out data -- the honest check that the net is learning rather than
    memorising the most recent generation."""
    net.eval()
    all_rows = np.arange(len(data)) if rows is None else np.asarray(rows)
    if len(all_rows) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "value_mae": 0.0}

    totals = {"policy_loss": 0.0, "value_loss": 0.0, "value_mae": 0.0}
    batches = 0

    for start in range(0, len(all_rows), batch_size):
        if batches >= max_batches:
            break
        rows_batch = all_rows[start : start + batch_size]
        obs = torch.from_numpy(dequantize(data.observations[rows_batch])).to(device)
        target_policy = torch.from_numpy(data.policy_batch(rows_batch)).to(device)
        target_value = torch.from_numpy(data.values[rows_batch]).to(device)

        logits, value_pred = net(obs)
        totals["policy_loss"] += float(_policy_cross_entropy(logits, target_policy))
        totals["value_loss"] += float(F.mse_loss(value_pred, target_value))
        totals["value_mae"] += float((value_pred - target_value).abs().mean())
        batches += 1

    return {k: round(v / max(batches, 1), 4) for k, v in totals.items()}
