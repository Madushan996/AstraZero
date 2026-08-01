"""AlphaZero residual network: a shared conv tower with policy and value heads.

Size is fully configurable so the same code runs a 4x64 net on a laptop CPU and a
20x256 net on an A100. The architecture is recorded inside every checkpoint, so a
checkpoint always knows how to rebuild its own model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NetConfig:
    """Everything needed to reconstruct the model."""

    in_channels: int
    board_h: int
    board_w: int
    action_size: int
    blocks: int = 6
    filters: int = 96
    value_hidden: int = 128
    policy_channels: int = 32
    value_channels: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})


class ResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    """Outputs raw policy logits and a tanh value in [-1, 1].

    Logits (not probabilities) are returned so that illegal-move masking can be done
    correctly by the caller: masking after softmax would leave probability mass
    stranded on illegal moves.
    """

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, config.filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(config.filters),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            *[ResidualBlock(config.filters) for _ in range(config.blocks)]
        )

        cells = config.board_h * config.board_w

        self.policy_head = nn.Sequential(
            nn.Conv2d(config.filters, config.policy_channels, 1, bias=False),
            nn.BatchNorm2d(config.policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(config.policy_channels * cells, config.action_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(config.filters, config.value_channels, 1, bias=False),
            nn.BatchNorm2d(config.value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(config.value_channels * cells, config.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tower(self.stem(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    @torch.no_grad()
    def predict(
        self, observations: torch.Tensor, legal_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched inference for MCTS: masked policy probabilities and values.

        `legal_masks` is a bool tensor of shape (batch, action_size). Illegal actions
        get -inf logits before the softmax, so they receive exactly zero probability.
        """
        self.eval()
        logits, values = self.forward(observations)
        logits = logits.masked_fill(~legal_masks, float("-inf"))
        return F.softmax(logits, dim=-1), values

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_network(config: NetConfig, device: torch.device | str = "cpu") -> AlphaZeroNet:
    return AlphaZeroNet(config).to(device)


def net_config_for(game: Any, **overrides: Any) -> NetConfig:
    """Derive a NetConfig from a Game, letting the caller override sizing."""
    channels, height, width = game.observation_shape
    return NetConfig(
        in_channels=channels,
        board_h=height,
        board_w=width,
        action_size=game.action_size,
        **overrides,
    )
