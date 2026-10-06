"""Shared model components for Qwen3-VL generation + classification."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def get_hidden_size(model: Any) -> int:
    config = model.config
    if hasattr(config, "text_config"):
        return int(config.text_config.hidden_size)
    return int(config.hidden_size)


class ClassificationHead(nn.Module):
    """CLSGen-style two-layer MLP over the final prompt representation.

    CLSGen uses a 2048-dimensional hidden layer and dropout=0.1 on an 8B
    decoder-only backbone.  LayerNorm is added here so the task head receives
    a stable float32 representation without changing the Qwen LM head.
    """

    def __init__(
        self,
        input_size: int,
        num_labels: int,
        hidden_size: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.dropout_probability = dropout
        self.norm = nn.LayerNorm(input_size)
        self.dense = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(hidden.float())
        hidden = self.dense(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return self.out_proj(hidden)
