"""DeFM token 到 RGB fusion 空间的残差 bottleneck adapter。"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


class DepthAdapter(nn.Module):
    """LayerNorm → Linear(C→C/r) → GELU → Linear(C/r→C) + residual。"""

    def __init__(self, embed_dim: int, reduction: int = 4) -> None:
        super().__init__()
        if embed_dim <= 0 or reduction <= 0 or embed_dim // reduction <= 0:
            raise ValueError("embed_dim and reduction must produce a positive bottleneck")
        bottleneck = embed_dim // reduction
        self.norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, embed_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, inputs + self.up(self.act(self.down(self.norm(inputs)))))
