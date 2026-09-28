from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from biot.biot import BIOTEncoder, ClassificationHead


class DualHeadBIOT(nn.Module):
    """One shared BIOT encoder with binary and four-level SAM heads."""

    def __init__(
        self,
        *,
        emb_size: int = 256,
        heads: int = 8,
        depth: int = 4,
        n_channels: int = 16,
        n_fft: int = 200,
        hop_length: int = 100,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or BIOTEncoder(
            emb_size=emb_size,
            heads=heads,
            depth=depth,
            n_channels=n_channels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.binary_head = ClassificationHead(emb_size, 2)
        self.level_head = ClassificationHead(emb_size, 4)

    def forward(self, inputs: torch.Tensor):
        shared = self.encoder(inputs)
        return shared, self.binary_head(shared), self.level_head(shared)
