"""
keyword_spotting.model
========================
TinyKWSNet - a small CNN binary classifier (NOVA vs unknown), sized so
it can eventually be replaced by a quantized MCU-compatible model
without changing anything else in the KWS pipeline.

Label convention: class index 0 = unknown/negative, class index 1 = NOVA.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyKWSNet(nn.Module):
    """
    Input : log-Mel spectrogram, shape (batch, 1, n_mels, n_frames)
    Output: logits, shape (batch, num_classes)
    """

    def __init__(self, n_mels: int = 40, num_classes: int = 2) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
