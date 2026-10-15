# sleepcl/staging/torch_eegnet.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EEGNetConfig:
    n_ch: int
    n_samp: int
    n_classes: int
    # EEGNet-ish hyperparams
    F1: int = 8
    D: int = 2
    F2: int = 16
    k1: int = 64
    k2: int = 16
    pool1: int = 4
    pool2: int = 8
    dropout: float = 0.25


class EEGNetSmall(nn.Module):
    """
    EEGNet-like compact CNN.
    Input: (B, C, T) in float32 (e.g., uV-scaled & normalised)
    Internally uses (B, 1, C, T) for Conv2d.
    """
    def __init__(self, cfg: EEGNetConfig):
        super().__init__()
        self.cfg = cfg

        F1, D, F2 = cfg.F1, cfg.D, cfg.F2
        k1, k2 = cfg.k1, cfg.k2

        # Temporal conv (1 x k1)
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, k1),
            padding=(0, k1 // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise spatial conv across channels (C x 1), groups=F1
        self.conv_spatial = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(cfg.n_ch, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)

        self.pool1 = nn.AvgPool2d(kernel_size=(1, cfg.pool1))
        self.drop1 = nn.Dropout(cfg.dropout)

        # Separable conv: depthwise temporal then pointwise
        self.sep_depth = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, k2),
            padding=(0, k2 // 2),
            groups=F1 * D,
            bias=False,
        )
        self.sep_point = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(F2)

        self.pool2 = nn.AvgPool2d(kernel_size=(1, cfg.pool2))
        self.drop2 = nn.Dropout(cfg.dropout)

        # Figure out classifier input dim dynamically
        with torch.no_grad():
            x = torch.zeros(1, cfg.n_ch, cfg.n_samp)  # (B, C, T)
            y = self._forward_features(x)
            feat_dim = y.shape[-1]
        self.classifier = nn.Linear(feat_dim, cfg.n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)

        x = self.conv_temporal(x)
        x = self.bn1(x)

        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = F.elu(x)

        x = self.pool2(x)
        x = self.drop2(x)

        # flatten: (B, F, 1, T') -> (B, F*T')
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._forward_features(x)
        logits = self.classifier(feats)
        return logits
