# model_resnet1d.py
# Drop-in replacement for your SmallCNN:
# Input:  x shape (batch, nbins)  (e.g., (64, 512))
# Output: logits shape (batch,)   (use BCEWithLogitsLoss)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # If shape changes (channels or stride), match residual path
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(
        self,
        nbins: int = 512,
        base_ch: int = 32,
        blocks=(2, 2, 2),   # number of residual blocks per stage
        dropout: float = 0.1
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
        )

        # Stages (downsample by stride=2 at each stage start)
        ch1 = base_ch
        ch2 = base_ch * 2
        ch3 = base_ch * 4

        self.stage1 = self._make_stage(ch1, ch1, n_blocks=blocks[0], stride=1, dropout=dropout)
        self.stage2 = self._make_stage(ch1, ch2, n_blocks=blocks[1], stride=2, dropout=dropout)
        self.stage3 = self._make_stage(ch2, ch3, n_blocks=blocks[2], stride=2, dropout=dropout)

        # Pool + head
        self.pool = nn.AdaptiveAvgPool1d(1)   # -> (B, ch3, 1)
        self.fc = nn.Linear(ch3, 1)

    def _make_stage(self, in_ch, out_ch, n_blocks, stride, dropout):
        layers = [ResBlock1D(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(n_blocks - 1):
            layers.append(ResBlock1D(out_ch, out_ch, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, L) -> (B, 1, L)
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (batch, nbins). Got {tuple(x.shape)}")
        x = x.unsqueeze(1)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.pool(x).squeeze(-1)    # (B, ch3)
        logit = self.fc(x).squeeze(-1)  # (B,)
        return logit


def build_model(nbins: int = 512):
    return ResNet1D(nbins=nbins)