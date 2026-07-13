"""
CosmiKAI — TransitCNN Model Definition
========================================
1-D Convolutional Neural Network designed for edge deployment onboard a
CubeSat-class satellite. The architecture is intentionally compact (~72K
parameters, 288 KB serialised) while achieving 81.17% AUPRC on real
confirmed Kepler transit data.

Input
-----
A phase-folded, median-binned, standardised light curve of shape (B, L),
where L = 512 by default.

Output
------
Raw logit of shape (B,). Apply ``torch.sigmoid`` to obtain a probability
in [0, 1]. Scores >= the configured threshold are classified as
TRANSIT_DETECTED.

Architecture summary
--------------------
    Conv1d(1->32, k=7)  -> ReLU                  # coarse edge/slope features
    Conv1d(32->64, k=7) -> ReLU                  # mid-level dip patterns
    Conv1d(64->128, k=5) -> ReLU                 # high-level transit shape
    AdaptiveAvgPool1d(1)                          # collapse to (B, 128, 1)
    Flatten -> Linear(128->128) -> ReLU -> Dropout(0.3) -> Linear(128->1)

Usage
-----
    from backened.model.src.model import TransitCNN
    import torch

    model = TransitCNN()
    model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logit = model(x)            # x: (B, 512)
        score = torch.sigmoid(logit)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransitCNN(nn.Module):
    """
    Lightweight 1-D CNN for exoplanet transit classification.

    Parameters
    ----------
    dropout : float
        Dropout probability before the final classification layer.
        Default 0.3 to prevent overfitting on the moderate-sized
        training set (~3,700 samples).

    Notes
    -----
    Layer names (``conv``, ``fc``) must match the trained weight file.
    Do NOT rename these without retraining the model.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        # ── Feature extraction backbone ──────────────────────────────
        # Three Conv1d blocks with increasing filter counts extract
        # progressively higher-level features from the 1D light curve.
        # AdaptiveAvgPool collapses the sequence dimension so the
        # classifier input is independent of the original length.
        self.conv = nn.Sequential(
            # Block 1 — detect simple edges and slopes in the light curve
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),

            # Block 2 — capture transit dip shapes and broader patterns
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),

            # Block 3 — recognise compound transit features
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),

            # Global average pooling — fixed-size output regardless of input length
            nn.AdaptiveAvgPool1d(1),
        )

        # ── Classification head ──────────────────────────────────────
        # Maps the 128-dim feature vector to a single transit/no-transit logit.
        # Dropout reduces overfitting given the moderate dataset size.
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, L)`` or ``(B, 1, L)`` — batch of standardised
            phase-folded light curves.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)`` — raw logits. Apply ``torch.sigmoid`` for
            classification probabilities.
        """
        # Add channel dimension if input is (B, L) instead of (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)         # (B, L) -> (B, 1, L)

        features = self.conv(x)         # (B, 128, 1)
        return self.fc(features).squeeze(1)  # (B,)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_model(weights_path: str, device: str = "cpu") -> TransitCNN:
    """
    Convenience function to load a trained TransitCNN from disk.

    Parameters
    ----------
    weights_path : str
        Path to the ``.pt`` state-dict file produced during training.
    device : str
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    TransitCNN
        Model in eval mode with loaded weights.
    """
    model = TransitCNN()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model