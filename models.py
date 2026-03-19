"""
Three neural distinguisher architectures for SPECK 32/64 cryptanalysis.

1. MLPDistinguisher    – 4-layer fully-connected network
2. CNNDistinguisher    – 1-D convolutional network
3. SiameseDistinguisher – twin-branch network with shared weights
"""

import torch
import torch.nn as nn


# ── 1. MLP ───────────────────────────────────────────────────────────
class MLPDistinguisher(nn.Module):
    """Multi-layer perceptron distinguisher."""

    def __init__(self, input_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),          nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── 2. CNN ───────────────────────────────────────────────────────────
class _ResBlock(nn.Module):
    """1-D residual block: two conv layers with a skip connection."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class CNNDistinguisher(nn.Module):
    """
    1-D CNN with residual blocks.

    For raw_pairs (64-bit): reshaped to 2 channels (C0, C1) x 32 bits.
    For xor_diff (32-bit): single channel x 32 bits.
    For concat_xor (96-bit): 3 channels (C0, C1, XOR) x 32 bits.
    """

    def __init__(self, input_dim=64):
        super().__init__()
        self.input_dim = input_dim
        in_channels = input_dim // 32       # 64->2, 32->1, 96->3
        seq_len = 32
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.res1 = _ResBlock(64)
        self.res2 = _ResBlock(64)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * seq_len, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        # reshape to (B, channels, 32)
        x = x.view(x.size(0), -1, 32)
        x = self.initial(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x).squeeze(-1)


# ── 3. Siamese ───────────────────────────────────────────────────────
class SiameseDistinguisher(nn.Module):
    """
    Twin-branch network.  Each branch processes one 32-bit ciphertext;
    the combiner receives [emb(C0), emb(C1), emb(C0) ⊕ emb(C1)].

    Expects input_dim=64 (raw_pairs mode: C0 32 bits ∥ C1 32 bits).
    """

    def __init__(self, input_dim=64):
        super().__init__()
        half = input_dim // 2
        self.branch = nn.Sequential(
            nn.Linear(half, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.ReLU(),
        )
        # combiner: concat of two embeddings + their XOR → 192
        self.combiner = nn.Sequential(
            nn.Linear(192, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 1),    nn.Sigmoid(),
        )

    def forward(self, x):
        half = x.shape[1] // 2
        c0 = x[:, :half]
        c1 = x[:, half:]
        e0 = self.branch(c0)
        e1 = self.branch(c1)
        diff = torch.abs(e0 - e1)       # learned "difference"
        combined = torch.cat([e0, e1, diff], dim=1)
        return self.combiner(combined).squeeze(-1)


# ── factory ──────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "MLP":     MLPDistinguisher,
    "CNN":     CNNDistinguisher,
    "Siamese": SiameseDistinguisher,
}


def build_model(name, input_dim):
    return MODEL_REGISTRY[name](input_dim=input_dim)


# ── quick shape check ────────────────────────────────────────────────
if __name__ == "__main__":
    for name, cls in MODEL_REGISTRY.items():
        m = cls(input_dim=64)
        dummy = torch.randn(8, 64)
        out = m(dummy)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{name:10s}  output shape={out.shape}  params={n_params:,}")
