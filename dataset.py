"""
Dataset generation for the SPECK 32/64 neural distinguisher.

Generates labeled (X, Y) pairs:
  Y = 1  →  ciphertext pair from SPECK with a fixed plaintext XOR difference
  Y = 0  →  two independent random 32-bit values (random permutation baseline)

Supports multiple input representations:
  - raw_pairs   : (C0_L, C0_R, C1_L, C1_R) as 64 bits
  - xor_diff    : (C0 ⊕ C1)                 as 32 bits
  - concat_xor  : raw_pairs ∥ xor_diff       as 96 bits
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from speck import speck_encrypt

# Fixed input difference  (Gohr's choice for SPECK 32/64)
DELTA_LEFT  = np.uint16(0x0040)
DELTA_RIGHT = np.uint16(0x0000)


# ── bit conversion ───────────────────────────────────────────────────
def to_bits(arr, nbits=16):
    """
    Convert a uint16 array of shape (N,) to a float32 binary array (N, nbits).
    MSB first.
    """
    arr = arr.astype(np.uint16)
    shifts = np.arange(nbits - 1, -1, -1, dtype=np.uint16)
    return ((arr[:, None] >> shifts) & 1).astype(np.float32)


# ── core generation ──────────────────────────────────────────────────
def generate_data(n_samples, nr_rounds, mode="raw_pairs",
                  delta=(DELTA_LEFT, DELTA_RIGHT)):
    """
    Parameters
    ----------
    n_samples  : int   – total number of samples (half real, half random)
    nr_rounds  : int   – number of SPECK rounds
    mode       : str   – 'raw_pairs' | 'xor_diff' | 'concat_xor'
    delta      : tuple – (delta_left, delta_right) uint16

    Returns
    -------
    X : np.ndarray, float32, shape (n_samples, D)
    Y : np.ndarray, float32, shape (n_samples,)
    """
    n_real   = n_samples // 2
    n_random = n_samples - n_real

    # ── real cipher pairs (label 1) ──────────────────────────────────
    P0 = np.random.randint(0, 0x10000, size=(n_real, 2), dtype=np.uint16)
    P1 = P0.copy()
    P1[:, 0] ^= delta[0]
    P1[:, 1] ^= delta[1]

    K = np.random.randint(0, 0x10000, size=(n_real, 4), dtype=np.uint16)

    C0 = speck_encrypt(P0, K, rounds=nr_rounds)
    C1 = speck_encrypt(P1, K, rounds=nr_rounds)

    # ── random pairs (label 0) ───────────────────────────────────────
    C0_rand = np.random.randint(0, 0x10000, size=(n_random, 2), dtype=np.uint16)
    C1_rand = np.random.randint(0, 0x10000, size=(n_random, 2), dtype=np.uint16)

    # ── stack and label ──────────────────────────────────────────────
    C0_all = np.concatenate([C0, C0_rand], axis=0)
    C1_all = np.concatenate([C1, C1_rand], axis=0)
    Y = np.concatenate([np.ones(n_real), np.zeros(n_random)]).astype(np.float32)

    # ── build input representation ───────────────────────────────────
    X = _build_representation(C0_all, C1_all, mode)

    # ── shuffle ──────────────────────────────────────────────────────
    idx = np.random.permutation(len(Y))
    return X[idx], Y[idx]


def _build_representation(C0, C1, mode):
    """Convert uint16 ciphertext arrays to a float32 bit matrix."""
    bits_c0_l = to_bits(C0[:, 0])   # (N, 16)
    bits_c0_r = to_bits(C0[:, 1])
    bits_c1_l = to_bits(C1[:, 0])
    bits_c1_r = to_bits(C1[:, 1])

    if mode == "raw_pairs":
        return np.concatenate([bits_c0_l, bits_c0_r,
                               bits_c1_l, bits_c1_r], axis=1)   # (N, 64)

    elif mode == "xor_diff":
        diff_l = to_bits(C0[:, 0] ^ C1[:, 0])
        diff_r = to_bits(C0[:, 1] ^ C1[:, 1])
        return np.concatenate([diff_l, diff_r], axis=1)          # (N, 32)

    elif mode == "concat_xor":
        diff_l = to_bits(C0[:, 0] ^ C1[:, 0])
        diff_r = to_bits(C0[:, 1] ^ C1[:, 1])
        return np.concatenate([bits_c0_l, bits_c0_r,
                               bits_c1_l, bits_c1_r,
                               diff_l, diff_r], axis=1)         # (N, 96)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── PyTorch helpers ──────────────────────────────────────────────────
def make_loader(X, Y, batch_size=5000, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ── quick sanity check ───────────────────────────────────────────────
if __name__ == "__main__":
    for m in ("raw_pairs", "xor_diff", "concat_xor"):
        X, Y = generate_data(1000, nr_rounds=5, mode=m)
        print(f"mode={m:12s}  X.shape={str(X.shape):12s}  "
              f"Y mean={Y.mean():.2f}")
