"""
SPECK 32/64 block cipher — vectorized NumPy implementation.

Block size : 32 bits (two 16-bit words)
Key size   : 64 bits (four 16-bit words)
Rounds     : 22 (default)
Rotation   : alpha=7 (right), beta=2 (left)

Reference test vector (from NSA spec):
  plaintext  = (0x6574, 0x694c)
  key        = (0x1918, 0x1110, 0x0908, 0x0100)
  ciphertext = (0xa868, 0x42f2)
"""

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────
def _u16(x):
    return np.asarray(x, dtype=np.uint16)


def rol16(x, r):
    """Left-rotate 16-bit words by *r* bits."""
    x = _u16(x)
    return ((x << r) | (x >> (16 - r))) & np.uint16(0xFFFF)


def ror16(x, r):
    """Right-rotate 16-bit words by *r* bits."""
    x = _u16(x)
    return ((x >> r) | (x << (16 - r))) & np.uint16(0xFFFF)


# ── one SPECK round ─────────────────────────────────────────────────
def speck_round(x, y, k):
    """Apply one SPECK round.  x, y, k are uint16 arrays of shape (N,)."""
    x = _u16(ror16(x, 7))
    x = _u16((x.astype(np.uint32) + y.astype(np.uint32)) & 0xFFFF)
    x = _u16(x ^ k)
    y = _u16(rol16(y, 2))
    y = _u16(y ^ x)
    return x, y


# ── key schedule ─────────────────────────────────────────────────────
def speck_key_schedule(key, rounds=22):
    """
    Parameters
    ----------
    key : ndarray, shape (N, 4), dtype uint16
        Four key words per sample.  Column order: key[:,0] is the
        most-significant word, key[:,3] is the least-significant.
    rounds : int

    Returns
    -------
    rk : list[ndarray]   –  *rounds* arrays, each shape (N,).
    """
    N = key.shape[0]
    # initial round key and l-values
    k = [_u16(key[:, 3])]                         # k[0]
    l = [_u16(key[:, 2]),                          # l[0]
         _u16(key[:, 1]),                          # l[1]
         _u16(key[:, 0])]                          # l[2]

    for i in range(rounds - 1):
        li = l[i]
        ki = k[i]
        # l[i+m-1], k[i+1] = speck_round(l[i], k[i], i)
        new_l, new_k = speck_round(li, ki, _u16(np.full(N, i, dtype=np.uint16)))
        l.append(new_l)
        k.append(new_k)

    return k                                       # length == rounds


# ── encryption ───────────────────────────────────────────────────────
def speck_encrypt(plaintext, key, rounds=22):
    """
    Parameters
    ----------
    plaintext : ndarray, shape (N, 2), dtype uint16
        plaintext[:,0] = left word, plaintext[:,1] = right word.
    key : ndarray, shape (N, 4), dtype uint16
    rounds : int

    Returns
    -------
    ciphertext : ndarray, shape (N, 2), dtype uint16
    """
    rk = speck_key_schedule(key, rounds)

    x = _u16(plaintext[:, 0])   # left
    y = _u16(plaintext[:, 1])   # right

    for i in range(rounds):
        x, y = speck_round(x, y, rk[i])

    ct = np.stack([x, y], axis=1)
    return ct


# ── quick self-test against the NSA test vector ──────────────────────
if __name__ == "__main__":
    pt  = np.array([[0x6574, 0x694c]], dtype=np.uint16)
    k   = np.array([[0x1918, 0x1110, 0x0908, 0x0100]], dtype=np.uint16)
    ct  = speck_encrypt(pt, k, rounds=22)
    assert ct[0, 0] == 0xa868 and ct[0, 1] == 0x42f2, \
        f"Test vector FAILED: got {ct[0]}"
    print("SPECK 32/64 test vector PASSED")
