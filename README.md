# Neural Cryptanalysis of SPECK 32/64

ML-based distinguishers that learn to tell apart reduced-round SPECK 32/64 ciphertext pairs from random permutations, inspired by [Gohr 2019](https://doi.org/10.1007/978-3-030-26948-7_15).

## Overview

Given a pair of plaintexts with a fixed XOR difference `ΔP = (0x0040, 0x0000)`, the cipher produces ciphertext pairs `(C, C')` with subtle statistical biases that decay as rounds increase. A random permutation produces no such bias. Neural networks can learn to detect this.

## Models

Three architectures are compared across 5–8 cipher rounds:

| Model | Description | Params |
|-------|-------------|--------|
| **MLP** | 4-layer fully connected network | ~115K |
| **CNN** | 1D convolutional network over bit vectors | ~17K |
| **Siamese** | Twin-branch network with shared weights | ~50K |

## Project Structure

```
speck.py          # SPECK 32/64 cipher (vectorized NumPy)
dataset.py        # Data generation (cipher pairs vs random)
models.py         # MLP, CNN, Siamese architectures (PyTorch)
train.py          # Training loop, evaluation, plotting
requirements.txt  # Dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

```bash
python train.py
```

Trains all 3 models on 5, 6, 7, and 8-round SPECK, then saves plots and results to `results/`.

## Input Representations

- **raw_pairs** — `(C₀, C₁)` as 64 bits (default)
- **xor_diff** — `C₀ ⊕ C₁` as 32 bits
- **concat_xor** — raw pairs + XOR difference as 96 bits
