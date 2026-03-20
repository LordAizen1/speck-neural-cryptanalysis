"""
Combined bonus experiments:
  1. Automatic search for effective input differences
  2. Classical vs ML-based differential distinguisher comparison
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import generate_data, make_loader, to_bits
from speck import speck_encrypt
from models import build_model

# ── config ───────────────────────────────────────────────────────────
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# =====================================================================
#  PART 1: AUTOMATIC INPUT DIFFERENCE SEARCH
# =====================================================================

DELTA_CANDIDATES = [
    (0x0040, 0x0000),   # Gohr's original
    (0x0080, 0x0000),
    (0x0020, 0x0000),
    (0x0001, 0x0000),
    (0x8000, 0x0000),
    (0x0000, 0x0040),
    (0x0000, 0x0001),
    (0x0040, 0x0040),
    (0x0060, 0x0000),
    (0x8004, 0x0000),
    (0x0100, 0x0000),
    (0x0400, 0x0000),
]


def generate_data_custom_delta(n_samples, nr_rounds, delta):
    """Generate xor_diff data with a custom input difference."""
    n_real = n_samples // 2
    n_random = n_samples - n_real

    P0 = np.random.randint(0, 0x10000, size=(n_real, 2), dtype=np.uint16)
    P1 = P0.copy()
    P1[:, 0] ^= np.uint16(delta[0])
    P1[:, 1] ^= np.uint16(delta[1])
    K = np.random.randint(0, 0x10000, size=(n_real, 4), dtype=np.uint16)
    C0 = speck_encrypt(P0, K, rounds=nr_rounds)
    C1 = speck_encrypt(P1, K, rounds=nr_rounds)

    C0_rand = np.random.randint(0, 0x10000, size=(n_random, 2), dtype=np.uint16)
    C1_rand = np.random.randint(0, 0x10000, size=(n_random, 2), dtype=np.uint16)

    C0_all = np.concatenate([C0, C0_rand], axis=0)
    C1_all = np.concatenate([C1, C1_rand], axis=0)
    Y = np.concatenate([np.ones(n_real), np.zeros(n_random)]).astype(np.float32)

    diff_l = to_bits(C0_all[:, 0] ^ C1_all[:, 0])
    diff_r = to_bits(C0_all[:, 1] ^ C1_all[:, 1])
    X = np.concatenate([diff_l, diff_r], axis=1)

    idx = np.random.permutation(len(Y))
    return X[idx], Y[idx]


def train_quick(delta, nr_rounds=6, n_train=200_000, n_test=50_000,
                epochs=20, patience=5):
    """Train a CNN quickly on the given delta and return accuracy."""
    X_train, Y_train = generate_data_custom_delta(n_train, nr_rounds, delta)
    X_test,  Y_test  = generate_data_custom_delta(n_test,  nr_rounds, delta)
    train_loader = make_loader(X_train, Y_train, 5000, shuffle=True)
    test_loader  = make_loader(X_test,  Y_test,  5000, shuffle=False)

    model = build_model("CNN", 32).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_acc = 0.0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += ((model(xb) > 0.5).float() == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        print(f"    epoch {epoch:2d}/{epochs}  acc={acc:.4f}", flush=True)

        if acc > best_acc:
            best_acc = acc
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return best_acc


def plot_delta_results(results):
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    labels = [f"0x{d[0]:04X}/0x{d[1]:04X}" for d, _ in sorted_results]
    accs = [a for _, a in sorted_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if a == max(accs) else "#3498db" for a in accs]
    bars = ax.barh(range(len(labels)), accs, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontfamily="monospace")
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Input Difference Search - 6-round SPECK 32/64 (CNN + xor_diff)")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(0.45, max(accs) + 0.03)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{acc:.2%}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "delta_search.png"), dpi=150)
    plt.close(fig)


def run_delta_search():
    print("\n" + "#" * 60, flush=True)
    print("  BONUS 1: AUTOMATIC INPUT DIFFERENCE SEARCH", flush=True)
    print("#" * 60, flush=True)

    results = []
    for i, delta in enumerate(DELTA_CANDIDATES):
        print(f"\n[{i+1}/{len(DELTA_CANDIDATES)}] delta = "
              f"(0x{delta[0]:04X}, 0x{delta[1]:04X})", flush=True)
        acc = train_quick(delta)
        results.append((delta, acc))
        print(f"  Best accuracy: {acc:.4f}", flush=True)

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 50, flush=True)
    print("    INPUT DIFFERENCE SEARCH RESULTS", flush=True)
    print("=" * 50, flush=True)
    print(f"  {'Rank':<6}{'Delta':>18}{'Accuracy':>12}", flush=True)
    print("  " + "-" * 34, flush=True)
    for rank, (delta, acc) in enumerate(sorted_results, 1):
        marker = " <-- best" if rank == 1 else ""
        print(f"  {rank:<6}(0x{delta[0]:04X}, 0x{delta[1]:04X})"
              f"{acc:>11.4f}{marker}", flush=True)
    print("=" * 50, flush=True)

    plot_delta_results(results)
    delta_json = {f"0x{d[0]:04X}_0x{d[1]:04X}": a for d, a in results}
    with open(os.path.join(RESULTS_DIR, "delta_search_results.json"), "w") as f:
        json.dump(delta_json, f, indent=2)

    return results


# =====================================================================
#  PART 2: CLASSICAL VS ML DIFFERENTIAL DISTINGUISHER
# =====================================================================

def classical_distinguisher_train(nr_rounds, n_profile=500_000,
                                  delta=(0x0040, 0x0000)):
    """
    Build a classical frequency-based distinguisher.

    Profile phase: encrypt many pairs under random keys, record the
    frequency of each 32-bit output difference delta_C.
    Returns: a set of 'high-frequency' output differences (those that
    appear more often than expected under uniform distribution).
    """
    P0 = np.random.randint(0, 0x10000, size=(n_profile, 2), dtype=np.uint16)
    P1 = P0.copy()
    P1[:, 0] ^= np.uint16(delta[0])
    P1[:, 1] ^= np.uint16(delta[1])
    K = np.random.randint(0, 0x10000, size=(n_profile, 4), dtype=np.uint16)

    C0 = speck_encrypt(P0, K, rounds=nr_rounds)
    C1 = speck_encrypt(P1, K, rounds=nr_rounds)

    # Compute 32-bit output differences as single uint32
    delta_c = (C0[:, 0].astype(np.uint32) << 16 | C0[:, 1].astype(np.uint32)) ^ \
              (C1[:, 0].astype(np.uint32) << 16 | C1[:, 1].astype(np.uint32))

    # Count frequencies
    unique, counts = np.unique(delta_c, return_counts=True)
    expected = n_profile / (2**32)   # expected count under uniform

    # Keep differences that appear significantly more than expected
    # Use threshold: appears at least 3x the expected frequency
    threshold = max(3 * expected, 2)  # at least 2 occurrences
    high_freq = set(unique[counts >= threshold])

    # Also compute the overall bias (total variation distance)
    freq = counts / n_profile
    uniform = 1.0 / (2**32)
    bias = np.sum(np.abs(freq - uniform)) / 2

    return high_freq, bias


def classical_distinguisher_test(high_freq_set, nr_rounds, n_test=100_000,
                                  delta=(0x0040, 0x0000)):
    """
    Test classical distinguisher accuracy.

    For each sample: generate a cipher pair or random pair.
    Classify as cipher if delta_C is in the high-frequency set.
    """
    n_real = n_test // 2
    n_random = n_test - n_real

    # Real pairs
    P0 = np.random.randint(0, 0x10000, size=(n_real, 2), dtype=np.uint16)
    P1 = P0.copy()
    P1[:, 0] ^= np.uint16(delta[0])
    P1[:, 1] ^= np.uint16(delta[1])
    K = np.random.randint(0, 0x10000, size=(n_real, 4), dtype=np.uint16)
    C0_real = speck_encrypt(P0, K, rounds=nr_rounds)
    C1_real = speck_encrypt(P1, K, rounds=nr_rounds)

    dc_real = (C0_real[:, 0].astype(np.uint32) << 16 | C0_real[:, 1].astype(np.uint32)) ^ \
              (C1_real[:, 0].astype(np.uint32) << 16 | C1_real[:, 1].astype(np.uint32))

    # Random pairs
    C0_rand = np.random.randint(0, 0x10000, size=(n_random, 2), dtype=np.uint16)
    C1_rand = np.random.randint(0, 0x10000, size=(n_random, 2), dtype=np.uint16)

    dc_rand = (C0_rand[:, 0].astype(np.uint32) << 16 | C0_rand[:, 1].astype(np.uint32)) ^ \
              (C1_rand[:, 0].astype(np.uint32) << 16 | C1_rand[:, 1].astype(np.uint32))

    # Classify: predict cipher if delta_C is in high_freq_set
    real_correct = sum(1 for dc in dc_real if dc in high_freq_set)
    rand_correct = sum(1 for dc in dc_rand if dc not in high_freq_set)

    accuracy = (real_correct + rand_correct) / n_test
    return accuracy


def train_ml_for_comparison(nr_rounds, n_train=500_000, n_test=100_000):
    """Train CNN with xor_diff for comparison."""
    X_train, Y_train = generate_data(n_train, nr_rounds, mode="xor_diff")
    X_test,  Y_test  = generate_data(n_test,  nr_rounds, mode="xor_diff")
    train_loader = make_loader(X_train, Y_train, 5000, shuffle=True)
    test_loader  = make_loader(X_test,  Y_test,  5000, shuffle=False)

    model = build_model("CNN", 32).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    best_acc = 0.0
    wait = 0
    best_state = None

    for epoch in range(1, 41):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += ((model(xb) > 0.5).float() == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        print(f"    epoch {epoch:2d}/40  acc={acc:.4f}", flush=True)

        if acc > best_acc:
            best_acc = acc
            wait = 0
        else:
            wait += 1
            if wait >= 7:
                break

    return best_acc


def plot_classical_vs_ml(round_counts, classical_accs, ml_accs):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(round_counts))
    width = 0.35

    bars1 = ax.bar(x - width/2, classical_accs, width, label="Classical (frequency-based)",
                   color="#e74c3c")
    bars2 = ax.bar(x + width/2, ml_accs, width, label="ML (CNN + xor_diff)",
                   color="#3498db")

    ax.set_xlabel("Number of SPECK rounds")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Classical vs ML Differential Distinguisher (SPECK 32/64)")
    ax.set_xticks(x)
    ax.set_xticklabels(round_counts)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.45, 1.05)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "classical_vs_ml.png"), dpi=150)
    plt.close(fig)


def run_classical_vs_ml():
    print("\n" + "#" * 60, flush=True)
    print("  BONUS 2: CLASSICAL VS ML DISTINGUISHER", flush=True)
    print("#" * 60, flush=True)

    round_counts = [3, 4, 5, 6]
    classical_accs = []
    ml_accs = []

    for nr in round_counts:
        print(f"\n{'='*50}", flush=True)
        print(f"  {nr}-round SPECK", flush=True)
        print(f"{'='*50}", flush=True)

        # Classical
        print("\n  -- Classical distinguisher --", flush=True)
        print("  Profiling...", flush=True)
        high_freq, bias = classical_distinguisher_train(nr, n_profile=500_000)
        print(f"  Found {len(high_freq)} high-frequency differences, "
              f"bias={bias:.6f}", flush=True)
        print("  Testing...", flush=True)
        c_acc = classical_distinguisher_test(high_freq, nr, n_test=100_000)
        classical_accs.append(c_acc)
        print(f"  Classical accuracy: {c_acc:.4f}", flush=True)

        # ML
        print("\n  -- ML distinguisher (CNN + xor_diff) --", flush=True)
        m_acc = train_ml_for_comparison(nr)
        ml_accs.append(m_acc)
        print(f"  ML accuracy: {m_acc:.4f}", flush=True)

    # Summary
    print("\n" + "=" * 55, flush=True)
    print("    CLASSICAL VS ML COMPARISON", flush=True)
    print("=" * 55, flush=True)
    print(f"  {'Rounds':<10}{'Classical':>14}{'ML (CNN)':>14}{'ML Advantage':>14}",
          flush=True)
    print("  " + "-" * 50, flush=True)
    for nr, c_acc, m_acc in zip(round_counts, classical_accs, ml_accs):
        adv = m_acc - c_acc
        print(f"  {nr:<10}{c_acc:>13.4f}{m_acc:>13.4f}{adv:>+13.4f}", flush=True)
    print("=" * 55, flush=True)

    plot_classical_vs_ml(round_counts, classical_accs, ml_accs)
    results = {
        "rounds": round_counts,
        "classical": classical_accs,
        "ml_cnn": ml_accs,
    }
    with open(os.path.join(RESULTS_DIR, "classical_vs_ml_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# =====================================================================
#  MAIN
# =====================================================================

def main():
    seed_everything(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    run_delta_search()
    run_classical_vs_ml()

    print("\nAll bonus experiments complete!", flush=True)
    print(f"Plots saved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
