"""
Representation comparison experiment.

Trains all 3 models across all 3 input representations at rounds 4, 5, 6
to study how representation choice affects distinguisher performance.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import generate_data, make_loader
from models import build_model

# ── config ───────────────────────────────────────────────────────────
SEED         = 42
ROUND_COUNTS = [4, 5, 6]
REPR_MODES   = {
    "raw_pairs":  64,   # (C0 || C1) as 64 bits
    "xor_diff":   32,   # (C0 XOR C1) as 32 bits
    "concat_xor": 96,   # (C0 || C1 || C0 XOR C1) as 96 bits
}
MODEL_NAMES  = ["MLP", "CNN", "Siamese"]
N_TRAIN      = 500_000
N_TEST       = 100_000
BATCH_SIZE   = 5000
EPOCHS       = 40
LR           = 1e-3
PATIENCE     = 7
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")


# ── reproducibility ──────────────────────────────────────────────────
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ── train one model ──────────────────────────────────────────────────
def train_model(model, train_loader, test_loader):
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    wait = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_loss = running_loss / n_batches

        # evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = (model(xb) > 0.5).float()
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total

        print(f"  epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"test_acc={acc:.4f}", flush=True)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


# ── plotting ─────────────────────────────────────────────────────────
def plot_repr_comparison(results):
    """Grouped bar chart: for each round, bars for each (model, repr) combo."""
    fig, axes = plt.subplots(1, len(ROUND_COUNTS), figsize=(5 * len(ROUND_COUNTS), 5),
                             sharey=True)
    if len(ROUND_COUNTS) == 1:
        axes = [axes]

    colors = {"raw_pairs": "#4C72B0", "xor_diff": "#DD8452", "concat_xor": "#55A868"}
    bar_width = 0.25

    for ax, nr in zip(axes, ROUND_COUNTS):
        x = np.arange(len(MODEL_NAMES))
        for i, (mode, _) in enumerate(REPR_MODES.items()):
            accs = [results.get((name, mode, nr), 0.5) for name in MODEL_NAMES]
            ax.bar(x + i * bar_width, accs, bar_width,
                   label=mode, color=colors[mode])

        ax.set_xlabel("Model")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"{nr}-round SPECK")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(MODEL_NAMES)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Impact of Input Representation on Distinguisher Accuracy",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "repr_comparison.png"), dpi=150)
    plt.close(fig)


def plot_repr_by_model(results):
    """One line plot per model: accuracy vs rounds for each representation."""
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(5 * len(MODEL_NAMES), 4),
                             sharey=True)
    markers = {"raw_pairs": "o", "xor_diff": "s", "concat_xor": "^"}
    colors = {"raw_pairs": "#4C72B0", "xor_diff": "#DD8452", "concat_xor": "#55A868"}

    for ax, name in zip(axes, MODEL_NAMES):
        for mode in REPR_MODES:
            accs = [results.get((name, mode, nr), 0.5) for nr in ROUND_COUNTS]
            ax.plot(ROUND_COUNTS, accs, marker=markers[mode], color=colors[mode],
                    linewidth=2, label=mode)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(name)
        ax.set_xticks(ROUND_COUNTS)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Representation Impact per Model", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "repr_by_model.png"), dpi=150)
    plt.close(fig)


def print_repr_summary(results):
    print("\n" + "=" * 70, flush=True)
    print("    REPRESENTATION COMPARISON SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for nr in ROUND_COUNTS:
        print(f"\n  {nr}-round SPECK:", flush=True)
        header = f"  {'Model':<10}" + "".join(f"{m:>14}" for m in REPR_MODES)
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)
        for name in MODEL_NAMES:
            row = f"  {name:<10}" + "".join(
                f"{results.get((name, mode, nr), 0.5):>14.4f}" for mode in REPR_MODES
            )
            print(row, flush=True)
    print("=" * 70, flush=True)


# ── main ─────────────────────────────────────────────────────────────
def main():
    seed_everything(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}   # (model_name, mode, nr) -> accuracy

    for nr in ROUND_COUNTS:
        for mode, input_dim in REPR_MODES.items():
            print(f"\n{'='*50}", flush=True)
            print(f"  {nr}-round SPECK | repr={mode} (dim={input_dim})",
                  flush=True)
            print(f"{'='*50}", flush=True)

            X_train, Y_train = generate_data(N_TRAIN, nr, mode=mode)
            X_test,  Y_test  = generate_data(N_TEST,  nr, mode=mode)
            train_loader = make_loader(X_train, Y_train, BATCH_SIZE, shuffle=True)
            test_loader  = make_loader(X_test,  Y_test,  BATCH_SIZE, shuffle=False)

            for name in MODEL_NAMES:
                # Siamese needs paired input (raw_pairs or concat_xor)
                # For xor_diff, Siamese splits 32 bits into 2x16 — still works
                print(f"\n-- {name} --", flush=True)
                model = build_model(name, input_dim)
                best_acc = train_model(model, train_loader, test_loader)
                results[(name, mode, nr)] = best_acc

    # plots & summary
    plot_repr_comparison(results)
    plot_repr_by_model(results)
    print_repr_summary(results)

    # save results
    serializable = {f"{k[0]}_{k[1]}_{k[2]}r": v for k, v in results.items()}
    with open(os.path.join(RESULTS_DIR, "repr_results.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nPlots saved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
