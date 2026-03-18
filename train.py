"""
Training & evaluation script for SPECK 32/64 neural distinguishers.

Trains MLP, CNN, and Siamese models across multiple round counts,
generates accuracy-vs-rounds plots, training curves, and ROC curves.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import generate_data, make_loader
from models import build_model

# ── config ───────────────────────────────────────────────────────────
SEED         = 42
ROUND_COUNTS = [5, 6, 7, 8]
N_TRAIN      = 500_000
N_TEST       = 100_000
BATCH_SIZE   = 5000
EPOCHS       = 30
LR           = 1e-3
PATIENCE     = 5            # early-stopping patience (epochs)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
MODE         = "raw_pairs"  # input representation for all models

MODELS_CFG = {
    # name: input_dim  (depends on MODE)
    "MLP":     64,
    "CNN":     64,
    "Siamese": 64,
}


# ── reproducibility ──────────────────────────────────────────────────
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ── train one model ──────────────────────────────────────────────────
def train_model(model, train_loader, test_loader, epochs=EPOCHS):
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "test_acc": []}
    best_acc = 0.0
    wait = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── train ────────────────────────────────────────────────────
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
        history["train_loss"].append(avg_loss)

        # ── evaluate ─────────────────────────────────────────────────
        acc = evaluate(model, test_loader)
        history["test_acc"].append(acc)

        print(f"  epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  "
              f"test_acc={acc:.4f}")

        # ── early stopping ───────────────────────────────────────────
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  early stop at epoch {epoch}")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)

    return best_acc, history


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = (model(xb) > 0.5).float()
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total


@torch.no_grad()
def get_scores(model, loader):
    """Return (all_labels, all_scores) for ROC/AUC."""
    model.eval()
    labels, scores = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        scores.append(model(xb).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(scores)


# ── plotting helpers ─────────────────────────────────────────────────
def plot_accuracy_vs_rounds(results):
    """Bar + line chart: accuracy vs cipher rounds for each model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"MLP": "o", "CNN": "s", "Siamese": "^"}
    for name in MODELS_CFG:
        accs = [results[(name, nr)] for nr in ROUND_COUNTS]
        ax.plot(ROUND_COUNTS, accs, marker=markers[name],
                linewidth=2, label=name)
    ax.axhline(0.5, color="gray", linestyle="--", label="Random guess")
    ax.set_xlabel("Number of SPECK rounds")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Distinguisher Accuracy vs Cipher Rounds (SPECK 32/64)")
    ax.set_xticks(ROUND_COUNTS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_rounds.png"), dpi=150)
    plt.close(fig)


def plot_training_curves(name, nr, history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss")
    ax1.set_title(f"{name} – {nr} rounds – Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["test_acc"], "r-")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Test Accuracy")
    ax2.set_title(f"{name} – {nr} rounds – Accuracy")
    ax2.axhline(0.5, color="gray", linestyle="--")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"training_{name}_{nr}r.png"), dpi=120)
    plt.close(fig)


def plot_roc_curves(nr, roc_data):
    """One ROC plot per round count with all models overlaid."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves – {nr}-round SPECK 32/64")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"roc_{nr}r.png"), dpi=120)
    plt.close(fig)


def print_summary(results):
    header = f"{'Model':<10}" + "".join(f"{'R='+str(r):>10}" for r in ROUND_COUNTS)
    print("\n" + "=" * len(header))
    print("         FINAL TEST ACCURACY SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in MODELS_CFG:
        row = f"{name:<10}" + "".join(
            f"{results[(name, r)]:>10.4f}" for r in ROUND_COUNTS
        )
        print(row)
    print("=" * len(header))


# ── main ─────────────────────────────────────────────────────────────
def main():
    seed_everything(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}          # (model_name, nr) -> accuracy
    all_histories = {}    # (model_name, nr) -> history dict
    all_roc = {}          # nr -> {model_name: (fpr, tpr, auc)}

    for nr in ROUND_COUNTS:
        print(f"\n{'='*50}")
        print(f"  Generating data for {nr}-round SPECK 32/64")
        print(f"{'='*50}")

        X_train, Y_train = generate_data(N_TRAIN, nr, mode=MODE)
        X_test,  Y_test  = generate_data(N_TEST,  nr, mode=MODE)

        train_loader = make_loader(X_train, Y_train, BATCH_SIZE, shuffle=True)
        test_loader  = make_loader(X_test,  Y_test,  BATCH_SIZE, shuffle=False)

        roc_data_nr = {}

        for name, input_dim in MODELS_CFG.items():
            print(f"\n── Training {name} on {nr} rounds ──")
            model = build_model(name, input_dim)
            best_acc, history = train_model(model, train_loader, test_loader)

            results[(name, nr)] = best_acc
            all_histories[(name, nr)] = history

            # training curve plot
            plot_training_curves(name, nr, history)

            # ROC data
            labels, scores = get_scores(model, test_loader)
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            roc_data_nr[name] = (fpr, tpr, roc_auc)

        all_roc[nr] = roc_data_nr
        plot_roc_curves(nr, roc_data_nr)

    # ── final plots & summary ────────────────────────────────────────
    plot_accuracy_vs_rounds(results)
    print_summary(results)

    # save numeric results
    serializable = {f"{k[0]}_{k[1]}r": v for k, v in results.items()}
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
