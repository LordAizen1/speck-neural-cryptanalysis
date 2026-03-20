"""
Transfer learning experiment.

Tests whether a model trained on round N can be fine-tuned to improve
performance on round N+1 compared to training from scratch.

Uses the CNN (best model) with xor_diff (best representation).
"""

import os
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
SEED        = 42
N_TRAIN     = 500_000
N_TEST      = 100_000
BATCH_SIZE  = 5000
EPOCHS      = 40
LR          = 1e-3
LR_FINETUNE = 3e-4       # lower LR for fine-tuning
PATIENCE    = 7
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODE        = "xor_diff"
INPUT_DIM   = 32
MODEL_NAME  = "CNN"

# Transfer pairs: train on source, fine-tune on target
TRANSFER_PAIRS = [
    (4, 5),
    (5, 6),
    (4, 6),
]


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_model(model, train_loader, test_loader, epochs, lr):
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "test_acc": []}
    best_acc = 0.0
    wait = 0
    best_state = None

    for epoch in range(1, epochs + 1):
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

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = (model(xb) > 0.5).float()
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        history["test_acc"].append(acc)

        print(f"  epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  "
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
    model.to(DEVICE)
    return best_acc, history


def plot_transfer_comparison(all_results):
    fig, axes = plt.subplots(1, len(TRANSFER_PAIRS), figsize=(5 * len(TRANSFER_PAIRS), 4.5),
                             sharey=True)
    if len(TRANSFER_PAIRS) == 1:
        axes = [axes]

    for ax, (src, tgt) in zip(axes, TRANSFER_PAIRS):
        scratch_hist = all_results[(src, tgt)]["scratch"]["history"]
        transfer_hist = all_results[(src, tgt)]["transfer"]["history"]

        epochs_s = range(1, len(scratch_hist["test_acc"]) + 1)
        epochs_t = range(1, len(transfer_hist["test_acc"]) + 1)

        ax.plot(epochs_s, scratch_hist["test_acc"], "b-o", markersize=3,
                label=f"From scratch ({all_results[(src,tgt)]['scratch']['acc']:.4f})")
        ax.plot(epochs_t, transfer_hist["test_acc"], "r-s", markersize=3,
                label=f"Transfer R{src}->R{tgt} ({all_results[(src,tgt)]['transfer']['acc']:.4f})")

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Target: {tgt}-round SPECK")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Transfer Learning: {MODEL_NAME} with {MODE}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "transfer_learning.png"), dpi=150)
    plt.close(fig)


def print_transfer_summary(all_results):
    print("\n" + "=" * 60, flush=True)
    print("    TRANSFER LEARNING SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  {'Transfer':>15}  {'From Scratch':>14}  {'Transfer':>14}  {'Gain':>8}",
          flush=True)
    print("  " + "-" * 55, flush=True)
    for (src, tgt) in TRANSFER_PAIRS:
        s_acc = all_results[(src, tgt)]["scratch"]["acc"]
        t_acc = all_results[(src, tgt)]["transfer"]["acc"]
        gain = t_acc - s_acc
        print(f"  R{src} -> R{tgt}:       {s_acc:>13.4f}  {t_acc:>13.4f}  {gain:>+7.4f}",
              flush=True)
    print("=" * 60, flush=True)


def main():
    seed_everything(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    for src_round, tgt_round in TRANSFER_PAIRS:
        print(f"\n{'='*50}", flush=True)
        print(f"  Transfer: R{src_round} -> R{tgt_round}", flush=True)
        print(f"{'='*50}", flush=True)

        # Generate target data
        print(f"\nGenerating {tgt_round}-round data...", flush=True)
        X_train, Y_train = generate_data(N_TRAIN, tgt_round, mode=MODE)
        X_test,  Y_test  = generate_data(N_TEST,  tgt_round, mode=MODE)
        tgt_train = make_loader(X_train, Y_train, BATCH_SIZE, shuffle=True)
        tgt_test  = make_loader(X_test,  Y_test,  BATCH_SIZE, shuffle=False)

        # ── Train from scratch on target ─────────────────────────────
        print(f"\n-- {MODEL_NAME} from scratch on {tgt_round} rounds --", flush=True)
        model_scratch = build_model(MODEL_NAME, INPUT_DIM)
        scratch_acc, scratch_hist = train_model(
            model_scratch, tgt_train, tgt_test, EPOCHS, LR)

        # ── Pre-train on source, then fine-tune on target ────────────
        print(f"\n-- Pre-training {MODEL_NAME} on {src_round} rounds --", flush=True)
        X_src_train, Y_src_train = generate_data(N_TRAIN, src_round, mode=MODE)
        X_src_test,  Y_src_test  = generate_data(N_TEST,  src_round, mode=MODE)
        src_train = make_loader(X_src_train, Y_src_train, BATCH_SIZE, shuffle=True)
        src_test  = make_loader(X_src_test,  Y_src_test,  BATCH_SIZE, shuffle=False)

        model_transfer = build_model(MODEL_NAME, INPUT_DIM)
        pretrain_acc, _ = train_model(
            model_transfer, src_train, src_test, EPOCHS, LR)
        print(f"  Pre-train accuracy on R{src_round}: {pretrain_acc:.4f}", flush=True)

        print(f"\n-- Fine-tuning on {tgt_round} rounds --", flush=True)
        transfer_acc, transfer_hist = train_model(
            model_transfer, tgt_train, tgt_test, EPOCHS, LR_FINETUNE)

        all_results[(src_round, tgt_round)] = {
            "scratch":  {"acc": scratch_acc, "history": scratch_hist},
            "transfer": {"acc": transfer_acc, "history": transfer_hist},
        }

    # ── plots & summary ──────────────────────────────────────────────
    plot_transfer_comparison(all_results)
    print_transfer_summary(all_results)

    # save results
    serializable = {}
    for (src, tgt), v in all_results.items():
        serializable[f"R{src}_to_R{tgt}_scratch"] = v["scratch"]["acc"]
        serializable[f"R{src}_to_R{tgt}_transfer"] = v["transfer"]["acc"]
    with open(os.path.join(RESULTS_DIR, "transfer_results.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nPlots saved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
