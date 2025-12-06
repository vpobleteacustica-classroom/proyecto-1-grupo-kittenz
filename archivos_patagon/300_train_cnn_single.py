import os
import csv
import json
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from scripts.download.src_tiny_mel_cnn_single import TinyMelCNN
from scripts.download.dataset_npy_single import make_loader, read_class_names

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, device):
    model.eval()
    ys, yhats = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            ys.extend(y.cpu().numpy().tolist())
            yhats.extend(preds.cpu().numpy().tolist())
    acc = accuracy_score(ys, yhats)
    cm = confusion_matrix(ys, yhats)
    return acc, cm, ys, yhats

def plot_confusion_matrix(cm, class_names, out_png):
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix (counts)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=os.path.join("data", "processed", "moodtheme", "splits_70_15_15.csv"))
    ap.add_argument("--class-names", default=os.path.join("data", "processed", "moodtheme", "class_names.json"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--time-frames", type=int, default=8192)
    ap.add_argument("--target-n-mels", type=int, default=96)
    ap.add_argument("--normalization", default="zscore")
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--out-dir", default=os.path.join("data", "processed", "moodtheme", "cnn_run"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    class_names = read_class_names(args.class_names)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    run_summary_rows = []
    for seed in seeds:
        set_seed(seed)
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        # DataLoaders
        train_loader = make_loader(args.splits, args.class_names, "train",
                                   batch_size=args.batch_size, time_frames=args.time_frames,
                                   target_n_mels=args.target_n_mels, normalization=args.normalization)
        val_loader = make_loader(args.splits, args.class_names, "val",
                                 batch_size=args.batch_size, time_frames=args.time_frames,
                                 target_n_mels=args.target_n_mels, normalization=args.normalization)
        test_loader = make_loader(args.splits, args.class_names, "test",
                                  batch_size=args.batch_size, time_frames=args.time_frames,
                                  target_n_mels=args.target_n_mels, normalization=args.normalization)

        # Modelo
        model = TinyMelCNN(num_classes=num_classes, in_channels=1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        history = []
        best_val_acc = -1.0
        best_epoch = -1
        best_model_path = os.path.join(seed_dir, "best_model.pt")

        for epoch in range(1, args.epochs+1):
            model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1
            train_loss = epoch_loss / max(1, epoch_batches)

            # Eval
            val_acc, _, _, _ = evaluate(model, val_loader, device)
            test_acc, test_cm, _, _ = evaluate(model, test_loader, device)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "test_acc": test_acc
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save({
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "seed": seed
                }, best_model_path)

            print(f"[seed {seed}] Epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")

        # Guardar historia
        hist_csv = os.path.join(seed_dir, "history.csv")
        with open(hist_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_acc","test_acc"])
            writer.writeheader()
            for h in history:
                writer.writerow(h)

        # Curvas
        try:
            epochs = [h["epoch"] for h in history]
            train_losses = [h["train_loss"] for h in history]
            val_accs = [h["val_acc"] for h in history]
            test_accs = [h["test_acc"] for h in history]
            fig, ax1 = plt.subplots(figsize=(10,5))
            ax2 = ax1.twinx()
            ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
            ax2.plot(epochs, val_accs, 'g-', label='Val Acc')
            ax2.plot(epochs, test_accs, 'r-', label='Test Acc')
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax2.set_ylabel("Accuracy")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(os.path.join(seed_dir, "curvas_loss_acc.png"))
            plt.close(fig)
        except Exception as e:
            print(f"No se pudo generar curvas: {e}")

        # Matriz de confusión (test)
        try:
            plot_confusion_matrix(test_cm, class_names, os.path.join(seed_dir, "confusion_matrix.png"))
            cm_csv = os.path.join(seed_dir, "confusion_matrix_counts.csv")
            with open(cm_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([""] + class_names)
                for i, row in enumerate(test_cm):
                    writer.writerow([class_names[i]] + row.tolist())
        except Exception as e:
            print(f"No se pudo guardar matriz de confusión: {e}")

        run_summary_rows.append({
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_model": best_model_path
        })

    summary_csv = os.path.join(args.out_dir, "run_summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed","best_epoch","best_val_acc","best_model"])
        writer.writeheader()
        for r in run_summary_rows:
            writer.writerow(r)

    print(f"Resumen global guardado en: {summary_csv}")

if __name__ == "__main__":
    main()