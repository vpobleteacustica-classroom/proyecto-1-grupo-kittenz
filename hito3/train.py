import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from emotion_mlp import EmotionMLP
from prepare_data import prepare

def compute_pos_weight(y_train: torch.Tensor) -> torch.Tensor:
    """
    pos_weight for BCEWithLogitsLoss: weight for positive examples per class.
    pos_weight_j = N_neg_j / N_pos_j
    """
    # y_train shape: [N, C], float32 in {0,1}
    y_np = y_train.cpu().numpy()
    pos = y_np.sum(axis=0)  # per class
    neg = y_np.shape[0] - pos
    # Avoid division by zero
    pos = np.where(pos == 0, 1.0, pos)
    pw = neg / pos
    return torch.tensor(pw, dtype=torch.float32)

def find_per_class_thresholds(y_val: torch.Tensor, probs_val: torch.Tensor,
                              grid=None) -> np.ndarray:
    """
    Per-class threshold selection to maximize F1 (micro-approx by per-label F1).
    Returns thresholds array of shape [C].
    """
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)  # 0.1, 0.15, ..., 0.9
    yv = y_val.cpu().numpy()
    pv = probs_val.cpu().numpy()
    C = yv.shape[1]
    thresholds = np.full(C, 0.5, dtype=np.float32)

    for j in range(C):
        best_t = 0.5
        best_f1 = -1.0
        y_true = yv[:, j]
        # skip if no positives and no negatives (unlikely)
        for t in grid:
            y_pred = (pv[:, j] >= t).astype(np.float32)
            # f1_score with zero_division to avoid warnings
            f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[j] = best_t
    return thresholds

def apply_thresholds(probs: torch.Tensor, thresholds: np.ndarray) -> torch.Tensor:
    """
    Apply per-class thresholds to probability tensor.
    probs: [N, C] float
    thresholds: [C] numpy
    """
    thr = torch.tensor(thresholds, dtype=protorch_dtype(probs), device=probs.device)
    return (probs >= thr).float()

def protorch_dtype(t: torch.Tensor):
    return torch.float32 if t.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16) else torch.float32

def train_epoch(model, optimizer, criterion, X_train, y_train, batch_size=256):
    model.train()
    total_loss = 0.0
    n = X_train.shape[0]
    for i in range(0, n, batch_size):
        xb = X_train[i:i+batch_size]
        yb = y_train[i:i+batch_size]
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * xb.shape[0]
    return total_loss / n

@torch.no_grad()
def evaluate_single(model, X, y):
    model.eval()
    logits = model(X)
    preds = torch.argmax(logits, dim=1)
    acc = (preds == y).float().mean().item()
    return {"accuracy": acc}

@torch.no_grad()
def evaluate_multi(model, X, y, thresholds=None):
    model.eval()
    logits = model(X)
    probs = torch.sigmoid(logits)
    if thresholds is None:
        preds = (probs >= 0.5).float()
    else:
        preds = apply_thresholds(probs, thresholds)

    y_np = y.cpu().numpy()
    preds_np = preds.cpu().numpy()

    f1_micro = f1_score(y_np, preds_np, average="micro", zero_division=0)
    f1_macro = f1_score(y_np, preds_np, average="macro", zero_division=0)
    subset_acc = np.mean(np.all(preds_np == y_np, axis=1))
    label_acc_mean = float((preds == y).float().mean().item())

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "subset_acc": subset_acc,
        "label_acc_mean": label_acc_mean,
        "probs": probs,  # return for threshold tuning on val
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Ruta al CSV de features (features_low.csv)")
    parser.add_argument("--mode", type=str, default="single", choices=["single","multi"], help="Modo de etiquetas")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold_tuning", action="store_true",
                        help="Ajusta umbrales por clase usando el conjunto de validación.")
    parser.add_argument("--use_pos_weight", action="store_true",
                        help="Usa pos_weight en BCEWithLogitsLoss para clases desbalanceadas (multi-label).")
    args = parser.parse_args()

    # Prepare data (70/15/15 split inside)
    X_train, X_val, X_test, y_train, y_val, y_test, input_dim, num_classes, tag_vocab = prepare(
        csv_path=args.csv, mode=args.mode, normalize=True, seed=42
    )

    model = EmotionMLP(input_dim=input_dim, num_classes=num_classes,
                       hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout)

    if args.mode == "single":
        criterion = nn.CrossEntropyLoss()
    else:
        if args.use_pos_weight:
            pw = compute_pos_weight(y_train)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    tuned_thresholds = None

    for epoch in range(args.epochs):
        loss = train_epoch(model, optimizer, criterion, X_train, y_train, batch_size=args.batch_size)

        if args.mode == "single":
            val_metrics = evaluate_single(model, X_val, y_val)
            test_metrics = evaluate_single(model, X_test, y_test)
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f} | "
                  f"ValAcc: {val_metrics['accuracy']:.4f} | TestAcc: {test_metrics['accuracy']:.4f}")
        else:
            # Multi-label: evaluate with default 0.5 thresholds
            val_metrics = evaluate_multi(model, X_val, y_val, thresholds=None)
            test_metrics = evaluate_multi(model, X_test, y_test, thresholds=None)

            # Per-class threshold tuning on validation (optional)
            if args.threshold_tuning:
                thresholds = find_per_class_thresholds(y_val, val_metrics["probs"])
                tuned_thresholds = thresholds
                # Re-evaluate with tuned thresholds
                val_metrics_tuned = evaluate_multi(model, X_val, y_val, thresholds=tuned_thresholds)
                test_metrics_tuned = evaluate_multi(model, X_test, y_test, thresholds=tuned_thresholds)
                print(
                    f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f} | "
                    f"Val(0.5): F1_micro {val_metrics['f1_micro']:.4f}, F1_macro {val_metrics['f1_macro']:.4f}, "
                    f"SubsetAcc {val_metrics['subset_acc']:.4f}, LabelAccMean {val_metrics['label_acc_mean']:.4f} | "
                    f"Test(0.5): F1_micro {test_metrics['f1_micro']:.4f}, F1_macro {test_metrics['f1_macro']:.4f}, "
                    f"SubsetAcc {test_metrics['subset_acc']:.4f}, LabelAccMean {test_metrics['label_acc_mean']:.4f}\n"
                    f"          Val(tuned): F1_micro {val_metrics_tuned['f1_micro']:.4f}, F1_macro {val_metrics_tuned['f1_macro']:.4f}, "
                    f"SubsetAcc {val_metrics_tuned['subset_acc']:.4f}, LabelAccMean {val_metrics_tuned['label_acc_mean']:.4f} | "
                    f"Test(tuned): F1_micro {test_metrics_tuned['f1_micro']:.4f}, F1_macro {test_metrics_tuned['f1_macro']:.4f}, "
                    f"SubsetAcc {test_metrics_tuned['subset_acc']:.4f}, LabelAccMean {test_metrics_tuned['label_acc_mean']:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f} | "
                    f"Val: F1_micro {val_metrics['f1_micro']:.4f}, F1_macro {val_metrics['f1_macro']:.4f}, "
                    f"SubsetAcc {val_metrics['subset_acc']:.4f}, LabelAccMean {val_metrics['label_acc_mean']:.4f} | "
                    f"Test: F1_micro {test_metrics['f1_micro']:.4f}, F1_macro {test_metrics['f1_macro']:.4f}, "
                    f"SubsetAcc {test_metrics['subset_acc']:.4f}, LabelAccMean {test_metrics['label_acc_mean']:.4f}"
                )

    # Guardado de modelo y thresholds (si se ajustaron)
    torch.save(model.state_dict(), "emotion_mlp.pt")
    if tuned_thresholds is not None:
        np.save("thresholds_per_class.npy", tuned_thresholds)
        print("Umbrales por clase guardados en thresholds_per_class.npy")
    print("Modelo guardado en emotion_mlp.pt")

if __name__ == "__main__":
    main()