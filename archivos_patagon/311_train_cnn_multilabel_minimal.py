import os, csv, json, random, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import f1_score

from scripts.download.src_tiny_mel_cnn_minimal import TinyMelCNNMinimal
from scripts.download.dataset_npy_multilabel import make_loader, read_class_names

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def average_precision(y_true, y_score):
    import sklearn.metrics as skm
    ap_per_class = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]; ys = y_score[:, c]
        if np.all(yt == yt[0]):
            ap = np.nan
        else:
            try:
                ap = skm.average_precision_score(yt, ys)
            except Exception:
                ap = np.nan
        ap_per_class.append(ap)
    mAP = np.nanmean(ap_per_class) if len(ap_per_class) > 0 else float("nan")
    return float(mAP), ap_per_class

def evaluate(model, loader, device, threshold=0.3):
    model.eval()
    ys, yscores = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys.append(y.cpu().numpy())
            yscores.append(probs)
    Y = np.vstack(ys); S = np.vstack(yscores)
    mAP, _ = average_precision(Y, S)
    Yhat = (S >= threshold).astype(np.int32)
    f1_micro = f1_score(Y, Yhat, average="micro", zero_division=0)
    f1_macro = f1_score(Y, Yhat, average="macro", zero_division=0)
    return mAP, f1_micro, f1_macro

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=os.path.join("data","processed","moodtheme","splits_70_15_15_multilabel.csv"))
    ap.add_argument("--class-names", default=os.path.join("data","processed","moodtheme","class_names_multilabel.json"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--time-frames", type=int, default=2048)
    ap.add_argument("--target-n-mels", type=int, default=96)
    ap.add_argument("--normalization", default="zscore")
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--out-dir", default=os.path.join("data","processed","moodtheme","cnn_run_multilabel_minimal"))
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    class_names = read_class_names(args.class_names)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for seed in seeds:
        set_seed(seed)
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        train_loader = make_loader(args.splits, args.class_names, "train",
                                   batch_size=args.batch_size, time_frames=args.time_frames,
                                   target_n_mels=args.target_n_mels, normalization=args.normalization)
        val_loader = make_loader(args.splits, args.class_names, "val",
                                 batch_size=args.batch_size, time_frames=args.time_frames,
                                 target_n_mels=args.target_n_mels, normalization=args.normalization)
        test_loader = make_loader(args.splits, args.class_names, "test",
                                  batch_size=args.batch_size, time_frames=args.time_frames,
                                  target_n_mels=args.target_n_mels, normalization=args.normalization)

        model = TinyMelCNNMinimal(num_classes=num_classes, in_channels=1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_map = -1.0
        best_epoch = -1
        best_model_path = os.path.join(seed_dir, "best_model.pt")
        history = []

        for epoch in range(1, args.epochs+1):
            model.train()
            tot_loss, batches = 0.0, 0
            for x, y in train_loader:
                x = x.to(device); y = y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
                batches += 1
            train_loss = tot_loss / max(1, batches)

            val_map, val_f1_micro, val_f1_macro = evaluate(model, val_loader, device, threshold=args.threshold)
            test_map, test_f1_micro, test_f1_macro = evaluate(model, test_loader, device, threshold=args.threshold)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mAP": val_map,
                "val_f1_micro": val_f1_micro,
                "val_f1_macro": val_f1_macro,
                "test_mAP": test_map,
                "test_f1_micro": test_f1_micro,
                "test_f1_macro": test_f1_macro,
                "lr": optimizer.param_groups[0]["lr"]
            })

            if val_map > best_val_map:
                best_val_map = val_map
                best_epoch = epoch
                torch.save({
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_mAP": val_map,
                    "val_f1_micro": val_f1_micro,
                    "val_f1_macro": val_f1_macro,
                    "test_mAP": test_map,
                    "test_f1_micro": test_f1_micro,
                    "test_f1_macro": test_f1_macro,
                    "seed": seed,
                    "threshold_used": args.threshold
                }, best_model_path)

            print(f"[seed {seed}] Epoch {epoch}/{args.epochs} - loss={train_loss:.4f} "
                  f"val_mAP={val_map:.3f} val_f1_micro={val_f1_micro:.3f} val_f1_macro={val_f1_macro:.3f} "
                  f"test_mAP={test_map:.3f} test_f1_micro={test_f1_micro:.3f} test_f1_macro={test_f1_macro:.3f} "
                  f"lr={optimizer.param_groups[0]['lr']:.5f}")

        # Guardar historia y resumen
        hist_csv = os.path.join(seed_dir, "history.csv")
        with open(hist_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_mAP","val_f1_micro","val_f1_macro","test_mAP","test_f1_micro","test_f1_macro","lr"])
            writer.writeheader()
            for h in history: writer.writerow(h)

        summary_csv = os.path.join(args.out_dir, "run_summary.csv")
        write_header = not os.path.exists(summary_csv)
        with open(summary_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seed","best_epoch","best_val_mAP","best_model"])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_mAP": best_val_map,
                "best_model": best_model_path
            })

    print("Entrenamiento minimal finalizado.")

if __name__ == "__main__":
    main()