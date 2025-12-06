import os, csv, json, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from datetime import datetime

from scripts.download.src_tiny_mel_cnn_multilabel import TinyMelCNNMulti
from scripts.download.dataset_npy_multilabel import make_loader, read_class_names, read_splits

# =========================
# Utilidades
# =========================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def average_precision(y_true, y_score):
    import sklearn.metrics as skm
    ap_per_class = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        ys = y_score[:, c]
        # Si todas las etiquetas son iguales (todo 0 o todo 1), AP no está definido
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

def compute_pos_weight_from_train(splits_csv, class_names):
    """
    Calcula pos_weight por clase usando conteos del split train.
    pos_weight[c] ~ N_neg_c / max(1, N_pos_c), con clipping para estabilidad.
    Nota: N_total se estima por número de samples en train (no suma de etiquetas).
    """
    rows = read_splits(splits_csv)
    train_rows = [r for r in rows if r.get("split", "") == "train"]
    num_train_samples = len(train_rows)

    # Conteo de positivos por clase (cantidad de samples del train que tienen la etiqueta c)
    idx_map = {c: i for i, c in enumerate(class_names)}
    pos_counts = np.zeros(len(class_names), dtype=np.int64)
    for r in train_rows:
        tags = (r.get("tags", "") or "").split(";")
        seen = set()
        for t in tags:
            t = t.strip()
            if t and t in idx_map and t not in seen:
                pos_counts[idx_map[t]] += 1
                seen.add(t)

    # N_neg = N_total - N_pos
    neg_counts = num_train_samples - pos_counts
    # pos_weight = N_neg / max(1, N_pos)
    pos_weight = (neg_counts / np.maximum(1, pos_counts)).astype(np.float32)

    # Estabilidad: clamp
    pos_weight = np.clip(pos_weight, 0.5, 10.0)
    return torch.tensor(pos_weight, dtype=torch.float32)

# SpecAugment ligero: máscaras de tiempo y frecuencia
def spec_augment(x, time_mask_ratio=0.15, freq_mask_ratio=0.10):
    """
    x: tensor [B, 1, H, T]
    Aplica enmascarado de tiempo y frecuencia simple.
    """
    B, C, H, T = x.shape
    if time_mask_ratio > 0:
        t_width = max(1, int(T * time_mask_ratio))
        for b in range(B):
            start = random.randint(0, max(0, T - t_width))
            x[b, :, :, start:start + t_width] = 0.0
    if freq_mask_ratio > 0:
        f_width = max(1, int(H * freq_mask_ratio))
        for b in range(B):
            start = random.randint(0, max(0, H - f_width))
            x[b, :, start:start + f_width, :] = 0.0
    return x

# =========================
# Evaluación con TTA y umbral configurable
# =========================
def evaluate(model, loader, device, tta_crops=0, time_frames=8192, threshold=0.5):
    """
    Devuelve mAP y F1 (micro/macro) usando umbral dado.
    tta_crops: número de crops adicionales aleatorios (además del crop del loader).
    """
    model.eval()
    ys, yscores = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Predicción base
            logits = model(x)

            # Test-time augmentation: crops adicionales aleatorios del mismo batch
            if tta_crops > 0:
                B, C, H, T = x.shape
                acc_logits = logits
                for _ in range(tta_crops):
                    # Re-generar un crop aleatorio a partir del tensor ya cargado
                    # Aquí implementamos un "jitter" temporal simple (si T > time_frames)
                    if T > time_frames:
                        start = random.randint(0, T - time_frames)
                        x_aug = x[:, :, :, start:start + time_frames]
                    else:
                        x_aug = x
                    acc_logits = acc_logits + model(x_aug)
                logits = acc_logits / (tta_crops + 1)

            probs = torch.sigmoid(logits).cpu().numpy()
            ys.append(y.cpu().numpy())
            yscores.append(probs)

    Y = np.vstack(ys)         # [N,C]
    S = np.vstack(yscores)    # [N,C]
    mAP, ap_per_class = average_precision(Y, S)

    Yhat_bin = (S >= threshold).astype(np.int32)
    f1_micro = f1_score(Y, Yhat_bin, average="micro", zero_division=0)
    f1_macro = f1_score(Y, Yhat_bin, average="macro", zero_division=0)

    return mAP, f1_micro, f1_macro

# =========================
# Entrenamiento
# =========================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=os.path.join("data","processed","moodtheme","splits_70_15_15_multilabel.csv"))
    ap.add_argument("--class-names", default=os.path.join("data","processed","moodtheme","class_names_multilabel.json"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--time-frames", type=int, default=8192)
    ap.add_argument("--target-n-mels", type=int, default=96)
    ap.add_argument("--normalization", default="zscore")
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--out-dir", default=os.path.join("data","processed","moodtheme","cnn_run_multilabel"))
    # Nuevos parámetros
    ap.add_argument("--threshold", type=float, default=0.3, help="Umbral para F1 en eval (por defecto 0.3)")
    ap.add_argument("--tta-crops", type=int, default=2, help="Crops adicionales en eval (TTA). 0 desactiva.")
    ap.add_argument("--specaug-time", type=float, default=0.15, help="Proporción de tiempo a enmascarar en entrenamiento.")
    ap.add_argument("--specaug-freq", type=float, default=0.10, help="Proporción de frecuencia a enmascarar en entrenamiento.")
    ap.add_argument("--use-pos-weight", action="store_true", help="Usa pos_weight en BCE calculado desde train.")
    ap.add_argument("--scheduler", choices=["none","plateau","cosine"], default="plateau")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    class_names = read_class_names(args.class_names)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    summary_rows = []

    # Calcular pos_weight si se solicita
    pos_weight = None
    if args.use_pos_weight:
        pos_weight = compute_pos_weight_from_train(args.splits, class_names).to(device)
        print("pos_weight (primeras 10):", pos_weight[:10].tolist())

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

        model = TinyMelCNNMulti(num_classes=num_classes, in_channels=1).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=7, verbose=True)
        elif args.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        else:
            scheduler = None

        best_val_map = -1.0
        best_epoch = -1
        best_model_path = os.path.join(seed_dir, "best_model.pt")
        history = []

        for epoch in range(1, args.epochs+1):
            model.train()
            tot_loss, batches = 0.0, 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                # SpecAugment ligero en entrenamiento
                if args.specaug_time > 0 or args.specaug_freq > 0:
                    x = spec_augment(x, time_mask_ratio=args.specaug_time, freq_mask_ratio=args.specaug_freq)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
                batches += 1

            train_loss = tot_loss / max(1, batches)

            # Eval con TTA y umbral configurable
            val_map, val_f1_micro, val_f1_macro = evaluate(model, val_loader, device,
                                                           tta_crops=args.tta_crops,
                                                           time_frames=args.time_frames,
                                                           threshold=args.threshold)
            test_map, test_f1_micro, test_f1_macro = evaluate(model, test_loader, device,
                                                              tta_crops=args.tta_crops,
                                                              time_frames=args.time_frames,
                                                              threshold=args.threshold)

            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_map)
                else:
                    scheduler.step()

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
                    "threshold_used": args.threshold,
                    "tta_crops_used": args.tta_crops,
                    "specaug_time": args.specaug_time,
                    "specaug_freq": args.specaug_freq,
                    "use_pos_weight": args.use_pos_weight,
                    "scheduler": args.scheduler
                }, best_model_path)

            print(f"[seed {seed}] Epoch {epoch}/{args.epochs} - loss={train_loss:.4f} "
                  f"val_mAP={val_map:.3f} val_f1_micro={val_f1_micro:.3f} val_f1_macro={val_f1_macro:.3f} "
                  f"test_mAP={test_map:.3f} test_f1_micro={test_f1_micro:.3f} test_f1_macro={test_f1_macro:.3f} "
                  f"lr={optimizer.param_groups[0]['lr']:.5f}")

        # Guardar historia
        hist_csv = os.path.join(seed_dir, "history.csv")
        with open(hist_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_mAP","val_f1_micro","val_f1_macro","test_mAP","test_f1_micro","test_f1_macro","lr"])
            writer.writeheader()
            for h in history: writer.writerow(h)

        summary_rows = []
        summary_rows.append({
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_mAP": best_val_map,
            "best_model": best_model_path
        })

        summary_csv = os.path.join(args.out_dir, "run_summary.csv")
        # Append or create
        write_header = not os.path.exists(summary_csv)
        with open(summary_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seed","best_epoch","best_val_mAP","best_model"])
            if write_header:
                writer.writeheader()
            for r in summary_rows: writer.writerow(r)

    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    main()