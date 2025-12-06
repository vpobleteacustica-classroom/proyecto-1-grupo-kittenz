import os
import csv
import json
import argparse
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit

def read_metadata(csv_path: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def write_splits(rows, splits, out_csv):
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["npy_path", "folder", "track_id", "label", "duration", "split"])
        writer.writeheader()
        for row, split in zip(rows, splits):
            out = dict(row)
            out["split"] = split
            writer.writerow(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=os.path.join("data", "processed", "moodtheme", "metadata_moodtheme.csv"))
    ap.add_argument("--out-csv", default=os.path.join("data", "processed", "moodtheme", "splits_70_15_15.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_metadata(args.metadata)
    if len(rows) == 0:
        raise RuntimeError("Metadata está vacía.")

    labels = [r["label"] for r in rows]
    idxs = list(range(len(rows)))

    # Primero separar test (15%)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    trainval_idx, test_idx = next(sss_test.split(idxs, labels))

    # Estratificar val (15% del total, equivalente a 15/85 ~ 0.176 de los trainval)
    labels_trainval = [labels[i] for i in trainval_idx]
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.1764706, random_state=args.seed)
    train_idx_rel, val_idx_rel = next(sss_val.split(trainval_idx, labels_trainval))

    train_idx = [trainval_idx[i] for i in train_idx_rel]
    val_idx = [trainval_idx[i] for i in val_idx_rel]

    splits = [""] * len(rows)
    for i in train_idx:
        splits[i] = "train"
    for i in val_idx:
        splits[i] = "val"
    for i in test_idx:
        splits[i] = "test"

    write_splits(rows, splits, args.out_csv)
    print(f"Splits guardados en: {args.out_csv}")
    print(f"Conteos -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

if __name__ == "__main__":
    main()