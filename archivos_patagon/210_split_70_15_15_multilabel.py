import os
import csv
import json
import argparse
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def read_metadata(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def read_classes(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def tags_to_multihot(tags_str, class_names):
    tags = tags_str.split(";") if tags_str else []
    vec = [0]*len(class_names)
    idx_map = {c:i for i,c in enumerate(class_names)}
    for t in tags:
        if t in idx_map:
            vec[idx_map[t]] = 1
    return vec

def write_splits(rows, splits, out_csv):
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["npy_path","folder","track_id","tags","duration","split"])
        writer.writeheader()
        for row, split in zip(rows, splits):
            out = dict(row); out["split"]=split
            writer.writerow(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=os.path.join("data","processed","moodtheme","metadata_moodtheme_multilabel.csv"))
    ap.add_argument("--classes", default=os.path.join("data","processed","moodtheme","class_names_multilabel.json"))
    ap.add_argument("--out-csv", default=os.path.join("data","processed","moodtheme","splits_70_15_15_multilabel.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_metadata(args.metadata)
    class_names = read_classes(args.classes)
    X = list(range(len(rows)))
    Y = [tags_to_multihot(r["tags"], class_names) for r in rows]

    # test 15%
    mss_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    trainval_idx, test_idx = next(mss_test.split(X, Y))

    # val 15% del total => 15/85 de trainval
    Y_trainval = [Y[i] for i in trainval_idx]
    mss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1764706, random_state=args.seed)
    train_idx_rel, val_idx_rel = next(mss_val.split(trainval_idx, Y_trainval))
    train_idx = [trainval_idx[i] for i in train_idx_rel]
    val_idx = [trainval_idx[i] for i in val_idx_rel]

    splits = [""]*len(rows)
    for i in train_idx: splits[i]="train"
    for i in val_idx: splits[i]="val"
    for i in test_idx: splits[i]="test"

    write_splits(rows, splits, args.out_csv)
    print(f"Splits multilabel guardados: {args.out_csv}")
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

if __name__ == "__main__":
    main()