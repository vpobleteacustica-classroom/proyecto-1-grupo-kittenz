import os
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def read_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_splits(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def detect_orientation(arr: np.ndarray):
    # Devuelve (n_mels, time, need_transpose, channels)
    if arr.ndim == 2:
        a, b = arr.shape
        if a <= b:
            return a, b, False, 1  # [n_mels, time]
        else:
            return b, a, True, 1   # [time, n_mels] -> transpose
    elif arr.ndim == 3:
        c, h, w = arr.shape
        if c == 1:
            return h, w, False, 1  # [1, n_mels, time]
        # genérico: menor dimensión como n_mels, mayor como time
        dims = [(0,c),(1,h),(2,w)]
        values = [d[1] for d in dims]
        min_idx = values.index(min(values))
        max_idx = values.index(max(values))
        n_mels = values[min_idx]
        time = values[max_idx]
        return n_mels, time, True, None
    else:
        raise ValueError(f"Arreglo con ndim={arr.ndim} no soportado.")

def zscore(x: torch.Tensor, eps=1e-6):
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)

class MelNpySingleLabelDataset(Dataset):
    def __init__(self, splits_csv, class_names_json, split="train",
                 time_frames=8192, target_n_mels=96, normalization="zscore"):
        self.rows = [r for r in read_splits(splits_csv) if r["split"] == split]
        self.class_names = read_class_names(class_names_json)
        self.label_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.time_frames = time_frames
        self.target_n_mels = target_n_mels
        self.normalization = normalization
        if len(self.rows) == 0:
            raise RuntimeError(f"No hay filas para split={split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        npy_path = row["npy_path"]
        label = row["label"]
        y = self.label_to_idx[label]

        arr = np.load(npy_path)  # numpy ndarray
        # Orientación
        n_mels, time, need_transpose, channels = detect_orientation(arr)
        if arr.ndim == 2 and need_transpose:
            arr = arr.T  # [n_mels, time]
        elif arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]  # [n_mels, time]
            else:
                arr = arr.reshape(arr.shape[-2], arr.shape[-1])

        # Forzar n_mels objetivo (interpolación bilineal si es necesario)
        x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [B=1, C=1, H=n_mels, W=time]
        H, W = x.shape[-2], x.shape[-1]
        target_H = self.target_n_mels
        if H != target_H:
            x = F.interpolate(x, size=(target_H, W), mode="bilinear", align_corners=False)

        # Recorte/padding temporal
        W = x.shape[-1]
        T = self.time_frames
        if W >= T:
            start = np.random.randint(0, W - T + 1)
            x = x[..., :, start:start+T]
        else:
            pad = T - W
            x = F.pad(x, (0, pad), mode="constant", value=float(x.min()))

        # Normalización
        if self.normalization == "zscore":
            x = zscore(x)
        elif self.normalization == "minmax":
            xmin = x.min()
            xmax = x.max()
            x = (x - xmin) / (xmax - xmin + 1e-6)
        # x: [1,1,target_n_mels,T]
        x = x.squeeze(0)  # [C=1, H, W]

        return x, y

def make_loader(splits_csv, class_names_json, split, batch_size=32, num_workers=4,
                time_frames=8192, target_n_mels=96, normalization="zscore"):
        ds = MelNpySingleLabelDataset(splits_csv, class_names_json, split=split,
                                      time_frames=time_frames, target_n_mels=target_n_mels,
                                      normalization=normalization)
        return DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"),
                          num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default=os.path.join("data", "processed", "moodtheme", "splits_70_15_15.csv"))
    parser.add_argument("--class-names", default=os.path.join("data", "processed", "moodtheme", "class_names.json"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--time-frames", type=int, default=8192)
    parser.add_argument("--target-n-mels", type=int, default=96)
    parser.add_argument("--normalization", default="zscore")
    args = parser.parse_args()

    loader = make_loader(args.splits, args.class_names, args.split,
                         batch_size=args.batch_size,
                         time_frames=args.time_frames,
                         target_n_mels=args.target_n_mels,
                         normalization=args.normalization)
    print(f"Loader OK: split={args.split}, batches={len(loader)}")