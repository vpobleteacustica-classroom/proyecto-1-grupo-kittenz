import os
import csv
import json
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
    if arr.ndim == 2:
        a,b = arr.shape
        if a<=b: return a,b,False,1
        else: return b,a,True,1
    elif arr.ndim == 3:
        c,h,w = arr.shape
        if c==1: return h,w,False,1
        else:
            return min(c,h,w), max(c,h,w), True, None
    else:
        raise ValueError(f"ndim={arr.ndim} no soportado")

def zscore(x: torch.Tensor, eps=1e-6):
    mean = x.mean(); std = x.std()
    return (x-mean)/(std+eps)

def tags_to_multihot(tags_str, class_names):
    tags = tags_str.split(";") if tags_str else []
    vec = torch.zeros(len(class_names), dtype=torch.float32)
    idx_map = {c:i for i,c in enumerate(class_names)}
    for t in tags:
        if t in idx_map:
            vec[idx_map[t]] = 1.0
    return vec

class MelNpyMultiLabelDataset(Dataset):
    def __init__(self, splits_csv, class_names_json, split="train",
                 time_frames=8192, target_n_mels=96, normalization="zscore"):
        self.rows = [r for r in read_splits(splits_csv) if r["split"]==split]
        self.class_names = read_class_names(class_names_json)
        self.time_frames = time_frames
        self.target_n_mels = target_n_mels
        self.normalization = normalization
        if len(self.rows)==0:
            raise RuntimeError(f"No hay filas para split={split}")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        x = np.load(row["npy_path"])
        n_mels, time, need_transpose, _ = detect_orientation(x)
        if x.ndim==2 and need_transpose:
            x = x.T
        elif x.ndim==3:
            if x.shape[0]==1: x = x[0]
            else: x = x.reshape(x.shape[-2], x.shape[-1])

        xt = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0) # [1,1,H,W]
        H,W = xt.shape[-2], xt.shape[-1]
        if H != self.target_n_mels:
            xt = F.interpolate(xt, size=(self.target_n_mels, W), mode="bilinear", align_corners=False)
        W = xt.shape[-1]
        T = self.time_frames
        if W >= T:
            start = np.random.randint(0, W-T+1) if self.normalization != "eval_center" else (W-T)//2
            xt = xt[..., :, start:start+T]
        else:
            pad = T-W
            xt = F.pad(xt, (0,pad), mode="constant", value=float(xt.min()))

        if self.normalization=="zscore":
            xt = zscore(xt)
        elif self.normalization=="minmax":
            xmin = xt.min(); xmax = xt.max()
            xt = (xt - xmin) / (xmax - xmin + 1e-6)

        xt = xt.squeeze(0) # [1,H,T]
        yt = tags_to_multihot(row["tags"], self.class_names) # [C]
        return xt, yt

def make_loader(splits_csv, class_names_json, split, batch_size=32, num_workers=4,
                time_frames=8192, target_n_mels=96, normalization="zscore"):
    ds = MelNpyMultiLabelDataset(splits_csv, class_names_json, split=split,
                                 time_frames=time_frames, target_n_mels=target_n_mels,
                                 normalization=normalization)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"),
                      num_workers=num_workers, pin_memory=True)