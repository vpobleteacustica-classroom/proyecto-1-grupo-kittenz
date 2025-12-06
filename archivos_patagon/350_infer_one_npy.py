import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from scripts.download.src_tiny_mel_cnn_single import TinyMelCNN
from scripts.download.dataset_npy_single import detect_orientation, zscore

def load_best_model(run_dir):
    best_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"No encontrado: {best_path}")
    ckpt = torch.load(best_path, map_location="cpu")
    return ckpt

def prepare_tensor_from_npy(npy_path, target_n_mels=96, time_frames=8192, normalization="zscore"):
    arr = np.load(npy_path)
    n_mels, time, need_transpose, channels = detect_orientation(arr)
    if arr.ndim == 2 and need_transpose:
        arr = arr.T
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])

    x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    H, W = x.shape[-2], x.shape[-1]
    if H != target_n_mels:
        x = F.interpolate(x, size=(target_n_mels, W), mode="bilinear", align_corners=False)
    W = x.shape[-1]
    T = time_frames
    if W >= T:
        start = 0
        x = x[..., :, start:start+T]
    else:
        pad = T - W
        x = F.pad(x, (0, pad), mode="constant", value=float(x.min()))
    if normalization == "zscore":
        x = zscore(x)
    elif normalization == "minmax":
        xmin = x.min()
        xmax = x.max()
        x = (x - xmin) / (xmax - xmin + 1e-6)
    return x  # [1,1,H,T]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Directorio del seed con best_model.pt")
    ap.add_argument("--npy", required=True, help="Ruta al .npy")
    ap.add_argument("--target-n-mels", type=int, default=96)
    ap.add_argument("--time-frames", type=int, default=8192)
    ap.add_argument("--normalization", default="zscore")
    args = ap.parse_args()

    ckpt = load_best_model(args.run_dir)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)
    model = TinyMelCNN(num_classes=num_classes, in_channels=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = prepare_tensor_from_npy(args.npy, target_n_mels=args.target_n_mels,
                                time_frames=args.time_frames, normalization=args.normalization)
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    pred_prob = float(probs[pred_idx])

    print(f"Npy: {args.npy}")
    print(f"Predicción: {pred_label} (p={pred_prob:.3f})")
    print("Top-5:")
    order = np.argsort(-probs)[:5]
    for i in order:
        print(f"  {class_names[i]}: {probs[i]:.3f}")

if __name__ == "__main__":
    main()