import os, json, argparse, numpy as np, torch, torch.nn.functional as F
from scripts.download.src_tiny_mel_cnn_multilabel import TinyMelCNNMulti
from scripts.download.dataset_npy_multilabel import detect_orientation, zscore

def load_best_model(run_dir):
    best_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"No encontrado: {best_path}")
    ckpt = torch.load(best_path, map_location="cpu")
    return ckpt

def prepare_tensor_from_npy(npy_path, target_n_mels=96, time_frames=8192, normalization="zscore"):
    arr = np.load(npy_path)
    n_mels, time, need_transpose, _ = detect_orientation(arr)
    if arr.ndim==2 and need_transpose: arr = arr.T
    elif arr.ndim==3:
        if arr.shape[0]==1: arr = arr[0]
        else: arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    H,W = x.shape[-2], x.shape[-1]
    if H != target_n_mels:
        x = F.interpolate(x, size=(target_n_mels, W), mode="bilinear", align_corners=False)
    W = x.shape[-1]; T = time_frames
    if W >= T:
        start = (W-T)//2  # centro en inferencia
        x = x[..., :, start:start+T]
    else:
        pad = T-W; x = F.pad(x, (0,pad), mode="constant", value=float(x.min()))
    if normalization=="zscore": x = zscore(x)
    elif normalization=="minmax":
        xmin=x.min(); xmax=x.max()
        x=(x-xmin)/(xmax-xmin+1e-6)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--target-n-mels", type=int, default=96)
    ap.add_argument("--time-frames", type=int, default=8192)
    ap.add_argument("--normalization", default="zscore")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    ckpt = load_best_model(args.run_dir)
    class_names = ckpt["class_names"]
    model = TinyMelCNNMulti(num_classes=len(class_names), in_channels=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = prepare_tensor_from_npy(args.npy, target_n_mels=args.target_n_mels,
                                time_frames=args.time_frames, normalization=args.normalization)
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    pred_indices = np.where(probs >= args.threshold)[0]
    preds = [(class_names[i], float(probs[i])) for i in pred_indices]
    # Top-5 por prob
    order = np.argsort(-probs)[:5]
    top5 = [(class_names[i], float(probs[i])) for i in order]

    print(f"Npy: {args.npy}")
    print("Etiquetas (umbral):")
    for lab, p in preds:
        print(f"  {lab}: {p:.3f}")
    print("Top-5:")
    for lab, p in top5:
        print(f"  {lab}: {p:.3f}")

if __name__ == "__main__":
    main()