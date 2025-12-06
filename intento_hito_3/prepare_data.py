import pandas as pd
import numpy as np
import torch
from typing import Tuple, List

# Columnas de entrada que vienen del CSV de extracción
FEATURE_COLUMNS = [
    "tempo","loudness","time_signature","spectral_centroid","zcr","rms_energy",
    "spectral_rolloff","onset_strength",
]
FEATURE_COLUMNS += [f"mfcc{i}_mean" for i in range(1,14)]
FEATURE_COLUMNS += [f"mfcc{i}_std" for i in range(1,14)]
FEATURE_COLUMNS += [f"chroma{i}_mean" for i in range(1,13)]

def read_features_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Asegurar que columnas existan
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en CSV: {missing}")
    # Asegurar columna de tags
    if "moodtheme_tags" not in df.columns:
        raise ValueError("CSV no contiene columna 'moodtheme_tags'.")
    return df

def build_tag_vocab(df: pd.DataFrame) -> List[str]:
    tags_set = set()
    for s in df["moodtheme_tags"].fillna("").astype(str):
        if not s.strip():
            continue
        for t in s.split(";"):
            t = t.strip()
            if t:
                tags_set.add(t)
    vocab = sorted(list(tags_set))
    return vocab

def encode_single_label(df: pd.DataFrame, tag_vocab: List[str]) -> np.ndarray:
    """
    Single-label: usa el primer tag de 'moodtheme_tags'; si no hay tags, asigna clase -1 (luego filtramos).
    """
    tag_to_idx = {t:i for i,t in enumerate(tag_vocab)}
    y_idx = []
    for s in df["moodtheme_tags"].fillna("").astype(str):
        if not s.strip():
            y_idx.append(-1)
            continue
        first = s.split(";")[0].strip()
        y_idx.append(tag_to_idx.get(first, -1))
    return np.array(y_idx, dtype=np.int64)

def encode_multi_label(df: pd.DataFrame, tag_vocab: List[str]) -> np.ndarray:
    """
    Multi-label: vector binario multi-hot de tamaño len(tag_vocab).
    """
    tag_to_idx = {t:i for i,t in enumerate(tag_vocab)}
    Y = np.zeros((len(df), len(tag_vocab)), dtype=np.float32)
    for i, s in enumerate(df["moodtheme_tags"].fillna("").astype(str)):
        if not s.strip():
            continue
        for t in s.split(";"):
            t = t.strip()
            if t in tag_to_idx:
                Y[i, tag_to_idx[t]] = 1.0
    return Y

def build_X(df: pd.DataFrame, normalize: bool=True) -> np.ndarray:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if normalize:
        # Normalización por columna: z-score
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std
    return X

def split_train_val_test(X: np.ndarray, y, train_ratio: float=0.7, val_ratio: float=0.15, seed: int=42):
    """
    Split aleatorio reproducible: 70% train, 15% val, 15% test.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    if isinstance(y, np.ndarray):
        return (X[train_idx], X[val_idx], X[test_idx],
                y[train_idx], y[val_idx], y[test_idx])
    else:
        y_arr = np.array(y)
        return (X[train_idx], X[val_idx], X[test_idx],
                y_arr[train_idx], y_arr[val_idx], y_arr[test_idx])

def to_tensors(X_train, X_val, X_test, y_train, y_val, y_test, multilabel: bool=False):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    if multilabel:
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
    return X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t

def prepare(csv_path: str, mode: str="single", normalize: bool=True, seed: int=42):
    """
    mode: "single" | "multi"
    Devuelve:
    X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t, input_dim, num_classes, tag_vocab
    """
    df = read_features_csv(csv_path)
    tag_vocab = build_tag_vocab(df)
    X = build_X(df, normalize=normalize)

    if mode == "single":
        y = encode_single_label(df, tag_vocab)
        # Filtrar filas sin etiqueta (-1)
        keep = y != -1
        X = X[keep]
        y = y[keep]
        num_classes = len(tag_vocab)
        multilabel = False
    elif mode == "multi":
        y = encode_multi_label(df, tag_vocab)
        num_classes = len(tag_vocab)
        multilabel = True
    else:
        raise ValueError("mode debe ser 'single' o 'multi'")

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, train_ratio=0.7, val_ratio=0.15, seed=seed)
    X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t = to_tensors(X_train, X_val, X_test, y_train, y_val, y_test, multilabel=multilabel)
    input_dim = X_train.shape[1]
    return X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t, input_dim, num_classes, tag_vocab
