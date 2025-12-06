import os
import csv
import tarfile
import hashlib
import requests
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import librosa
import essentia.standard as es
from tqdm import tqdm
import sys

# Configuración
REPO_ROOT = Path(os.environ.get("REPO_ROOT", "./mtg-jamendo-dataset")).resolve()
DATA_TYPE = os.environ.get("DATA_TYPE", "audio-low")  # "audio" | "audio-low"
DOWNLOAD_FROM = os.environ.get("DOWNLOAD_FROM", "mtg-fast")  # "mtg-fast" | "mtg"
WORK_DIR = Path(os.environ.get("WORK_DIR", "./work_moodtheme")).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = Path(os.environ.get("OUTPUT_CSV", "./features_moodtheme.csv")).resolve()
CHECKPOINT_CSV = Path(os.environ.get("CHECKPOINT_CSV", "./processed_tars.csv")).resolve()
MAX_TARS = int(os.environ.get("MAX_TARS", "0"))  # 0 = procesar todos los TARs

# Parámetros de análisis Librosa (moderados para velocidad)
LIBROSA_N_FFT = int(os.environ.get("LIBROSA_N_FFT", "1024"))
LIBROSA_HOP = int(os.environ.get("LIBROSA_HOP", "1024"))

def compute_sha256(filename: Path) -> str:
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_file(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        with tempfile.NamedTemporaryFile(prefix=output.name, dir=output.parent, delete=False) as tmp:
            tmp_name = tmp.name
            with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {output.name}") as pbar:
                for chunk in r.iter_content(chunk_size=512 * 1024):
                    if chunk:
                        tmp.write(chunk)
                        pbar.update(len(chunk))
        shutil.move(tmp_name, output)

def build_base_url(dataset: str, data_type: str, download_from: str) -> str:
    if download_from == "mtg-fast":
        return f"https://cdn.freesound.org/mtg-jamendo/{dataset}/{data_type}/"
    elif download_from == "mtg":
        return f"https://essentia.upf.edu/documentation/datasets/mtg-jamendo/{dataset}/{data_type}/"
    else:
        raise ValueError("download_from debe ser 'mtg-fast' o 'mtg'.")

def load_moodtheme_metadata(repo_root: Path) -> Dict[int, List[str]]:
    # Usa el parser oficial del repo para admitir múltiples columnas de tags
    tsv_path = repo_root / "data" / "autotagging_moodtheme.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"No se encuentra {tsv_path}. Verifica REPO_ROOT: {repo_root}")
    scripts_dir = repo_root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import commons  # del repo
    tracks, tags_by_cat, _extra = commons.read_file(str(tsv_path))
    mood_tags_map: Dict[int, List[str]] = {}
    for tid, tinfo in tracks.items():
        moodset = tinfo.get("mood/theme", set())
        mood_tags_map[tid] = sorted(list(moodset))
    return mood_tags_map

def extract_features(audio_path: Path) -> Dict[str, object]:
    # Carga completa del audio
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    audio_essentia = es.MonoLoader(filename=str(audio_path))()

    # Tempo
    tempo_val, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo_val).reshape(()))

    # Loudness (Essentia)
    loudness = es.Loudness()(audio_essentia)

    # Compás heurístico basado en tempo
    def estimate_time_signature_value(tempo_value: float) -> int:
        if tempo_value < 70:
            return 3
        elif tempo_value < 100:
            return 2
        else:
            return 4
    time_signature = estimate_time_signature_value(tempo)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=LIBROSA_N_FFT, hop_length=LIBROSA_HOP)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    # Spectral features
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=LIBROSA_N_FFT, hop_length=LIBROSA_HOP)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=LIBROSA_N_FFT, hop_length=LIBROSA_HOP)))
    rms_energy = float(np.mean(librosa.feature.rms(y=y, frame_length=LIBROSA_N_FFT, hop_length=LIBROSA_HOP)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=LIBROSA_N_FFT, hop_length=LIBROSA_HOP)))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_fft=LIBROSA_N_FFT, hop_length=LIBROSA_HOP), axis=1)

    # Onset strength
    onset_strength = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))

    feat = {
        "tempo": float(tempo),
        "loudness": float(loudness),
        "time_signature": int(time_signature),
        "spectral_centroid": spectral_centroid,
        "zcr": zcr,
        "rms_energy": rms_energy,
        "spectral_rolloff": spectral_rolloff,
        "onset_strength": onset_strength,
    }
    for i, v in enumerate(mfcc_mean, start=1):
        feat[f"mfcc{i}_mean"] = float(v)
    for i, v in enumerate(mfcc_std, start=1):
        feat[f"mfcc{i}_std"] = float(v)
    for i, v in enumerate(chroma, start=1):
        feat[f"chroma{i}_mean"] = float(v)
    return feat

def load_tar_checksums(repo_root: Path, data_type: str) -> List[Tuple[str, str]]:
    base = repo_root / "data" / "download"
    if data_type == "audio":
        fname = base / "autotagging_moodtheme_audio_sha256_tars.txt"
    elif data_type == "audio-low":
        fname = base / "autotagging_moodtheme_audio-low_sha256_tars.txt"
    else:
        raise ValueError("data_type debe ser 'audio' o 'audio-low'.")
    if not fname.exists():
        raise FileNotFoundError(f"No se encuentra {fname}.")
    entries = []
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            sha256 = parts[0]
            tarname = parts[1]
            entries.append((tarname, sha256))
    return entries

def read_processed_tars(checkpoint_csv: Path, output_csv: Path) -> set:
    processed = set()
    if checkpoint_csv.exists():
        try:
            df = pd.read_csv(checkpoint_csv)
            if "tar" in df.columns:
                processed.update(df["tar"].dropna().astype(str).tolist())
        except Exception:
            pass
    if output_csv.exists():
        try:
            df = pd.read_csv(output_csv, usecols=["tar"])
            processed.update(df["tar"].dropna().astype(str).unique().tolist())
        except Exception:
            pass
    return processed

def append_checkpoint(checkpoint_csv: Path, tarname: str) -> None:
    header = not checkpoint_csv.exists()
    with open(checkpoint_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(["tar"])
        writer.writerow([tarname])

def append_rows_to_csv(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=header)

def process_tar(dataset: str,
                data_type: str,
                download_from: str,
                tarname: str,
                expected_sha: str,
                work_dir: Path,
                mood_tags_map: Dict[int, List[str]],
                output_csv: Path,
                checkpoint_csv: Path) -> None:
    base_url = build_base_url(dataset, data_type, download_from)
    tar_url = base_url + tarname
    tar_path = work_dir / tarname

    print(f"\n== Procesando TAR: {tarname} ==")
    if not tar_path.exists():
        download_file(tar_url, tar_path)
    else:
        print(f"Ya existe TAR local: {tar_path}")

    sha_local = compute_sha256(tar_path)
    if sha_local != expected_sha:
        print(f"Checksum inválido para {tarname}. Reintentando descarga...")
        tar_path.unlink(missing_ok=True)
        download_file(tar_url, tar_path)
        sha_local = compute_sha256(tar_path)
        if sha_local != expected_sha:
            raise RuntimeError(f"Checksum no coincide para {tarname} tras reintento.")

    extracted_dir = work_dir / (tarname.replace(".tar", ""))
    extracted_dir.mkdir(parents=True, exist_ok=True)

    rows_to_append: List[Dict[str, object]] = []

    start_tar = time.time()
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".mp3")]
        tar.extractall(path=extracted_dir)
        processed_files = 0
        with tqdm(total=len(members), desc=f"Extracting features from {tarname}") as pbar:
            for m in members:
                file_start = time.time()
                audio_file_path = extracted_dir / m.name
                basename = os.path.basename(m.name)
                try:
                    track_id = int(os.path.splitext(basename)[0].replace(".low", ""))
                except ValueError:
                    track_id = None

                try:
                    feats = extract_features(audio_file_path)
                except Exception as e:
                    print(f"Error extrayendo features de {audio_file_path}: {e}")
                    pbar.update(1)
                    continue

                file_ms = (time.time() - file_start) * 1000.0

                row = {
                    "tar": tarname,
                    "filename": m.name,
                    "basename": basename,
                    "track_id": track_id,
                    "processing_time_ms": round(file_ms, 2),
                }
                row.update(feats)

                mood_tags = []
                if track_id is not None and track_id in mood_tags_map:
                    mood_tags = mood_tags_map[track_id]
                row["moodtheme_tags"] = ";".join(mood_tags)

                rows_to_append.append(row)

                try:
                    audio_file_path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"No se pudo borrar {audio_file_path}: {e}")

                processed_files += 1
                pbar.update(1)

    tar_secs = time.time() - start_tar
    avg_ms = (tar_secs * 1000.0) / max(1, processed_files)
    print(f"Tiempo TAR {tarname}: {round(tar_secs,1)} s, promedio por archivo: {round(avg_ms,1)} ms")

    if rows_to_append:
        append_rows_to_csv(output_csv, rows_to_append)
        print(f"Añadidas {len(rows_to_append)} filas a {OUTPUT_CSV}")

    try:
        shutil.rmtree(extracted_dir, ignore_errors=True)
    except Exception as e:
        print(f"No se pudo borrar carpeta {extracted_dir}: {e}")

    try:
        tar_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"No se pudo borrar TAR {tar_path}: {e}")

    append_checkpoint(checkpoint_csv, tarname)
    print(f"TAR marcado como procesado en {CHECKPOINT_CSV}")

def main():
    dataset = "autotagging_moodtheme"
    print(f"Repo root: {REPO_ROOT}")
    print("Cargando metadatos del subset mood/theme con parser del repo...")
    mood_tags_map = load_moodtheme_metadata(REPO_ROOT)

    print("Cargando lista de TARs y checksums...")
    tar_entries = load_tar_checksums(REPO_ROOT, DATA_TYPE)

    already_processed = read_processed_tars(CHECKPOINT_CSV, OUTPUT_CSV)
    print(f"TARs ya procesados detectados: {len(already_processed)}")

    processed_count = 0
    for tarname, sha in tar_entries:
        if tarname in already_processed:
            print(f"Saltando {tarname}: ya procesado.")
            continue
        process_tar(dataset=dataset,
                    data_type=DATA_TYPE,
                    download_from=DOWNLOAD_FROM,
                    tarname=tarname,
                    expected_sha=sha,
                    work_dir=WORK_DIR,
                    mood_tags_map=mood_tags_map,
                    output_csv=OUTPUT_CSV,
                    checkpoint_csv=CHECKPOINT_CSV)
        processed_count += 1
        if MAX_TARS > 0 and processed_count >= MAX_TARS:
            print(f"Se alcanzó el límite de {MAX_TARS} TARs. Finalizando.")
            break

    print("Proceso por lotes completado.")

if __name__ == "__main__":
    main()