import os
import csv
import json
import argparse
from typing import List, Dict, Tuple

def parse_tsv(tsv_path: str) -> List[Dict]:
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows

def normalize_first_tag(tags_field: str) -> str:
    if tags_field is None:
        return ""
    parts = tags_field.strip().split("\t")
    if len(parts) == 0:
        parts = tags_field.strip().split()  # fallback
    first = parts[0] if parts else tags_field.strip()
    # quitar prefijo "mood/theme---"
    if first.startswith("mood/theme---"):
        first = first.replace("mood/theme---", "")
    return first.lower()

def path_to_folder_and_track(path_field: str) -> Tuple[str, str]:
    # Ej: "02/12102.mp3" -> folder="02", track_id="12102"
    segs = path_field.strip().split("/")
    if len(segs) != 2:
        raise ValueError(f"PATH inesperado: {path_field}")
    folder = segs[0]
    track_num = segs[1]
    if not track_num.endswith(".mp3"):
        raise ValueError(f"PATH no termina en .mp3: {path_field}")
    track_id = track_num.replace(".mp3", "")
    return folder, track_id

def find_existing_npy(base_files_dir: str) -> Dict[Tuple[str,str], str]:
    # Retorna mapping ((folder, track_id) -> npy_path) para archivos existentes
    mapping = {}
    if not os.path.isdir(base_files_dir):
        raise FileNotFoundError(f"No existe {base_files_dir}")
    for folder in sorted(os.listdir(base_files_dir)):
        folder_path = os.path.join(base_files_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith(".npy"):
                continue
            track_id = fname[:-4]
            mapping[(folder, track_id)] = os.path.join(folder_path, fname)
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=os.path.join("data", "autotagging_moodtheme.tsv"), help="Ruta del TSV original de MTG")
    ap.add_argument("--files-dir", default=os.path.join("scripts", "download", "files"), help="Base de los .npy")
    ap.add_argument("--out-dir", default=os.path.join("data", "processed", "moodtheme"), help="Salida")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = parse_tsv(args.tsv)
    npy_map = find_existing_npy(args.files_dir)

    out_rows = []
    classes = set()

    for r in rows:
        path_field = r["PATH"]
        tags_field = r["TAGS"]
        duration = r.get("DURATION", "")
        try:
            folder, track_id = path_to_folder_and_track(path_field)
        except Exception as e:
            # saltar filas raras
            continue
        key = (folder, track_id)
        if key not in npy_map:
            continue  # solo quedarnos con los tracks que tienen .npy presente

        label = normalize_first_tag(tags_field)
        if not label:
            continue
        classes.add(label)

        out_rows.append({
            "npy_path": npy_map[key],
            "folder": folder,
            "track_id": track_id,
            "label": label,
            "duration": duration
        })

    # Guardar metadata CSV
    out_csv = os.path.join(args.out_dir, "metadata_moodtheme.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["npy_path", "folder", "track_id", "label", "duration"])
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    # Guardar clases
    class_names = sorted(classes)
    out_json = os.path.join(args.out_dir, "class_names.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)

    print(f"Metadata guardada en: {out_csv} (total {len(out_rows)} filas)")
    print(f"Clases guardadas en: {out_json} (total {len(class_names)} clases)")

if __name__ == "__main__":
    main()