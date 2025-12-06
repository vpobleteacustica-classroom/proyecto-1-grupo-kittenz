import os
import csv
import json
import argparse

def parse_tsv(tsv_path):
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows

def path_to_folder_and_track(path_field):
    segs = path_field.strip().split("/")
    folder = segs[0]
    track_id = segs[1].replace(".mp3","")
    return folder, track_id

def normalize_tags(tags_field):
    # El campo TAGS trae etiquetas separadas por tabs; las normalizamos
    if tags_field is None:
        return []
    parts = tags_field.strip().split("\t")
    out = []
    for p in parts:
        p = p.strip()
        if p.startswith("mood/theme---"):
            p = p.replace("mood/theme---", "")
        if p:
            out.append(p.lower())
    # quitar duplicados conservando orden
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def find_existing_npy(base_files_dir):
    mapping = {}
    for folder in sorted(os.listdir(base_files_dir)):
        folder_path = os.path.join(base_files_dir, folder)
        if not os.path.isdir(folder_path): continue
        for fname in os.listdir(folder_path):
            if not fname.endswith(".npy"): continue
            track_id = fname[:-4]
            mapping[(folder, track_id)] = os.path.join(folder_path, fname)
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=os.path.join("data", "autotagging_moodtheme.tsv"))
    ap.add_argument("--files-dir", default=os.path.join("scripts", "download", "files"))
    ap.add_argument("--out-dir", default=os.path.join("data", "processed", "moodtheme"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = parse_tsv(args.tsv)
    npy_map = find_existing_npy(args.files_dir)

    out_rows = []
    class_set = set()

    for r in rows:
        folder, track_id = path_to_folder_and_track(r["PATH"])
        key = (folder, track_id)
        if key not in npy_map:
            continue
        tags = normalize_tags(r["TAGS"])
        if not tags:
            continue
        # limitar vocabulario a las etiquetas observadas en tus npy
        for t in tags:
            class_set.add(t)
        out_rows.append({
            "npy_path": npy_map[key],
            "folder": folder,
            "track_id": track_id,
            "tags": ";".join(tags),  # separador por ';'
            "duration": r.get("DURATION","")
        })

    # Guardar metadata
    meta_csv = os.path.join(args.out_dir, "metadata_moodtheme_multilabel.csv")
    with open(meta_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["npy_path","folder","track_id","tags","duration"])
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    class_names = sorted(class_set)
    classes_json = os.path.join(args.out_dir, "class_names_multilabel.json")
    with open(classes_json, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)

    print(f"Metadata multilabel: {meta_csv} ({len(out_rows)} filas)")
    print(f"Clases multilabel: {classes_json} ({len(class_names)} clases)")

if __name__ == "__main__":
    main()