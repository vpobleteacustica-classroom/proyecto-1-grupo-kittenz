import os
import sys
import json
import numpy as np
from collections import Counter

def is_npy_file(path: str) -> bool:
    return path.lower().endswith(".npy")

def probe_npy(path: str):
    try:
        arr = np.load(path)
    except Exception as e:
        return {"path": path, "error": str(e)}

    info = {
        "path": path,
        "dtype": str(arr.dtype),
        "shape": tuple(arr.shape),
        "ndim": arr.ndim,
        "min": float(np.min(arr)) if arr.size > 0 else None,
        "max": float(np.max(arr)) if arr.size > 0 else None,
        "mean": float(np.mean(arr)) if arr.size > 0 else None,
        "std": float(np.std(arr)) if arr.size > 0 else None,
        "has_nan": bool(np.isnan(arr).any()) if arr.size > 0 else False,
        "has_inf": bool(np.isinf(arr).any()) if arr.size > 0 else False,
    }

    # Heurística de orientación:
    orient = {"n_mels": None, "time": None, "channels": None, "layout": "unknown"}
    if info["ndim"] == 2:
        a, b = info["shape"]
        n_mels_candidate = min(a, b)
        time_candidate = max(a, b)
        orient["n_mels"] = n_mels_candidate
        orient["time"] = time_candidate
        if a < b:
            orient["layout"] = "[n_mels, time]"
        else:
            orient["layout"] = "[time, n_mels] (transpose needed)"
    elif info["ndim"] == 3:
        c, h, w = info["shape"]
        # Caso típico: (channels, n_mels, time)
        if c in (1,):
            orient["channels"] = c
            orient["n_mels"] = h
            orient["time"] = w
            orient["layout"] = "[channels, n_mels, time]"
        else:
            # Intento general: menor dimensión ~ n_mels, mayor ~ time
            dims = sorted([(0, c), (1, h), (2, w)], key=lambda x: x[1])
            n_mels_dim_idx, n_mels_val = dims[0]
            time_dim_idx, time_val = dims[-1]
            channels_dim_idx, channels_val = dims[1]
            orient["n_mels"] = n_mels_val
            orient["time"] = time_val
            orient["channels"] = channels_val if channels_val in (1, 2, 3) else None
            orient["layout"] = f"3D mixed (n_mels@dim{n_mels_dim_idx}, time@dim{time_dim_idx}, ch@dim{channels_dim_idx})"

    info["orientation"] = orient
    return info

def main():
    # Directorio base por defecto: scripts/download/files
    base_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("scripts", "download", "files")
    if not os.path.isdir(base_dir):
        print(f"ERROR: No existe el directorio base: {base_dir}")
        print("Uso: python scripts/download/050_probe_npy.py [directorio_base_files]")
        sys.exit(1)

    results = []
    shape_counter = Counter()
    n_mels_counter = Counter()
    time_counter = Counter()
    dtype_counter = Counter()
    layout_counter = Counter()
    stats_global = {
        "count": 0,
        "min_global": None,
        "max_global": None,
        "mean_global": None,
        "std_global": None,
    }
    means, stds, mins, maxs = [], [], [], []

    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if not is_npy_file(fname):
                continue
            fpath = os.path.join(folder_path, fname)
            info = probe_npy(fpath)
            results.append(info)
            if "error" in info:
                print(f"ERROR leyendo {fpath}: {info['error']}")
                continue
            shape_counter[info["shape"]] += 1
            dtype_counter[info["dtype"]] += 1
            layout_counter[info["orientation"]["layout"]] += 1
            nm = info["orientation"]["n_mels"]
            tt = info["orientation"]["time"]
            if nm is not None:
                n_mels_counter[nm] += 1
            if tt is not None:
                time_counter[tt] += 1

            stats_global["count"] += 1
            if info["min"] is not None:
                mins.append(info["min"])
            if info["max"] is not None:
                maxs.append(info["max"])
            if info["mean"] is not None:
                means.append(info["mean"])
            if info["std"] is not None:
                stds.append(info["std"])

    if mins:
        stats_global["min_global"] = float(np.min(mins))
    if maxs:
        stats_global["max_global"] = float(np.max(maxs))
    if means:
        stats_global["mean_global"] = float(np.mean(means))
    if stds:
        stats_global["std_global"] = float(np.mean(stds))

    summary = {
        "base_dir": base_dir,
        "counts": {
            "total_files": stats_global["count"],
            "by_shape": shape_counter.most_common(),
            "by_dtype": dtype_counter.most_common(),
            "by_layout": layout_counter.most_common(),
        },
        "orientation": {
            "n_mels_distribution": n_mels_counter.most_common(),
            "time_distribution": time_counter.most_common(),
        },
        "value_stats": stats_global,
        "examples": results[:50],  # primeras 50 para inspección
    }

    # Imprimir resumen
    print("=== Resumen de espectrogramas (.npy) ===")
    print(f"Directorio base: {summary['base_dir']}")
    print(f"Total de archivos .npy: {summary['counts']['total_files']}")
    print("\nShapes más comunes:")
    for shp, cnt in summary["counts"]["by_shape"]:
        print(f"  {shp}: {cnt}")
    print("\nDtypes más comunes:")
    for dt, cnt in summary["counts"]["by_dtype"]:
        print(f"  {dt}: {cnt}")
    print("\nLayouts detectados:")
    for ly, cnt in summary["counts"]["by_layout"]:
        print(f"  {ly}: {cnt}")

    print("\nDistribución n_mels (heurística):")
    for val, cnt in summary["orientation"]["n_mels_distribution"]:
        print(f"  {val}: {cnt}")

    print("\nDistribución time (heurística):")
    for val, cnt in summary["orientation"]["time_distribution"]:
        print(f"  {val}: {cnt}")

    print("\nEstadísticas globales de valores:")
    print(json.dumps(summary["value_stats"], indent=2))

    # Guardar JSON
    out_json = os.path.join("scripts", "download", "probe_summary.json")
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResumen completo guardado en: {out_json}")
    except Exception as e:
        print(f"\nNo se pudo guardar el JSON: {e}")

    # Sugerencias
    if summary["orientation"]["n_mels_distribution"]:
        suggested_n_mels = summary["orientation"]["n_mels_distribution"][0][0]
        print(f"\nSugerencia: usar n_mels fijo = {suggested_n_mels} (más frecuente)")
    if summary["orientation"]["time_distribution"]:
        # usar percentil 75 de los valores de time presentes en los ejemplos guardados
        time_vals = [r["orientation"]["time"] for r in results if "orientation" in r and r["orientation"]["time"] is not None]
        if time_vals:
            suggested_time = int(np.percentile(time_vals, 75))
            print(f"Sugerencia: fijar longitud temporal ~ {suggested_time} frames (p75)")

if __name__ == "__main__":
    main()