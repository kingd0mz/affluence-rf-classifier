"""Evaluate all 5-band combinations (out of 17) and record Macro F1 and per-class recall.

Usage: run directly. Options at top of file for quick adjustments (n_estimators, sample_fraction).
"""
import os
import itertools
import datetime
import csv

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score

from step1_load_data import load_raster_and_vectors
from step2_extract_samples import extract_training_samples


# ------------------------- CONFIG ------------------------- #
INPUT_RASTER = "data/raw/stacked_17bands.tiff"
TRAIN_VECTOR = "data/training/training_data.gpkg"
OUT_DIR = "data/outputs"
OUT_CSV = os.path.join(OUT_DIR, "band_search_results.csv")
OUT_TXT = os.path.join(OUT_DIR, "band_search_results.txt")

# Random forest / sampling defaults -- lower values speed up testing.
N_ESTIMATORS = 200
TEST_SIZE = 0.30
RANDOM_STATE = 42

# Optional: limit number of combinations (for quick testing). Set to None to run all.
# Default to 50 combos for quick evaluation as requested.
MAX_COMBOS = None


BAND_NAMES = [
        "bu","building_density","glcm_contrast","glcm_entropy",
        "glcm_homogeneity","lst","mndwi","ndbi","ndvi",
        "proximity_park","proximity_resto","proximity_road",
        "proximity_thirdspace","proximity_transport","dem","land_cover","ntl"
    ]


def ensure_outdir():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)


def evaluate_combo(X, y, indices):
    X_sub = X[:, indices]

    X_train, X_val, y_train, y_val = train_test_split(
        X_sub, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)

    macro_f1 = f1_score(y_val, y_pred, average="macro")

    labels = np.unique(y)
    recalls = recall_score(y_val, y_pred, labels=labels, average=None)
    recall_dict = {int(lbl): float(rec) for lbl, rec in zip(labels, recalls)}

    return macro_f1, recall_dict


def main():
    print("STEP 1: Loading raster and training data...")
    arr, meta, gdf = load_raster_and_vectors(raster_path=INPUT_RASTER, vector_path=TRAIN_VECTOR)

    print("STEP 2: Extracting training samples...")
    X, y = extract_training_samples(arr=arr, raster_path=INPUT_RASTER, gdf=gdf)

    n_bands = X.shape[1]
    assert n_bands == len(BAND_NAMES), f"Expected {len(BAND_NAMES)} band names, found {n_bands} bands"

    # --- Determine most important feature by training a quick RF on all bands ---
    print("Training quick RF to estimate global feature importance...")
    quick_clf = RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                       random_state=RANDOM_STATE, n_jobs=-1)
    quick_clf.fit(X, y)
    importances = quick_clf.feature_importances_
    top_idx = int(np.argmax(importances))
    print(f"Most important feature: index {top_idx+1} -> {BAND_NAMES[top_idx]}")

    # --- Generate combinations that include the top feature ---
    remaining = [i for i in range(n_bands) if i != top_idx]
    combos = []
    for comb in itertools.combinations(remaining, 4):
        full = tuple(sorted((top_idx,) + comb))
        combos.append(full)

    if MAX_COMBOS is not None:
        combos = combos[:MAX_COMBOS]

    print(f"Evaluating {len(combos)} combinations (including top feature) from {n_bands} bands...")

    ensure_outdir()

    # CSV header
    labels_all = sorted(np.unique(y).tolist())
    recall_headers = [f"recall_cls_{lbl}" for lbl in labels_all]
    header = ["combo_idx", "band_indices_1based", "band_names", "macro_f1"] + recall_headers + ["n_samples"]

    results = []
    for i, combo in enumerate(combos, start=1):
        indices = list(combo)
        band_names = [BAND_NAMES[idx] for idx in indices]

        macro_f1, recall_dict = evaluate_combo(X, y, indices)

        row = {
            "combo_idx": i,
            "band_indices_1based": ",".join(str(idx + 1) for idx in indices),
            "band_names": ",".join(band_names),
            "macro_f1": float(macro_f1),
            "n_samples": int(len(y))
        }

        for lbl in labels_all:
            row[f"recall_cls_{lbl}"] = recall_dict.get(int(lbl), 0.0)

        results.append(row)

        if i % 100 == 0:
            print(f"Processed {i}/{len(combos)} combos. Latest macro F1: {macro_f1:.4f} -- {band_names}")

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Write sorted TXT summary (top 50)
    results_sorted = sorted(results, key=lambda x: -x["macro_f1"])
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = OUT_TXT.replace('.txt', f"_{now}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Band search results - top combos by Macro F1 ({now})\n")
        f.write("Columns: combo_idx, band_indices_1based, band_names, macro_f1, ")
        f.write(",".join(recall_headers) + ", n_samples\n\n")

        top_n = min(50, len(results_sorted))
        for r in results_sorted[:top_n]:
            line = [str(r[head]) for head in header]
            f.write("\t".join(line) + "\n")

    print("Done. CSV saved to:", OUT_CSV)
    print("Summary (top combos) saved to:", summary_path)


if __name__ == "__main__":
    main()
