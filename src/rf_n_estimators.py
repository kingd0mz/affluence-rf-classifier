"""
Random Forest sensitivity analysis:
Impact of number of estimators on Macro F1 score (all bands).

Outputs:
- CSV with n_estimators vs Macro F1
"""

import os
import csv
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from step1_load_data import load_raster_and_vectors
from step2_extract_samples import extract_training_samples


# ------------------------- CONFIG ------------------------- #
INPUT_RASTER = "data/raw/stacked_17bands.tiff"
TRAIN_VECTOR = "data/training/training_data.gpkg"

OUT_DIR = "data/outputs"
OUT_CSV = os.path.join(OUT_DIR, "rf_n_estimators_f1.csv")

TEST_SIZE = 0.30
RANDOM_STATE = 42

# Estimator sweep
N_ESTIMATORS_LIST = list(range(100, 2100, 100))  # 100 → 1000


def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("Loading raster and training data...")
    arr, meta, gdf = load_raster_and_vectors(
        raster_path=INPUT_RASTER,
        vector_path=TRAIN_VECTOR
    )

    print("Extracting training samples...")
    X, y = extract_training_samples(
        arr=arr,
        raster_path=INPUT_RASTER,
        gdf=gdf
    )

    # Train / validation split (fixed split for fair comparison)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    ensure_outdir()

    results = []

    print("\nRunning estimator sweep...")
    for n_estimators in N_ESTIMATORS_LIST:
        print(f"Training RF with n_estimators = {n_estimators}...")

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        macro_f1 = f1_score(y_val, y_pred, average="macro")

        results.append({
            "n_estimators": n_estimators,
            "macro_f1": float(macro_f1),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val))
        })

        print(f"  → Macro F1: {macro_f1:.4f}")

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n_estimators", "macro_f1", "n_train", "n_val"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nDone.")
    print("CSV saved to:", OUT_CSV)


if __name__ == "__main__":
    main()
