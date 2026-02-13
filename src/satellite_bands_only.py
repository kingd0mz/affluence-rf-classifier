"""
Satellite-only Random Forest evaluation.
Reports Macro F1 score and feature importance.

Purpose:
- Measure how much predictive power comes purely from satellite imagery
- No proximity / POI / building density features
"""
import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from step1_load_data import load_raster_and_vectors
from step2_extract_samples import extract_training_samples
from step4_classify_raster import classify_tiled
from utils import write_geotiff, mask_raster_to_boundary


# ------------------------- CONFIG ------------------------- #
INPUT_RASTER = "data/raw/stacked_17bands.tiff"
TRAIN_VECTOR = "data/training/training_data.gpkg"
OUT_PATH = f"data/outputs/classified_affluence__satellite_only{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"

TEST_SIZE = 0.30
RANDOM_STATE = 42
N_ESTIMATORS = 1200


# --------------------- FEATURE SETS ---------------------- #
FEAT_NAMES = [
    "bu","building_density","glcm_contrast","glcm_entropy",
    "glcm_homogeneity","lst","mndwi","ndbi","ndvi",
    "proximity_park","proximity_resto","proximity_road",
    "proximity_thirdspace","proximity_transport",
    "dem","land_cover","ntl"
]

SATELLITE_FEAT_NAMES = [
    "bu","glcm_contrast","glcm_entropy",
    "glcm_homogeneity","lst","mndwi","ndbi","ndvi",
    "dem","land_cover","ntl"
]


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

    # --- Select satellite-only bands ---
    satellite_indices = [FEAT_NAMES.index(n) for n in SATELLITE_FEAT_NAMES]
    X_sat = X[:, satellite_indices]

    print(f"Using {X_sat.shape[1]} satellite-only bands:")
    print(", ".join(SATELLITE_FEAT_NAMES))

    # --- Train / validation split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_sat,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # --- Train RF ---
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    print("\n=== VALIDATION RESULTS ===")
    y_pred = clf.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average="macro")

    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred, digits=4))

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_val, y_pred))

    print("\n================ RESULTS ================")
    print(f"Macro F1 (satellite-only): {macro_f1:.4f}")

    # --- Feature importance ---
    print("\nFeature importance:")
    importances = clf.feature_importances_
    sorted_feats = sorted(
        zip(SATELLITE_FEAT_NAMES, importances),
        key=lambda x: -x[1]
    )

    for name, imp in sorted_feats:
        print(f"{name:20s} {imp:.4f}")

    print("========================================")

    # --- Step 4: Classify raster ---
    print("STEP 4: Classifying full raster...")
    rf_map = classify_tiled(
        raster_path=INPUT_RASTER,
        clf=clf,
        meta=meta
    )

    # --- Step 5: Save output ---
    print("STEP 5: Saving final classification...")
    write_geotiff(
        path=OUT_PATH,
        array=rf_map,
        meta=meta
    )


if __name__ == "__main__":
    main()
