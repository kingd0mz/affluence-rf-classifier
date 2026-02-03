# ------------------------- IMPORTS ------------------------- #
import os
import datetime

from step1_load_data import load_raster_and_vectors
from step2_extract_samples import extract_training_samples
from step3_train_classifier import train_classifier
from step4_classify_raster import classify_tiled
from utils import write_geotiff, mask_raster_to_boundary


# ------------------------- PATHS ------------------------- #
INPUT_RASTER = "data/raw/stacked_17bands.tiff"
TRAIN_VECTOR = "data/training/training_data.gpkg"
# ADMIN_BDRY = "data/raw/ncr_admbdry.geojson"
OUT_PATH = f"data/outputs/classified_affluence_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"


# ------------------------- MAIN PIPELINE ------------------------- #
def main():

    # --- Step 1: Load raster and data ---
    print("STEP 1: Loading raster and training data...")
    arr, meta, gdf = load_raster_and_vectors(
        raster_path=INPUT_RASTER,
        vector_path=TRAIN_VECTOR
    )

    # --- Step 2: Extract samples ---
    print("STEP 2: Extracting training samples...")
    X, y = extract_training_samples(
        arr=arr,
        raster_path=INPUT_RASTER,
        gdf=gdf
    )

    # --- Step 3: Train classifier ---
    print("STEP 3: Training classifier...")
    clf = train_classifier(X, y)

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

    print("DONE! Output saved to:", OUT_PATH)


if __name__ == "__main__":
    main()
