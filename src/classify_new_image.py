# ------------------------- IMPORTS ------------------------- #
import joblib
import rasterio
from step1_load_data import load_raster_and_vectors
from step4_classify_raster import classify_tiled
from step5_postprocess_rules import apply_ndvi_rules
from utils import write_geotiff

# ------------------------- PATHS ------------------------- #

NEW_RASTER = "data/raw/QC_268_Stacked_3857.tif"       # <-- change this!
MODEL_PATH = "models/classifier_rf.joblib"  # saved RF model
OUT_PATH   = "data/outputs/classified_new_image.tif"

# ------------------------- MAIN ------------------------- #

def main():
    print("Loading trained classifier...")
    clf = joblib.load(MODEL_PATH)

    print("Loading new UAV raster...")
    # note: we don't need vector/training for new images
    arr, meta, _ = load_raster_and_vectors(NEW_RASTER, None)

    print("Classifying new image (tiled)...")
    rf_map = classify_tiled(NEW_RASTER, clf, meta)

    print("Applying NDVI vegetation splitting...")
    final_map = apply_ndvi_rules(arr, rf_map)

    print("Saving output...")
    write_geotiff(OUT_PATH, final_map, meta)

    print("\nDONE! Classified image saved to:", OUT_PATH)

if __name__ == "__main__":
    main()
