# ------------------------- IMPORTS ------------------------- #
from step1_load_data import load_raster_and_vectors
from step2_extract_samples import extract_training_samples
from step3_train_classifier import train_classifier
from step4_classify_raster import classify_tiled
from utils import write_geotiff


RAW_RASTER = "data/raw/stacked_17bands.tiff"
TRAIN_VECTOR = "data/training/training_data.gpkg"
OUT_PATH = "data/outputs/classified_affluence.tif"

# ------------------------- MAIN PIPELINE ------------------------- #

def main():
    print("STEP 1: Loading raster and training data...")
    arr, meta, gdf = load_raster_and_vectors(RAW_RASTER, TRAIN_VECTOR)

    print("STEP 2: Extracting training samples...")
    X, y = extract_training_samples(arr, RAW_RASTER, gdf)

    print("STEP 3: Training classifier...")
    clf = train_classifier(X, y)

    print("STEP 4: Classifying full raster...")
    rf_map = classify_tiled(RAW_RASTER, clf, meta)

    print("Saving final classification...")
    write_geotiff(OUT_PATH, final_map, meta)

    print("DONE! Output saved to:", OUT_PATH)


if __name__ == "__main__":
    main()
