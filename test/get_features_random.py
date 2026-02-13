import json
import random
import sys

# ---------------------------
# CONFIG
# ---------------------------
INPUT_GEOJSON = "data/E_1.geojson"
OUTPUT_GEOJSON = f"new_{INPUT_GEOJSON.split('.')[0]}.geojson"
Y = 50  # number of features to randomly select
RANDOM_SEED = 42  # optional for reproducibility
# ---------------------------


def sample_geojson(input_path, output_path, y, seed=None):
    if seed is not None:
        random.seed(seed)

    # Load GeoJSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data["type"] != "FeatureCollection":
        raise ValueError("Input GeoJSON must be a FeatureCollection")

    features = data["features"]
    total_features = len(features)

    print(f"Total features in file: {total_features}")

    if y > total_features:
        raise ValueError(f"Requested {y} features but only {total_features} available.")

    # Random sample without replacement
    sampled_features = random.sample(features, y)

    # Create new FeatureCollection
    new_geojson = {
        "type": "FeatureCollection",
        "features": sampled_features
    }

    # Preserve CRS if present
    if "crs" in data:
        new_geojson["crs"] = data["crs"]

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_geojson, f, indent=2)

    print(f"Saved {y} random features to {output_path}")


if __name__ == "__main__":
    sample_geojson(INPUT_GEOJSON, OUTPUT_GEOJSON, Y, RANDOM_SEED)
