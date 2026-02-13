#!/usr/bin/env python3

"""
geojson_summary_and_combine.py

1. Summarizes feature counts per GeoJSON
2. Summarizes class_id (1–5) distribution
3. Combines all GeoJSON files
4. Keeps only geometry + class_id
5. Saves combined GeoJSON

Usage:
    python geojson_summary_and_combine.py
"""

import geopandas as gpd
import pandas as pd
import glob
import os
import sys

# ========================= CONFIG =========================
INPUT_FOLDER = "data"      # <-- CHANGE THIS
OUTPUT_FILE = "combined.geojson"
EXPECTED_CLASSES = [1, 2, 3, 4, 5]
# ==========================================================


def main():

    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Folder does not exist: {INPUT_FOLDER}")
        sys.exit(1)

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.geojson"))

    if not files:
        print("No GeoJSON files found.")
        sys.exit(1)

    grand_total_features = 0
    grand_class_counts = {cls: 0 for cls in EXPECTED_CLASSES}
    gdfs = []

    print("\n==============================")
    print("      PER FILE SUMMARY")
    print("==============================")

    for file in sorted(files):

        try:
            gdf = gpd.read_file(file)
        except Exception as e:
            print(f"\nSkipping {file} (error reading file): {e}")
            continue

        filename = os.path.basename(file)
        file_total = len(gdf)
        grand_total_features += file_total

        print(f"\nFile: {filename}")
        print(f"Total features: {file_total}")

        if "class_id" not in gdf.columns:
            print("  WARNING: 'class_id' not found. Skipping file.")
            continue

        # Ensure numeric class_id
        try:
            gdf["class_id"] = gdf["class_id"].astype(int)
        except:
            print("  WARNING: Could not convert class_id to int.")

        # Count classes
        class_counts = gdf["class_id"].value_counts().to_dict()

        for cls in EXPECTED_CLASSES:
            count = class_counts.get(cls, 0)
            grand_class_counts[cls] += count
            print(f"  Class {cls}: {count}")

        # Keep only required columns
        gdf = gdf[["class_id", "geometry"]]

        gdfs.append(gdf)

    # ========================= OVERALL SUMMARY =========================

    print("\n==============================")
    print("        OVERALL SUMMARY")
    print("==============================")

    print(f"\nTOTAL FEATURES (ALL FILES): {grand_total_features}\n")

    for cls in EXPECTED_CLASSES:
        total = grand_class_counts[cls]
        percent = (
            (total / grand_total_features) * 100
            if grand_total_features > 0
            else 0
        )
        print(f"Class {cls}: {total} ({percent:.2f}%)")

    # ========================= COMBINE =========================

    if not gdfs:
        print("\nNo valid GeoDataFrames to combine.")
        sys.exit(1)

    print("\nCombining GeoJSON files...")

    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs=gdfs[0].crs
    )

    # Save
    combined.to_file(OUTPUT_FILE, driver="GeoJSON")

    print(f"\nCombined GeoJSON saved to: {OUTPUT_FILE}")
    print(f"Total combined features: {len(combined)}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
