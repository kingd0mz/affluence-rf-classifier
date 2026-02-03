# Affluence RF Classifier

Train a Random Forest on labeled point samples and classify a multi-band raster
into affluence classes. The full pipeline is a single command:

```bash
python src/main.py
```

## How it works

1. Load the stacked raster and the training points.
2. Extract per-pixel samples under each training point.
3. Train a Random Forest and report validation metrics.
4. Classify the entire raster in tiles to keep memory stable.
5. Write the classified GeoTIFF and save the model.

These steps map to the scripts in `src/step1_*` through `src/step4_*`, and are
orchestrated by `src/main.py`.

## Expected inputs

Place your data in the `data/` folder (this directory is git-ignored).

```
data/
  raw/
    stacked_17bands.tiff
  training/
    training_data.gpkg
  outputs/
```

### Raster: `data/raw/stacked_17bands.tiff`

The raster must have 17 bands, ordered to match the feature names used by the
classifier:

1. lst
2. ntl
3. dem
4. glcm_entropy
5. glcm_contrast
6. glcm_homogeneity
7. land_cover
8. building_density
9. proximity_road
10. proximity_thirdspace
11. proximity_transport
12. proximity_resto
13. proximity_park
14. ndvi
15. ndbi
16. bu
17. mndwi

### Training points: `data/training/training_data.gpkg`

A point layer (GPKG or Shapefile) with a `class_id` integer field. The model
expects class labels 1-4.

## Outputs

Running `python src/main.py` produces:

- `data/outputs/classified_affluence_YYYYMMDD_HHMMSS.tif` (classified raster)
- `models/classifier_rf.joblib` (saved Random Forest model)

## Repository structure

```
affluence-rf-classifier/
  README.md
  LICENSE
  .gitignore
  data/
    raw/        # input rasters (ignored in Git)
    training/   # training vectors (ignored in Git)
    outputs/    # classification outputs (ignored in Git)
  models/       # saved classifiers
  src/
    step1_load_data.py
    step2_extract_samples.py
    step3_train_classifier.py
    step4_classify_raster.py
    utils.py
    main.py
  requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
