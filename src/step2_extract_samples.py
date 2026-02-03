import numpy as np
import rasterio

def extract_training_samples(arr, raster_path, gdf):
    """
    Extract per-pixel training samples from a raster using point geometries.

    Assumptions:
    - gdf.geometry contains POINT geometries
    - gdf['class_id'] contains labels (may include NaN)
    - arr shape: (bands, rows, cols)
    """

    # --- Extract point coordinates ---
    coords = [(geom.x, geom.y) for geom in gdf.geometry]

    with rasterio.open(raster_path) as src:
        rows_cols = [src.index(x, y) for x, y in coords]

    rows = np.array([rc[0] for rc in rows_cols])
    cols = np.array([rc[1] for rc in rows_cols])

    _, H, W = arr.shape

    # --- Spatial validity mask ---
    valid_mask = (
        (rows >= 0) & (rows < H) &
        (cols >= 0) & (cols < W)
    )

    dropped = len(rows) - np.count_nonzero(valid_mask)
    if dropped > 0:
        print(f"WARNING: {dropped} training points were outside raster bounds and removed.")

    # --- Apply spatial mask ---
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    gdf_valid = gdf.loc[valid_mask].copy()

    # --- Drop missing class labels ---
    before = len(gdf_valid)
    gdf_valid = gdf_valid.dropna(subset=["class_id"])
    after = len(gdf_valid)

    if before != after:
        print(f"WARNING: {before - after} training points had NaN class_id and were removed.")

    # --- Final labels ---
    y = gdf_valid["class_id"].astype(int).values

    # --- Align rows/cols with remaining labels ---
    rows = rows[:len(y)]
    cols = cols[:len(y)]

    # --- Extract pixel values ---
    X = arr[:, rows, cols].T  # shape: (n_samples, n_bands)

    # --- Final sanity check ---
    assert len(X) == len(y), "X and y length mismatch after filtering"
    assert set(np.unique(y)).issubset({1, 2, 3, 4}), \
        f"Unexpected class labels found: {np.unique(y)}"

    return X, y
