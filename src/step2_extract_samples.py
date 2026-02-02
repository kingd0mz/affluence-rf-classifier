import numpy as np
import rasterio

def extract_training_samples(arr, raster_path, gdf):
    coords = [(geom.x, geom.y) for geom in gdf.geometry]

    with rasterio.open(raster_path) as src:
        rows_cols = [src.index(x, y) for x, y in coords]

    rows = np.array([rc[0] for rc in rows_cols])
    cols = np.array([rc[1] for rc in rows_cols])

    H, W = arr.shape[1], arr.shape[2]

    # VALID MASK
    valid_mask = (
        (rows >= 0) & (rows < H) &
        (cols >= 0) & (cols < W)
    )

    # Count dropped points
    dropped = len(rows) - np.count_nonzero(valid_mask)
    if dropped > 0:
        print(f"WARNING: {dropped} training points were outside raster bounds and removed.")

    # Filter rows/cols and labels
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    y = gdf["class_id"].astype(int).values[valid_mask]

    # Extract pixel values safely
    X = arr[:, rows, cols].T

    return X, y
