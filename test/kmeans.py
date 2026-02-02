import rasterio
import numpy as np
from sklearn.cluster import KMeans

input_tif = "input.tif"
output_tif = "output_5classes.tif"

with rasterio.open("input.tif") as src:
    arr = src.read()  # (bands, rows, cols)
    meta = src.meta.copy()

bands, rows, cols = arr.shape
pixels = arr.reshape(bands, -1).T  # (n_pixels, n_bands)

kmeans = KMeans(n_clusters=7, random_state=42)
labels = kmeans.fit_predict(pixels)

out = labels.reshape(rows, cols).astype("uint8") + 1

meta.update(dtype="uint8", count=1)

with rasterio.open("output_7classes.tif", "w", **meta) as dst:
    dst.write(out, 1)
