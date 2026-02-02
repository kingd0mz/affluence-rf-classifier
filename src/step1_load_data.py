import numpy as np
import rasterio
import geopandas as gpd

# def load_raster_and_vectors(raster_path, vector_path):
#     src = rasterio.open(raster_path)
#     arr = src.read()  # (13, H, W)

#     # Add random band (band 14)
#     H, W = arr.shape[1], arr.shape[2]
#     random_band = np.random.rand(H, W).astype(arr.dtype)
#     arr = np.vstack([arr, random_band[np.newaxis, :, :]])

#     # Update metadata
#     meta = src.meta.copy()
#     meta["count"] = 14

#     # Load vector and reproject if needed
#     gdf = gpd.read_file(vector_path)
#     if gdf.crs != src.crs:
#         gdf = gdf.to_crs(src.crs)

#     return arr, meta, gdf

def load_raster_and_vectors(raster_path, vector_path=None):
    src = rasterio.open(raster_path)
    arr = src.read()
    meta = src.meta.copy()

    gdf = None
    if vector_path is not None:
        gdf = gpd.read_file(vector_path)
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

    return arr, meta, gdf
