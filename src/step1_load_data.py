import numpy as np
import rasterio
import geopandas as gpd

def load_raster_and_vectors(raster_path, vector_path=None):
    src = rasterio.open(raster_path)
    arr = src.read()
    meta = src.meta.copy()

    gdf = None
    if vector_path is not None:
        gdf = gpd.read_file(vector_path)
        print("Raster CRS:", src.crs)
        print("Vector CRS:", gdf.crs)
        # if gdf.crs != src.crs:
        #     gdf = gdf.to_crs(src.crs)

    return arr, meta, gdf
