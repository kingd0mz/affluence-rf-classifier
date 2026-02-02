import rasterio

def write_geotiff(path, array, meta):
    meta2 = meta.copy()
    meta2.update(
        dtype="uint8",
        count=1,
        nodata=0
    )

    with rasterio.open(path, "w", **meta2) as dst:
        dst.write(array.astype("uint8"), 1)
