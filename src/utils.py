import rasterio
from rasterio.mask import mask
import geopandas as gpd


def mask_raster_to_boundary(
    raster_path: str,
    boundary_geojson: str,
    output_raster: str
):
    """
    Mask and crop a raster to an administrative boundary.
    Assumes CRS is already aligned (no reprojection).
    """

    # Load boundary
    gdf = gpd.read_file(boundary_geojson)

    with rasterio.open(raster_path) as src:

        if src.crs is None:
            raise ValueError("Raster has no CRS")

        if gdf.crs != src.crs:
            raise ValueError(
                f"CRS mismatch: raster={src.crs}, boundary={gdf.crs}"
            )

        masked, transform = mask(
            dataset=src,
            shapes=gdf.geometry,
            crop=True,
            nodata=src.nodata
        )

        meta = src.meta.copy()
        meta.update({
            "height": masked.shape[1],
            "width": masked.shape[2],
            "transform": transform
        })

    with rasterio.open(output_raster, "w", **meta) as dst:
        dst.write(masked)


def write_geotiff(path, array, meta):
    """
    Write a single-band uint8 GeoTIFF with safe nodata handling.
    Classes expected: 1–4
    """

    meta_out = meta.copy()
    meta_out.update(
        dtype="uint8",
        count=1,
        nodata=255
    )

    with rasterio.open(path, "w", **meta_out) as dst:
        dst.write(array.astype("uint8"), 1)
