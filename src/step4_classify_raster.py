import numpy as np
import rasterio
from rasterio.windows import Window

# ---------------- TILE-BASED CLASSIFICATION ---------------- #

def classify_tiled(raster_path, clf, meta, tile_size=1024):
    with rasterio.open(raster_path) as src:
        H, W = src.height, src.width
        n_bands = src.count

        # Output array (uint8 classification)
        out_arr = np.zeros((H, W), dtype=np.uint8)

        print(f"Raster size: {W} x {H}")
        print(f"Classifying in {tile_size}x{tile_size} tiles...")

        total_tiles = ((H - 1) // tile_size + 1) * ((W - 1) // tile_size + 1)
        done = 0

        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):

                win = Window(
                    x, y,
                    min(tile_size, W - x),
                    min(tile_size, H - y)
                )

                block = src.read(window=win)
                h, w = block.shape[1], block.shape[2]
                X_block = block.reshape(n_bands, -1).T

                y_pred = clf.predict(X_block)
                out_arr[y:y+h, x:x+w] = y_pred.reshape(h, w)

                # Progress update
                done += 1
                pct = (done / total_tiles) * 100
                if pct % 10 == 0 or done == total_tiles:
                    print(f"\rClassifying tiles: {pct:.2f}% complete", end="")

        print("\nTile classification complete.")

        return out_arr
