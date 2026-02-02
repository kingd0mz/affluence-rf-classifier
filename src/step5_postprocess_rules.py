import numpy as np

def apply_ndvi_rules(arr, rf_map):
    ndvi = arr[10, :, :]  # band11 zero-index 10

    out = rf_map.copy()

    veg_mask = (rf_map == 1)

    out[veg_mask & (ndvi <= 0.3)] = 1
    out[veg_mask & (ndvi > 0.3) & (ndvi <= 0.5)] = 2
    out[veg_mask & (ndvi > 0.5)] = 3

    return out
