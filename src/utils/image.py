
import cv2
import rasterio
import numpy as np

def create_color_image(red_path, green_path, blue_path):
    red = cv2.imread(red_path, cv2.IMREAD_GRAYSCALE)
    green = cv2.imread(green_path, cv2.IMREAD_GRAYSCALE)
    blue = cv2.imread(blue_path, cv2.IMREAD_GRAYSCALE)

    if red is None or green is None or blue is None:
        raise ValueError('Failed to load one or more bands.')

    if not (red.shape == green.shape == blue.shape):
        raise ValueError('Bands must have same dimensions.')

    return cv2.merge([red, green, blue])

def create_rgba_image(red, green, blue):
    alpha = np.where((red == 0) & (green == 0) & (blue == 0), 0, 255).astype(np.uint8)
    return cv2.merge([red, green, blue, alpha])

def save_raster(output_path, data, reference_path, count=3, dtype='uint8'):
    with rasterio.open(reference_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs

    meta.update({
        'count': count,
        'dtype': dtype,
        'driver': 'GTiff',
        'transform': transform,
        'crs': crs
    })

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data.transpose(2, 0, 1))
