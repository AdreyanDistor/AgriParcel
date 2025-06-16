import sys
sys.path.insert(0, 'src/utils')
sys.path.insert(0, 'src/sam2')

import os
import rasterio
import numpy as np
from rasterio.plot import reshape_as_image
from io import *
from mask_utils import save_mask
import gc
import torch
from concurrent.futures import ThreadPoolExecutor


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_images(folder, extensions=('.tif', '.tiff', '.jpg', '.png')):
    return [f for f in os.listdir(folder) if f.lower().endswith(extensions)]

def get_basename(path):
    return os.path.splitext(os.path.basename(path))[0]


def generate_masks(directory, output_dir, mask_generator, return_segmented=True, name='mask'):
    images = list_images(directory, extensions=('.tif', '.tiff'))
    os.makedirs(output_dir, exist_ok=True)

    for img_file in images:
        img_path = os.path.join(directory, img_file)
        name_base = get_basename(img_file)

        if any(f.startswith(f'{name_base}_{name}_') for f in os.listdir(output_dir)):
            print(f'Skipping {name_base} — already processed.')
            continue

        with rasterio.open(img_path) as src:
            img_data = src.read()
            profile = src.profile
            transform = src.transform
            crs = src.crs

            img_rgb = reshape_as_image(img_data[:3] if img_data.shape[0] >= 3 else np.stack([img_data[0]] * 3, axis=0))

        masks = mask_generator.generate(img_rgb)

        with ThreadPoolExecutor() as pool:
            for i, mask in enumerate(masks):
                pool.submit(save_mask, mask, img_rgb, transform, profile, crs, name_base, i, output_dir, return_segmented, name)

        del img_data, img_rgb, masks
        gc.collect()
        torch.cuda.empty_cache()
