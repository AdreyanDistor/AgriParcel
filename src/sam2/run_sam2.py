
import os
import rasterio
import numpy as np
from rasterio.plot import reshape_as_image
from utils.io import list_images, get_basename
from sam2.mask_utils import save_mask
import gc
import torch
from concurrent.futures import ThreadPoolExecutor

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
