
import os
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from sam2.run_sam2 import generate_masks
from delineation.object_delineation import delineate
from eval.iou_eval import evaluate_iou
import gc
import torch

def eval_masks_parallel(base_dir, mask_generator, print_csv=False, output_name='Eval_SAM2'):
    image_files = [f for f in os.listdir(base_dir) if f.endswith('.tif')]
    shape_files = [f for f in os.listdir(base_dir) if f.endswith('.shp')]
    results = []

    for tif in image_files:
        base = tif.split('_')[0]
        shapefile = next((s for s in shape_files if base in s), None)
        if not shapefile:
            continue

        image_path = os.path.join(base_dir, tif)
        shape_path = os.path.join(base_dir, shapefile)

        with rasterio.open(image_path) as src:
            transform = src.transform
            img = src.read()
            crs = src.crs
            rgb = np.moveaxis(img[:3], 0, -1) if img.shape[0] >= 3 else np.stack([img[0]] * 3, axis=-1)

        masks = mask_generator.generate(rgb)
        polygons = []
        for m in masks:
            mask_arr = np.array(m['segmentation'], dtype=np.uint8)
            polygons.extend(delineate(mask_arr, transform))

        pred_gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        truth_gdf = gpd.read_file(shape_path)
        if truth_gdf.crs != crs:
            truth_gdf = truth_gdf.to_crs(crs)

        stats = evaluate_iou(truth_gdf, pred_gdf)
        stats['satellite_image'] = tif
        stats['shapefile'] = shapefile
        results.append(stats)

        del masks, img, polygons, rgb
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    if print_csv:
        df.to_csv(f'{output_name}.csv', index=False)
    return df
