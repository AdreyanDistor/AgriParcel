import os
import sys
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
import geopandas as gpd
from torchvision import transforms
from tqdm import tqdm
from shapely.geometry import box
from shapely.ops import unary_union
from rasterio.features import rasterize
from scipy.ndimage import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from clip_test_model import load_model
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def iou_masks(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

def coverage_fn(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    area_mask1 = mask1.sum()
    return intersection / area_mask1 if area_mask1 > 0 else 0.0

def save_mask_overlay(image, pred_mask, ground_truth_mask, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    red_cmap = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 0.4]])
    green_cmap = ListedColormap([[0, 0, 0, 0], [0, 1, 0, 0.4]])
    plt.imshow(pred_mask, cmap=red_cmap)
    plt.imshow(ground_truth_mask, cmap=green_cmap)
    handles = [
        plt.Line2D([0], [0], color='red', lw=4, label='Predicted'),
        plt.Line2D([0], [0], color='green', lw=4, label='Ground Truth')
    ]
    plt.legend(handles=handles, loc='lower right')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_bbox_from_mask(mask_array):
    labeled_mask, _ = label(mask_array)
    props = regionprops(labeled_mask)
    if not props:
        return None
    min_row, min_col, max_row, max_col = props[0].bbox
    return min_row, max_row, min_col, max_col

def eval_image(image_path, ground_truth_geom, mask_generator, clip_model, preprocess, results_dir, class_names, confidence_threshold, device):
    image = np.array(Image.open(image_path).convert('RGB'))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    masks_data = mask_generator.generate(image)
    masks = [np.array(mask['segmentation'], dtype=np.uint8) for mask in masks_data]

    with rasterio.open(image_path) as src:
        transform = src.transform
        profile = src.profile
        crs = src.crs

    valid_masks = []

    for idx, mask_array in enumerate(masks):
        bbox = get_bbox_from_mask(mask_array)
        if bbox is None:
            continue

        min_row, max_row, min_col, max_col = bbox
        height, width = mask_array.shape
        min_row = max(min_row - 1, 0)
        max_row = min(max_row + 1, height)
        min_col = max(min_col - 1, 0)
        max_col = min(max_col + 1, width)

        cropped_mask = mask_array[min_row:max_row, min_col:max_col]
        cropped_image = image[min_row:max_row, min_col:max_col]

        alpha = (cropped_mask * 255).astype(np.uint8)
        segmented = np.dstack((cropped_image, alpha))
        output_data = np.moveaxis(segmented, -1, 0)

        west, north = transform * (min_col, min_row)
        new_transform = rasterio.transform.from_origin(west, north, transform.a, -transform.e)

        output_profile = profile.copy()
        output_profile.update(
            height=output_data.shape[1],
            width=output_data.shape[2],
            dtype=rasterio.uint8,
            count=4,
            transform=new_transform,
            crs=crs
        )

        os.makedirs(results_dir, exist_ok=True)
        mask_filename = f'{image_name}_mask_{idx}.tif'
        mask_path = os.path.join(results_dir, mask_filename)

        with rasterio.open(mask_path, 'w', **output_profile) as dst:
            dst.write(output_data)

        image_t = preprocess(Image.fromarray(cropped_image)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_t)
            logits = clip_model.logit_scale.exp() * image_features @ clip_model.text_projection.t()
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        pred_class = class_names[np.argmax(probs)]
        confidence = np.max(probs)

        if pred_class == 'Farmland' and confidence >= confidence_threshold:
            valid_masks.append(mask_array)
        else:
            os.remove(mask_path)

    if not valid_masks:
        return 0.0, 0.0

    pred_mask = np.logical_or.reduce(valid_masks).astype(np.uint8)
    with rasterio.open(image_path) as src:
        transform = src.transform
    image_bounds = rasterio.transform.array_bounds(image.shape[0], image.shape[1], transform)
    image_box = box(*image_bounds)
    clipped_geoms = [geom.intersection(image_box) for geom in ground_truth_geom if geom.is_valid and not geom.is_empty]
    ground_truth_mask = rasterize(
        [(geom, 1) for geom in clipped_geoms if not geom.is_empty],
        out_shape=image.shape[:2],
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8
    )

    overlay_path = os.path.join(results_dir, f'{image_name}_overlay.png')
    gt_mask_path = os.path.join(results_dir, f'{image_name}_gt_mask.png')
    Image.fromarray(ground_truth_mask * 255).save(gt_mask_path)
    save_mask_overlay(image, pred_mask, ground_truth_mask, overlay_path)
    print('mask saved')

    coverage_value = coverage_fn(pred_mask, ground_truth_mask)
    iou_value = iou_masks(pred_mask, ground_truth_mask)
    return coverage_value, iou_value

# Main pipeline setup

main_dir = '/rhome/adist003/bigdata/eval_sam2'
tif_files = sorted([os.path.join(main_dir, f) for f in os.listdir(main_dir) if f.endswith('.tif')])
shp_files = sorted([os.path.join(main_dir, f) for f in os.listdir(main_dir) if f.endswith('.shp')])

image_dirs = tif_files
geometry_list = []

for tif_path, shp_path in zip(tif_files, shp_files):
    with rasterio.open(tif_path) as src:
        image_crs = src.crs
    gdf = gpd.read_file(shp_path)
    if gdf.crs != image_crs:
        gdf = gdf.to_crs(image_crs)
    geometry_list.append(gdf.geometry)

clip_model_path = '/rhome/adist003/bigdata/CLIP/binary_model_trial_21_new.pth'
results_dir = 'results_new'
class_names = ['Farmland', 'NotFarmland']
confidence_threshold = 0.5

model_cfg_path = 'configs/sam2.1/sam2.1_hiera_l.yaml'
checkpoint_path = 'sam2.1_hiera_large.pt'
device = 'cuda'

clip_model, preprocess = load_model(clip_model_path, device)
sam2 = build_sam2(model_cfg_path, checkpoint_path, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=16,
    points_per_batch=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    stability_score_offset=1.0,
    crop_n_layers=0,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=0,
    use_m2m=False
)

results_name = 'clip_classification_results_new_1'
results_path = f'{results_name}.csv'
if not os.path.exists(results_path):
    with open(results_path, 'w') as f:
        f.write('coverage,iou\n')

for image_path, geom_list in zip(image_dirs, geometry_list):
    coverage_val, iou_val = eval_image(
        image_path, geom_list, mask_generator,
        clip_model, preprocess,
        results_dir, class_names, confidence_threshold, device
    )
    with open(results_path, 'a') as f:
        f.write(f'{coverage_val},{iou_val}\n')