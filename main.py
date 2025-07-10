import sys
sys.path.insert(0, 'src/delineation')
sys.path.insert(0, 'src/clip')
sys.path.insert(0, 'src/sam2')
sys.path.insert(0, 'sam2')

import os
import argparse
import torch
import numpy as np
from PIL import Image
import rasterio
import geopandas as gpd
from torchvision import transforms
from model_utils import load_clip_model
from object_delineation import delineate
from run_sam2 import generate_masks
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import shapely
import json
import time

def calculate_acres(geometry, crs):
    if crs.is_projected:
        area_in_sqm = geometry.area
        acres = area_in_sqm * 0.000247105
    else:
        acres = 0
    return acres

def generate_farmland_data(
    input_directory,
    vector_data_dir,
    model_cfg_path,
    checkpoint_path,
    clip_model_path,
    output_path,
    unsimplified_path,
    mask_json_name,
    confidence_threshold=0.5,
    points_per_side=32,
    points_per_batch=64,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    stability_score_offset=1.0,
    crop_n_layers=0,
    crop_n_points_downscale_factor=1,
    box_nms_thresh=0.5,
    min_mask_region_area=0,
    use_m2m=False,
    dir_crs=4326
):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam2 = build_sam2(model_cfg_path, checkpoint_path, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        crop_n_layers=crop_n_layers,
        box_nms_thresh=box_nms_thresh,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        use_m2m=use_m2m
    )

    os.makedirs(vector_data_dir, exist_ok=True)
    generate_masks(input_directory, vector_data_dir, mask_generator, return_segmented=True)

    clip_model, preprocess = load_clip_model(clip_model_path, device)
    transform = transforms.Compose([preprocess])
    class_names = ['Farmland', 'NotFarmland']
    polygons, simplified_polygons = [], []
    confidences, accepted_ids, accepted_filenames = [], [], []
    clipped_mask_json = []
    curr_id, mask_id = 0, 0
    total_area_processed, total_masks_processed = 0, 0
    img_crs = None

    for mask_file in os.listdir(vector_data_dir):
        mask_path = os.path.join(vector_data_dir, mask_file)
        if not mask_file.lower().endswith(('.tif', '.tiff')):
            continue
        try:
            with Image.open(mask_path) as img:
                image_t = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feats = clip_model.encode_image(image_t)
                    logits = clip_model.logit_scale.exp() * feats @ clip_model.text_projection.t()
                    probs = logits.softmax(dim=-1).cpu().numpy()[0]

                pred_class = class_names[np.argmax(probs)]
                confidence = np.max(probs)
                print(f'Mask: {mask_file} | Prediction: {pred_class} ({confidence:.2f})')

                if pred_class != 'Farmland' or confidence < confidence_threshold:
                    print(f'{mask_file} is not Farmland (below threshold)')
                    continue

                with rasterio.open(mask_path) as geo:
                    affine_transform = geo.transform
                    img_crs = geo.crs

                if img.mode != 'RGB':
                    img = img.convert('RGBA')

                img_array = np.array(img)

                try:
                    bin_mask = (img_array[:, :, 3] > 0).astype(np.uint8)
                    polys = delineate(bin_mask, affine_transform)
                    if not polys:
                        print(f'No polygons generated from {mask_file}')
                        continue
                except Exception as e:
                    print(f'Error during delineation in {mask_file}: {e}')
                    continue

                try:
                    simplified = shapely.simplify(polys, tolerance=10)
                except Exception as e:
                    print(f'Error simplifying polygons in {mask_file}: {e}')
                    simplified = polys

                polygons.extend(polys)
                simplified_polygons.extend(simplified)

                for p in polys:
                    accepted_ids.append(curr_id)
                    accepted_filenames.append(mask_file)
                    confidences.append(confidence)
                    curr_id += 1

                try:
                    total_area_processed += sum(calculate_acres(p, img_crs) for p in polys)
                    total_masks_processed += len(polys)
                except Exception as e:
                    print(f'Error calculating area/perimeter in {mask_file}: {e}')

                clipped_mask_json.append({
                    'id': mask_id,
                    'filename': mask_file,
                    'confidence': float(confidence)
                })
                mask_id += 1

        except Exception as e:
            print(f'Unhandled error processing {mask_file}: {e}')

    if img_crs is None:
        img_crs = dir_crs

    for suffix, geoms in [('nonsimplified', polygons), ('simplified', simplified_polygons)]:
        try:
            if not geoms:
                print(f'No geometries for {suffix}, skipping write.')
                continue
            gdf = gpd.GeoDataFrame(geometry=geoms, crs=img_crs)
            gdf['confidence'] = confidences
            gdf['area_sqft'] = gdf.to_crs(epsg=2272).area
            gdf['perimeter_sqft'] = gdf.to_crs(epsg=2272).length
            gdf.insert(0, 'id', accepted_ids)
            gdf.insert(1, 'file_name', accepted_filenames)

            for column in gdf.select_dtypes(include=['float16']).columns:
                gdf[column] = gdf[column].astype('float32')

            out_name = output_path if suffix == 'simplified' else unsimplified_path
            gdf.to_file(f'{out_name}.geojson', driver='GeoJSON')
        except Exception as e:
            print(f'Failed to write {suffix} GeoJSON: {e}')

    try:
        with open(f'{mask_json_name}.json', 'w') as f:
            json.dump(clipped_mask_json, f, indent=4)
    except Exception as e:
        print(f'Failed to write mask metadata JSON: {e}')

    total_time = time.time() - start_time
    print('\n=== Pipeline Summary ===')
    print(f'Total area processed: {total_area_processed:.4f} acres')
    print(f'Total masks processed: {total_masks_processed}')
    print(f'Total time taken: {total_time:.2f} seconds')
    print(f'Acres/sec: {total_area_processed / total_time:.4f}')
    print('Pipeline complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Agri Parcel Vector Generation Pipeline')
    parser.add_argument('--input', type=str, default='data/test_data', help='Input directory with .tif images')
    parser.add_argument('--output', type=str, default='outputs/final_output', help='Path to output GeoJSON (simplified)')
    parser.add_argument('--unsimplified', type=str, default='outputs/final_output_nonsimplified', help='Path to unsimplified GeoJSON')
    parser.add_argument('--json', type=str, default='outputs/final_metadata', help='Path to save metadata JSON')
    parser.add_argument('--maskdir', type=str, default='outputs/farmland_masks', help='Folder to store generated masks')
    parser.add_argument('--clip_model', type=str, default='models/clip_model.pth')
    parser.add_argument('--sam_config', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--sam_ckpt', type=str, default='models/sam2.1_hiera_large.pt')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for CLIP classifier')
    parser.add_argument('--crs', type=int, default=3857, help='Coordinate System of Satellite Imagery')
    args = parser.parse_args()

    generate_farmland_data(
        input_directory=args.input,
        vector_data_dir=args.maskdir,
        model_cfg_path=args.sam_config,
        checkpoint_path=args.sam_ckpt,
        clip_model_path=args.clip_model,
        output_path=args.output,
        unsimplified_path=args.unsimplified,
        mask_json_name=args.json,
        confidence_threshold=args.conf,
        dir_crs=args.crs
    )
