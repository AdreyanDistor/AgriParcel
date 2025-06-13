
import numpy as np
import rasterio
from rasterio.transform import from_origin
from skimage.measure import label, regionprops
import os

def get_bbox(mask):
    labeled = label(mask)
    props = regionprops(labeled)
    return props[0].bbox if props else None

def save_mask(mask, image_rgb, transform, profile, crs, image_name, idx, output_dir, return_segmented=True, name='mask'):
    mask_array = np.array(mask['segmentation'], dtype=np.uint8)
    bbox = get_bbox(mask_array)
    if bbox is None:
        print(f'No object in mask {idx} from {image_name}, skipping.')
        return

    min_r, max_r, min_c, max_c = bbox
    min_r, max_r = max(min_r - 1, 0), min(max_r + 1, mask_array.shape[0])
    min_c, max_c = max(min_c - 1, 0), min(max_c + 1, mask_array.shape[1])
    cropped_mask = mask_array[min_r:max_r, min_c:max_c]

    if return_segmented:
        cropped_img = image_rgb[min_r:max_r, min_c:max_c]
        data = (cropped_img * cropped_mask[..., np.newaxis]).transpose(2, 0, 1)
        count = 3
    else:
        data = (cropped_mask[np.newaxis, ...] * 255).astype(np.uint8)
        count = 1

    west, north = transform * (min_c, min_r)
    new_transform = from_origin(west, north, transform.a, -transform.e)
    profile.update(height=data.shape[1], width=data.shape[2], count=count, transform=new_transform, dtype='uint8', crs=crs)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{image_name}_{name}_{idx}.tif')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data)
    print(f'Saved mask {idx} to {out_path}')
