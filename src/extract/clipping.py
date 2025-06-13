
import os
import geopandas as gpd
import rasterio
import rasterio.mask
from utils.io import ensure_dir
import pprint

p = pprint.PrettyPrinter(indent=2)

def clip_raster_by_vector(raster_path, vector_path, output_dir):
    ensure_dir(output_dir)

    with rasterio.open(raster_path) as src:
        gdf = gpd.read_file(vector_path)
        if src.crs != gdf.crs:
            gdf = gdf.to_crs(src.crs)

        for i, row in enumerate(gdf.itertuples()):
            geom = row.geometry
            p.pprint(geom)
            try:
                out_image, transform = rasterio.mask.mask(src, [geom], crop=True)
                out_image[out_image == src.nodata] = 0
                meta = src.meta.copy()
                meta.update({
                    'driver': 'GTiff',
                    'height': out_image.shape[1],
                    'width': out_image.shape[2],
                    'transform': transform
                })

                out_path = os.path.join(output_dir, f'data_{i}.tif')
                with rasterio.open(out_path, 'w', **meta) as dst:
                    dst.write(out_image)
                print(f'Clipped image saved to {out_path}')
            except ValueError as e:
                print(f'Skipping geometry {i}: {e}')
