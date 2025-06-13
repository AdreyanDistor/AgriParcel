
import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
from tqdm import tqdm
from simplify_utils import evaluate_simplification

def run_simplification_analysis(gdf_path, output_csv, eps_range=(0.01, 10), samples=50, meters_per_pixel=4):
    gdf = gpd.read_file(gdf_path).to_crs(3857)
    gdf['geometry'] = gdf['geometry'].apply(shapely.make_valid)
    epsilons = np.linspace(*eps_range, samples)

    results = []
    for epsilon in tqdm(epsilons, desc='Simplification'):
        for _, row in gdf.iterrows():
            res = evaluate_simplification(row.geometry, epsilon, meters_per_pixel)
            if res:
                res['id'] = row.get('OBJECTID', None)
                results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
