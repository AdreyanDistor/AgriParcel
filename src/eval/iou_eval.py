
import geopandas as gpd
from shapely import make_valid
from tqdm import tqdm

def evaluate_iou(truth_gdf, pred_gdf, threshold=0.5):
    if truth_gdf.crs != pred_gdf.crs:
        truth_gdf = truth_gdf.to_crs(pred_gdf.crs)

    truth = [make_valid(g) for g in truth_gdf.geometry]
    pred = [make_valid(g) for g in pred_gdf.geometry]
    matched_preds = set()
    tp = 0

    for t in tqdm(truth, desc='IOU Matching'):
        for i, p in enumerate(pred):
            if i in matched_preds:
                continue
            inter = t.intersection(p).area
            union = t.union(p).area
            if union > 0 and inter / union >= threshold:
                tp += 1
                matched_preds.add(i)
                break

    fp = len(pred) - len(matched_preds)
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall
    }
