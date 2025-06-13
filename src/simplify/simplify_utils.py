
from utils.geometry import jaccard_similarity, total_sides
from shapely.geometry import Polygon, MultiPolygon
from shapely import make_valid

def evaluate_simplification(original, tolerance, meters_per_pixel=4):
    original = make_valid(original)
    if not isinstance(original, (Polygon, MultiPolygon)):
        return None

    simplified = original.simplify(tolerance)
    if simplified.is_empty or not simplified.is_valid:
        return None

    iou = jaccard_similarity(original, simplified)
    og_edges = total_sides(original)
    simp_edges = total_sides(simplified)
    edge_loss = og_edges - simp_edges

    return {
        'iou': iou,
        'original_edges': og_edges,
        'simplified_edges': simp_edges,
        'edge_loss': edge_loss,
        'epsilon_m': tolerance,
        'epsilon_px': tolerance / meters_per_pixel
    }
