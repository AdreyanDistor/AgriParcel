
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import make_valid

def jaccard_similarity(poly1, poly2):
    poly1, poly2 = make_valid(poly1), make_valid(poly2)
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0

def total_sides(geometry):
    if isinstance(geometry, MultiPolygon):
        return sum(len(p.exterior.coords) - 1 for p in geometry.geoms)
    if isinstance(geometry, Polygon):
        return len(geometry.exterior.coords) - 1
    return 0

def simplify_geometry(geometry, tolerance):
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon([g.simplify(tolerance) for g in geometry.geoms])
    return geometry.simplify(tolerance)

def reproject_pixel(gdf, width, height):
    minx, miny, maxx, maxy = gdf.total_bounds
    sx, sy = width / (maxx - minx), height / (maxy - miny)

    def scale(geom):
        if geom.geom_type == 'Polygon':
            coords = [((x - minx) * sx, (y - miny) * sy) for x, y in geom.exterior.coords]
            return Polygon(coords)
        if geom.geom_type == 'MultiPolygon':
            return MultiPolygon([scale(p) for p in geom.geoms])
        return geom

    gdf['geometry'] = gdf['geometry'].apply(scale)
    return gdf
