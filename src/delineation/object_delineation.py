from PIL import Image
import numpy as np
from shapely.geometry import LinearRing, Polygon
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import pprint

class vertex_pointer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.next = None
        self.visited = False

def grid_to_world(c, r, transform = None):
    if transform is None or transform == 1:
        transform = 1
    lon, lat = transform * (c, r)
    return lon, lat

def orthogonal_line_detection(mask: np.ndarray) -> list[vertex_pointer]:
    h, w = mask.shape
    top_vertices = [None] * (w + 1)
    left_vertex = None
    corners = []
    mask_padded = np.pad(mask, ((1, 1)), 'constant', constant_values=0)
    for y in range(1, h + 1):
        left_vertex = None
        for x in range(1, w + 1):
            curr_block = [mask_padded[y - 1, x - 1], mask_padded[y - 1, x], mask_padded[y, x - 1], mask_padded[y, x]]
            pixel_type = curr_block[0] * 1 + curr_block[1] * 2 + curr_block[2] * 4 + curr_block[3] * 8
            new_vertex = vertex_pointer(x - 1, y - 1)
            if pixel_type == 1:
                if left_vertex is not None:
                    left_vertex.next = new_vertex
                new_vertex.next = top_vertices[x]
            elif pixel_type == 2:
                if top_vertices[x] is not None:
                    top_vertices[x].next = new_vertex
                left_vertex = new_vertex
            elif pixel_type == 4:
                new_vertex.next = left_vertex
                top_vertices[x] = new_vertex
            elif pixel_type == 6:
                v1 = vertex_pointer(x - 1, y - 1)
                if top_vertices[x]:
                    top_vertices[x].next = v1
                new_vertex.next = left_vertex
                top_vertices[x] = left_vertex
                left_vertex = v1
            elif pixel_type == 7:
                left_vertex = new_vertex
                top_vertices[x] = left_vertex
                corners.append(left_vertex)
            elif pixel_type == 8:
                left_vertex = new_vertex
                top_vertices[x] = left_vertex
                corners.append(left_vertex)
            elif pixel_type == 9:
                left_vertex.next = new_vertex
                new_vertex.next = top_vertices[x]
                left_vertex = vertex_pointer(x - 1, y - 1)
                top_vertices[x] = left_vertex
                corners.append(top_vertices[x])
            elif pixel_type == 11:
                top_vertices[x] = left_vertex.next = new_vertex
            elif pixel_type == 13:
                left_vertex = new_vertex
                new_vertex.next = top_vertices[x]
            elif pixel_type == 14:
                new_vertex.next = left_vertex
                top_vertices[x].next = new_vertex
    return corners

def ring_formation_polygon(corners: list[vertex_pointer], transform = 1) -> list[Polygon]:
    polygons = []
    ccw = []
    cw = []
    i = 0
    for corner in corners:
        i = i + 1
        if corner is not None and not corner.visited:
            coords = []
            p = corner
            
            while p is not None:
                lon, lat = grid_to_world(p.x, p.y, transform)
                coords.append((lon, lat))
                p.visited = True
                p = p.next
                if p == corner:
                    break
            if len(coords) > 2:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                ring = LinearRing(coords)
                if ring.is_ccw:
                    ccw.append(ring)
                else:
                    cw.append(ring)
    for exterior in ccw:
        holes = []
        for interior in cw[:]:
            if exterior.contains(interior):
                holes.append(interior)
                cw.remove(interior)
        polygons.append(Polygon(exterior, holes))
    return polygons

def delineate(mask: np.ndarray, transform = 1) -> list[Polygon]:
    corners = orthogonal_line_detection(mask)
    
    for corner in corners:
        if corner.next == None:
            print(f'{corner} this pointers next is None')
    
    polygons = ring_formation_polygon(corners, transform)
    return polygons

def test1():
    mask = np.zeros((20, 20), dtype=int)
    mask[2:8, 3:10] = 1
    mask[14:16, 4:7] = 0 
    mask[5:15, 12:18] = 1
    mask[8:12, 14:16] = 0 
    mask[10, 10] = 1
    mask[1, 15] = 1
    mask[2, 16] = 1
    mask[3, 17] = 1
    polygons = delineate(mask=mask)
    gdf = gpd.GeoDataFrame(geometry=polygons)
    gdf.crs = 'EPSG:4326'
    gdf.to_file('w.geojson', driver='GeoJSON')

def binarize(img_path):
    return np.array(Image.open(img_path).convert('1'))