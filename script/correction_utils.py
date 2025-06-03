import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.strtree import STRtree
from shapely.affinity import rotate
from shapely.ops import nearest_points, unary_union
from rtree import index
import random
import math


def merge_geoseries_obstacles(*geoseries_list):
    """Merge multiple GeoSeries into a single obstacles collection"""
    all_geometries = []
    
    for geoseries in geoseries_list:
        # Extract valid geometries from each GeoSeries
        valid_geoms = [geom for geom in geoseries.geometry if not geom.is_empty]
        all_geometries.extend(valid_geoms)
    
    # Combine all geometries
    return unary_union(all_geometries) if all_geometries else Polygon()


def boundary_correction():


    return 0



def cartographic_point_extractor(point, max_distance, obstacles_geometry, fov_box, angles):

    results = []
    point = Point(point[0], point[1])
    x,y = point.x, point.y
    
    for angle in angles:
        # Create line using trigonometry
        end_x = x + max_distance * math.cos(math.radians(angle))
        end_y = y + max_distance * math.sin(math.radians(angle))
        line = LineString([(x, y), (end_x, end_y)])
        
        # Find intersections with both obstacles and box fov_boxboundary
        intersection = line.intersection(obstacles_geometry)
        # Check if intersection is inside the fov_box
        if intersection.is_empty:
            continue
        if intersection.geom_type == 'Point':
            if not fov_box.contains(intersection):
                continue
        else:
            # For MultiPoint, LineString, etc., check if any part is inside the box
            if not intersection.intersects(fov_box):
                continue

        # Find closest intersection point
        if intersection.geom_type == 'Point':
            closest = (intersection.x, intersection.y)
        else:
            # Extract all points from intersection
            points = []
            if hasattr(intersection, 'geoms'):
                for geom in intersection.geoms:
                    if geom.geom_type == 'Point':
                        points.append((geom.x, geom.y))
                    else:
                        points.extend(geom.coords)
            else:
                points.extend(intersection.coords)
            
            closest = min(points, key=lambda p: Point(p).distance(point))
    
        results.append(LineString([(x, y), closest]))
        results[-1] = closest

    return results

def point_correction(point, max_distance, obstacles_geometry, fov_box, angles=(90,-90)):
    
    intersecting_objects = cartographic_point_extractor(point, max_distance, obstacles_geometry, fov_box, angles)
    print(f"Intersecting objects: {intersecting_objects}")



    return intersecting_objects

def multipoint_correction():

    return 0