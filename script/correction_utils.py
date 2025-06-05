import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, unary_union
import random


def merge_geoseries_obstacles(*geoseries_list):
    """Merge multiple GeoSeries into a single obstacles collection"""
    all_geometries = []
    
    for geoseries in geoseries_list:
        # Extract valid geometries from each GeoSeries
        valid_geoms = [geom for geom in geoseries.geometry if not geom.is_empty]
        all_geometries.extend(valid_geoms)
    
    # Combine all geometries
    return unary_union(all_geometries) if all_geometries else Polygon()


def boundary_correction(point, obstacles_geometry):
    
    point = Point(point[0],point[1])
    
    from shapely.geometry import Point, GeometryCollection
from shapely.ops import nearest_points

def boundary_correction(point, obstacles_geometry):
    point = Point(point[0], point[1])
    
    # Check if point is outside any geometry in the collection
    is_outside = any(geom.contains(point) for geom in obstacles_geometry.geoms)
    
    if is_outside:
        # Find the nearest geometry in the collection
        nearest_geom = min(obstacles_geometry.geoms, key=lambda geom: geom.distance(point))
        
        # Get the nearest point on the nearest geometry
        adjusted_point = nearest_points(nearest_geom, point)[0]
        
        return (adjusted_point.x, adjusted_point.y)
    
    # If not outside, return original point
    return (point.x, point.y)


def cartographic_point_extractor(point, max_distance, obstacles_geometry, fov_box, line_angle, angles):
    """Extract the closest points on the obstacles inside the fov_box with the specified angles."""
    results = []
    point = Point(point[0], point[1])
    x, y = point.x, point.y


    for angle in angles:
        # Adjust the angle relative to the fov_box orientation
        adjusted_angle = np.radians(angle) + line_angle
        # Create line using trigonometry
        end_x = x + max_distance * np.cos(adjusted_angle)
        end_y = y + max_distance * np.sin(adjusted_angle)
        line = LineString([(x, y), (end_x, end_y)])
        
        # Find intersections with both obstacles and box boundary
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

def point_correction(point, max_distance, obstacles_geometry, fov_box, line_angle,left_distance, right_distance, angles=(90,-90)):
    
    intersecting_objects = cartographic_point_extractor(point, max_distance, obstacles_geometry, fov_box, line_angle, angles)
    # print(f"Intersecting objects: {intersecting_objects}") # Debug

    if len(intersecting_objects) > 1:
        dist_left = np.linalg.norm(np.array(intersecting_objects[0]) - np.array(point)) # Euclidean distance is the L2 norm
        dist_right = np.linalg.norm(np.array(intersecting_objects[1]) - np.array(point))

        print(f"Dist right: {dist_right}") # Debug
        print(f"ZED right: {right_distance}\n")
        print(f"Dist left: {dist_left}")
        print(f"ZED left: {left_distance}\n")

        # If the distance from the camera is None correction is 0.
        # Return positive when point is to far away from the edge, and the other way around. (right_distance is negative (coordinate system))
        left_correction =  (left_distance or dist_left) - dist_left      
        right_correction =  - (dist_right + (right_distance or dist_right))

        print(f"Correction: {right_correction}, {left_correction}\n") # Debug
        print(f"Line angle: {line_angle}")

        correction = min(left_correction, right_correction)

    elif len(intersecting_objects) == 1:
        dist = np.linalg.norm(np.array(intersecting_objects) - np.array(point))
        min_dist_ZED = min(
            d for d in [left_distance, right_distance] if d is not None
            ) if any(d is not None for d in [left_distance, right_distance]) else None
        
        if min_dist_ZED == left_distance:
            correction = (min_dist_ZED or dist) - dist

        elif min_dist_ZED == right_distance:
            correction = - (dist + (min_dist_ZED or dist))

        else:
            correction = 0
        

    else: correction = 0


    if (-np.pi/2) < line_angle < 0:
        line_angle = np.pi/2 - line_angle
    elif -np.pi < line_angle < (-np.pi/2):
        line_angle = np.pi - line_angle
    elif (-np.pi/2) < line_angle < (-np.pi * 3/4):
        line_angle = -3/4 * np.pi - line_angle
    else: line_angle = line_angle

    rot_matrix = np.array([
        [np.cos(line_angle), np.sin(line_angle)],
        [-np.sin(line_angle), np.cos(line_angle)]
    ])
    rotated_point = rot_matrix @ np.array([0,correction]).T
    corrected_point = (point[0] + rotated_point[0], point[1] + rotated_point[1])
    
    print(f"Corrected_point: {corrected_point}") # Debug
    return intersecting_objects, corrected_point

def multipoint_correction():

    return 0