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


# def boundary_correction(point, obstacles_geometry):
#     point = Point(point[0], point[1])
    
#     # Check if point is outside any geometry in the collection
#     is_inside = any(geom.contains(point) for geom in obstacles_geometry.geoms)
#     print(f"Point inside buildings: {is_inside}")
#     if is_inside:
#         # Find the nearest geometry in the collection
#         nearest_geom = min(obstacles_geometry.geoms, key=lambda geom: geom.distance(point))
        
#         # Get the nearest point on the nearest geometry
#         adjusted_point = nearest_points(nearest_geom, point)[0]
        
#         return (adjusted_point.x, adjusted_point.y)
    
#     # If not outside, return original point
#     return (point.x, point.y)

def boundary_correction(point, obstacles_geometry, crossings_area):
    """Correct a point to the nearest boundary of obstacles if it's inside any obstacle.
    
    Args:
        point: Tuple of (x, y) coordinates
        obstacles_geometry: Shapely geometry collection of obstacles
        crossing_areas: Shaeply geometry of the crossing areas
        
    Returns:
        Tuple of corrected (x, y) coordinates
    """
    point = Point(point[0], point[1])
    
    # Check if point is inside any obstacle
    containing_geoms = [geom for geom in obstacles_geometry.geoms if geom.contains(point)]
    
    is_inside_crossing = crossings_area.geometry.apply(lambda geom: geom.contains(point))

    if not containing_geoms or is_inside_crossing.any():
        return (point.x, point.y)  # Return original if not inside any obstacle or in crossing_area
    
    # For all containing geometries, find the nearest point on their boundary
    boundary_points = []
    for geom in containing_geoms:
        # Get the boundary of the geometry
        boundary = geom.boundary
        
        # Handle different boundary types (Polygon vs MultiPolygon)
        if boundary.geom_type == 'MultiLineString':
            for line in boundary.geoms:
                nearest = nearest_points(line, point)[0]
                boundary_points.append(nearest)
        else:
            nearest = nearest_points(boundary, point)[0]
            boundary_points.append(nearest)
    
    # Find the closest boundary point among all candidates
    if boundary_points:
        closest_point = min(boundary_points, key=lambda p: p.distance(point))
        return (closest_point.x, closest_point.y)
    
    return (point.x, point.y)  # Fallback (shouldn't normally happen)


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

    # if abs(correction) > 1:
    #     correction = 1 * np.sign(correction)
    # else: correction = correction


    if (-np.pi/2) < line_angle < 0:
        correction = np.array([correction,0]).T
    elif -np.pi < line_angle < (-np.pi/2):
        correction = np.array([0,correction]).T
    elif (-np.pi/2) < line_angle < (-np.pi * 3/4):
        correction = np.array([correction,0]).T
    else: correction = np.array([0,correction]).T

    rot_matrix = np.array([
        [np.cos(line_angle), np.sin(line_angle)],
        [-np.sin(line_angle), np.cos(line_angle)]
    ])
    rotated_point = rot_matrix @ correction
    corrected_point = (point[0] + rotated_point[0], point[1] + rotated_point[1])
    
    print(f"Corrected_point: {corrected_point}") # Debug
    return intersecting_objects, corrected_point

def multipoint_correction():

    return 0