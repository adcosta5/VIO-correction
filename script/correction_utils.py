import geopandas as gpd
import numpy as np
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box
from shapely.ops import nearest_points, unary_union
import random

from fastSAM_utils import FastSAMutils


def merge_geoseries_obstacles(*geoseries_list):
    """Merge multiple GeoSeries into a single obstacles collection"""
    all_geometries = []
    
    for geoseries in geoseries_list:
        # Extract valid geometries from each GeoSeries
        valid_geoms = [geom for geom in geoseries.geometry if not geom.is_empty]
        all_geometries.extend(valid_geoms)
    
    # Combine all geometries
    return unary_union(all_geometries) if all_geometries else Polygon()

def boundary_correction(point, obstacles_geometry, crossings_area):
    """Correct a point to the nearest boundary of obstacles if it's inside any obstacle.
    
    Args:
        point: Tuple of (x, y) coordinates
        obstacles_geometry: Shapely geometry collection of obstacles or a single shaeply geometry
        crossing_areas: Shaeply geometry of the crossing areas
        
    Returns:
        Tuple of corrected (x, y) coordinates
    """
    point = Point(point[0], point[1])
    
    # Check if point is inside any obstacle
    if isinstance(obstacles_geometry, GeometryCollection):
        containing_geoms = [geom for geom in obstacles_geometry.geoms if geom.contains(point)]
    else:
        containing_geoms = obstacles_geometry.geometry[obstacles_geometry.geometry.contains(point)]
        containing_geoms = list(containing_geoms) if not containing_geoms.empty else []

    is_inside_crossing = crossings_area.geometry.apply(lambda geom: geom.contains(point)).any()

    if not containing_geoms or is_inside_crossing:
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

def point_correction(point, max_distance, obstacles_geometry, building_area, crossing_area, fov_box, line_angle,left_distance, right_distance, angles=(90,-90)):
    
    intersecting_objects = cartographic_point_extractor(point, max_distance, obstacles_geometry, fov_box, line_angle, angles)
    # print(f"Intersecting objects: {intersecting_objects}") # Debug

    if len(intersecting_objects) > 1:
        dist_left = np.linalg.norm(np.array(intersecting_objects[0]) - np.array(point)) # Euclidean distance is the L2 norm
        dist_right = np.linalg.norm(np.array(intersecting_objects[1]) - np.array(point))

        # print(f"Dist right: {dist_right}") # Debug
        # print(f"ZED right: {right_distance}\n")
        # print(f"Dist left: {dist_left}")
        # print(f"ZED left: {left_distance}\n")

        # If the distance from the camera is None correction is 0.
        # Return positive when point is to far away from the edge, and the other way around. (right_distance is negative (coordinate system))
        left_correction =  (left_distance or dist_left) - dist_left      
        right_correction =  - (dist_right + (right_distance or dist_right))

        # print(f"Correction: {right_correction}, {left_correction}\n") # Debug
        # print(f"Line angle: {line_angle}")

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

    if abs(correction) > 1.5:
        correction = 1.5 * np.sign(correction)
    else: correction = correction


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
    
    # print(f"Corrected_point: {corrected_point}") # Debug

    corrected_point = boundary_correction(corrected_point, building_area, crossing_area)

    return intersecting_objects, corrected_point


def point_cloud_rotation(real_point, line_angle, mask, point_cloud, contours):
    
    rotated_point_cloud = []
    for contour in contours:
        for point in contour:
            x, y = int(point[0,0]), int(point[0,1])
            point3D = point_cloud.get_value(x,y)
            if (-np.pi/2) < line_angle < 0:
                pcp = np.array([point3D[1][1],point3D[1][0]]).T
            elif -np.pi < line_angle < (-np.pi/2):
                pcp = np.array([point3D[1][0],point3D[1][1]]).T
            elif (-np.pi/2) < line_angle < (-np.pi * 3/4):
                pcp = np.array([point3D[1][1],point3D[1][0]]).T
            else: pcp = np.array([point3D[1][0],point3D[1][1]]).T

            rot_matrix = np.array([
                [np.cos(line_angle), -np.sin(line_angle)],
                [np.sin(line_angle), np.cos(line_angle)]
            ])
            rotated_point = rot_matrix @ pcp
            transformed_point = (real_point[0] + rotated_point[0], real_point[1] + rotated_point[1])
            rotated_point_cloud.append(transformed_point)
    print(f"Transformed points: \n {rotated_point_cloud}")
    return rotated_point_cloud

from scipy.spatial import KDTree
from scipy.optimize import minimize

def robust_icp_2d(source, target, max_iterations=100, tolerance=1e-8):
    """
    Enhanced ICP with:
    - Point cloud normalization
    - Better initialization
    - Outlier rejection
    """
    # 1. Normalize point clouds to [0,1] range
    src, src_scale, src_offset = normalize_points(source)
    tgt, tgt_scale, tgt_offset = normalize_points(target)
    
    # 2. Initialize transformation
    params = np.zeros(3)  # [tx, ty, theta]
    errors = []
    
    for i in range(max_iterations):
        # 3. Find closest points (with outlier rejection)
        kdtree = KDTree(tgt)
        distances, indices = kdtree.query(src)
        
        # Reject outliers (top 10% farthest points)
        threshold = np.percentile(distances, 90)
        valid = distances < threshold
        corresponding = tgt[indices[valid]]
        filtered_src = src[valid]
        
        # 4. Compute error
        mean_error = np.mean(distances[valid])
        errors.append(mean_error)
        
        # Check convergence
        if i > 0 and abs(errors[-1] - errors[-2]) < tolerance:
            break
            
        # 5. Optimize transformation
        def objective(p):
            tx, ty, theta = p
            R = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
            transformed = (R @ filtered_src.T).T + np.array([tx, ty])
            return np.mean(np.linalg.norm(transformed - corresponding, axis=1))
            
        res = minimize(objective, params, method='L-BFGS-B')
        params = res.x
        
        # 6. Apply current transformation to full source
        tx, ty, theta = params
        R = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
        src = (R @ src.T).T + np.array([tx, ty])
    
    # 7. Denormalize results
    final_src = denormalize_points(src, src_scale, src_offset)
    final_tx = params[0] * tgt_scale[0] + tgt_offset[0] - src_offset[0]
    final_ty = params[1] * tgt_scale[1] + tgt_offset[1] - src_offset[1]
    
    return final_src, (final_tx, final_ty, params[2]), errors

def normalize_points(points):
    """Normalize points to [0,1] range"""
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    scale = maxs - mins
    scale[scale == 0] = 1  # Avoid division by zero
    return (points - mins) / scale, scale, mins

def denormalize_points(points, scale, offset):
    """Convert normalized points back to original scale"""
    return points * scale + offset


def multipoint_correction(point, max_distance, obstacles_geometry, building_area, crossing_area, fov_box, line_angle, mask, point_cloud, contours, angles=list(range(-90, 90, 5))):

    import matplotlib.pyplot as plt
    
    intersecting_objects = cartographic_point_extractor(point, max_distance, obstacles_geometry, fov_box, line_angle, angles)
    rotated_point_cloud = point_cloud_rotation(point, line_angle, mask, point_cloud, contours)
    
    # Run ICP
    aligned, (tx, ty, theta), errors = robust_icp_2d(np.array(rotated_point_cloud), intersecting_objects)

    intersecting_objects = np.array(intersecting_objects)
    rotated_point_cloud = np.array(rotated_point_cloud)


    # Plot results
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(intersecting_objects[:,0], intersecting_objects[:,1], c='blue', label='OSM (target)')
    plt.scatter(rotated_point_cloud[:,0], rotated_point_cloud[:,1], c='red', label='ZED (source)')
    plt.title('Before ICP')
    plt.legend()
    
    plt.subplot(122)
    plt.scatter(intersecting_objects[:,0], intersecting_objects[:,1], c='blue', label='OSM (target)')
    plt.scatter(aligned[:,0], aligned[:,1], c='green', label='ZED (aligned)')
    plt.title('After ICP')
    plt.legend()
    
    plt.figure()
    plt.plot(errors)
    plt.title('ICP Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Error')
    plt.show()
    
    print(f"Final transformation: translation=({tx:.3f}, {ty:.3f}), rotation={theta:.3f} radians")
    
    return intersecting_objects, rotated_point_cloud