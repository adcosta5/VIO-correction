import geopandas as gpd
import numpy as np
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box
from shapely.ops import nearest_points, unary_union
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from scipy.optimize import minimize
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt

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
        #correction = 1.5 * np.sign(correction)
        return intersecting_objects, point
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
            pcp = np.array([point3D[1][0],point3D[1][1]]).T

            rot_matrix = np.array([
                [np.cos(line_angle), -np.sin(line_angle)],
                [np.sin(line_angle), np.cos(line_angle)]
            ])
            rotated_point = rot_matrix @ pcp
            transformed_point = (real_point[0] + rotated_point[0], real_point[1] + rotated_point[1])
            rotated_point_cloud.append(transformed_point)

    return rotated_point_cloud

def line_identificator(point_cloud, point, dist=10, n_samples=50):
    if point_cloud.ndim == 1:
        point_cloud = point_cloud.reshape(-1, 3)

    # Add check for minimum number of points
    if len(point_cloud) < 2:
        print("Not enough points to fit a line (need at least 2 points)")
        return None, None

    # Fit the first line
    ransac1 = RANSACRegressor(min_samples=2, residual_threshold=0.03, max_trials=1000)
    ransac1.fit(point_cloud[:, 0].reshape(-1, 1), point_cloud[:, 1])
    
    # Get inliers and outliers
    inlier_mask1 = ransac1.inlier_mask_
    line1_inliers = point_cloud[inlier_mask1]
    outliers = point_cloud[~inlier_mask1]

    x_min = point[0]
    x_max = x_min + 10
    x_samples = np.linspace(x_min, x_max, n_samples)
    y_samples = ransac1.predict(x_samples.reshape(-1, 1))

    line1 = np.column_stack((x_samples, y_samples))

    # Line 1 equation (y = m1*x + b1)
    m1 = ransac1.estimator_.coef_[0]
    b1 = ransac1.estimator_.intercept_
    # print(f"Line 1: y = {m1:.4f}x + {b1:.4f} | Inliers: {len(line1_inliers)}")

    line2 = None
    m2 = 0
    b2 = 0

    if len(outliers) >= 2:
        ransac2 = RANSACRegressor(min_samples=2, residual_threshold=0.03, max_trials=1000)
        ransac2.fit(outliers[:, 0].reshape(-1, 1), outliers[:, 1])
        
        inlier_mask2 = ransac2.inlier_mask_
        line2_inliers = outliers[inlier_mask2]

        x_samples = np.linspace(x_min, x_max, n_samples)
        y_samples = ransac2.predict(x_samples.reshape(-1, 1))
        
        line2_candidate = np.column_stack((x_samples, y_samples))
        # Line 2 equation (y = m2*x + b2)
        m2_candidate = ransac2.estimator_.coef_[0]
        b2_candidate = ransac2.estimator_.intercept_

        # Calculate the angle between the two lines in degrees in order to obtain the most parallel lines possible
        angle = np.abs(np.arctan((m2_candidate - m1) / (1 + m1 * m2_candidate))) * 180 / np.pi
        # print(f"Angle: {angle}") # Debug
        if angle <= 2:  # Only keep line2 if angle is <= 2 degrees
            line2 = line2_candidate
            m2 = m2_candidate
            b2 = b2_candidate
            # print(f"Line 2: y = {m2:.4f}x + {b2:.4f} | Inliers: {len(line2_inliers)}")
        else:
            print(f"Angle between lines ({angle:.2f}°) > 1°. Discarding line2.")
    else:
        print("Not enough outliers to fit a second line.")

    # # # DEBUG OF THE LINE EXTRACTOR BY PLOTTING
    # plt.scatter(point_cloud[:, 0], point_cloud[:, 1], label="All Points", color='gray', alpha=0.5)
    # plt.scatter(line1_inliers[:, 0], line1_inliers[:, 1], color='red', label="Line 1 Inliers")
    # if m2 != 0:  # Only plot line2 if it was kept
    #     plt.scatter(line2_inliers[:, 0], line2_inliers[:, 1], color='blue', label="Line 2 Inliers")

    # # Plot the fitted lines
    # x_vals = np.array([point_cloud[:, 0].min(), point_cloud[:, 0].max()])
    # plt.plot(x_vals, m1 * x_vals + b1, 'r-', label=f"Line 1: y = {m1:.2f}x + {b1:.2f}")
    # if m2 != 0:  # Only plot line2 if it was kept
    #     plt.plot(x_vals, m2 * x_vals + b2, 'b-', label=f"Line 2: y = {m2:.2f}x + {b2:.2f}")

    # plt.xlabel("X Coordinate")
    # plt.ylabel("Y Coordinate")
    # plt.legend()
    # plt.show()

    return line1, line2

def icp_2d(source, target, max_iterations=500, tolerance=1e-8):
    """
    2D Iterative Closest Point (ICP) algorithm.
    
    Parameters:
    - source: numpy array (Nx2), source point cloud to align to target
    - target: numpy array (Mx2), target point cloud
    - max_iterations: int, maximum number of iterations
    - tolerance: float, convergence tolerance
    
    Returns:
    - aligned_source: numpy array (Nx2), source points after alignment
    - transformation: tuple (R, t), rotation matrix and translation vector
    - distances: list, mean distances at each iteration
    """
    # Make copies to avoid modifying original arrays
    src = np.copy(source)
    dst = np.copy(target)
    
    # Initialize transformation
    R = np.eye(2)  # Identity matrix (no rotation)
    t = np.zeros(2)  # Zero translation
    
    prev_error = 0
    distances = []
    
    for i in range(max_iterations):
        # Find nearest neighbors between source and target
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst)
        distances, indices = nbrs.kneighbors(src)
        
        # Compute current error
        mean_error = np.mean(distances)
        distances = np.append(distances,mean_error)
        
        # Check for convergence
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
        
        # Get corresponding points from target
        correspondences = dst[indices.ravel()]
        
        # Compute centroids
        src_centroid = np.mean(src, axis=0)
        corr_centroid = np.mean(correspondences, axis=0)
        
        # Center the points
        src_centered = src - src_centroid
        corr_centered = correspondences - corr_centroid
        
        # Compute covariance matrix
        H = src_centered.T @ corr_centered
        
        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = corr_centroid - R @ src_centroid
        
        # Apply transformation
        src = (R @ src.T).T + t
    
    # Final transformation
    transformation = (R, t)
    aligned_source = src
    
    return aligned_source, transformation, distances

def multipoint_correction(point, max_distance, obstacles_geometry, crossing_area, fov_box, line_angle, mask, point_cloud, contours, angles=list(range(-90, 90, 5))):

    intersecting_objects = cartographic_point_extractor(point, max_distance, obstacles_geometry, fov_box, line_angle, angles)
    intersecting_objects = np.array(intersecting_objects)

    OSM_line1, OSM_line2 = line_identificator(intersecting_objects, point)
    if OSM_line1 is None:  # If no lines could be fitted
        return 0, 0, point
    
    if OSM_line2 is not None:
        OSM_line = np.append(OSM_line1, OSM_line2, axis=0)
    else:
        OSM_line = OSM_line1

    rotated_point_cloud = point_cloud_rotation(point, line_angle, mask, point_cloud, contours)
    rotated_point_cloud = np.array(rotated_point_cloud)

    if len(rotated_point_cloud) == 0:
        return OSM_line, 0, point
    
    ZED_line1, ZED_line2 = line_identificator(rotated_point_cloud, point)

    if ZED_line1 is None: # If no lines could be fitted
        return OSM_line, 0, point

    if ZED_line2 is not None:
        ZED_line = np.append(ZED_line1, ZED_line2, axis=0)
        aligned, (R, t), errors= icp_2d(ZED_line, OSM_line)
    else: 
        ZED_line = ZED_line1
        aligned, (R, t), errors= icp_2d(ZED_line1, OSM_line1)

    # # Plot results for Debug
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.scatter(ZED_line[:,0], ZED_line[:,1], c='red', label='ZED (target)')
    # plt.scatter(OSM_line[:,0], OSM_line[:,1], c='blue', label='OSM (source)')
    # plt.title('Before Alignment')
    # plt.legend()
    
    # plt.subplot(122)
    # plt.scatter(ZED_line[:,0], ZED_line[:,1], c='red', label='ZED (target)')
    # plt.scatter(OSM_line[:,0], OSM_line[:,1], c='blue', label='OSM (source)')
    # plt.scatter(aligned[:,0], aligned[:,1], c='green', label='OSM (aligned)')
    # plt.title('After Alignment')
    # plt.legend()
    
    # plt.figure()
    # plt.plot(errors)
    # plt.title('ICP Convergence')
    # plt.xlabel('Iteration')
    # plt.ylabel('Mean Error')
    # plt.show()
    
    # print("Estimated rotation:\n", R) # Debug
    # print("Estimated translation:", t)
    

    corr_point = np.array([point[0], point[1]])
    corrected_point = (R @ corr_point.T).T + t

    corrected_point = boundary_correction(corrected_point, obstacles_geometry, crossing_area)

    return OSM_line, ZED_line, corrected_point