import re
import math
import numpy as np
import osmnx as ox
import pandas as pd
import geopandas as gpd

from pyproj import Transformer
from shapely.geometry import Point
from osmnx._errors import InsufficientResponseError

def latlon_to_utm(lat, lon):
    """
    Convert latitude and longitude to UTM coordinates.
    Returns: (easting, northing, zone_number, zone_letter)
    """
    # Determine the UTM zone number
    zone_number = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'    # Create a transformer for lat/lon to UTM
    proj_str = f"+proj=utm +zone={zone_number} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs("epsg:4326", proj_str, always_xy=True)    
    easting, northing = transformer.transform(lon, lat)

    return easting, northing, zone_number, hemisphere


def GT_reader(seq):
    """
    Parse GT trajectory file and extract bounding box, UTM zone, and initial point.    
    Args:
    seq (str): Sequence ID to match (e.g., "00", "01", etc.)    
    Returns:
    tuple: containing max_lat, min_lat, max_lon, min_lon, zone_number, initial_point, and intital angle
    """    
    file_path = "./datasets/IRI_sequences_GT.txt"   
     # Read and parse the file
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()    
        
    # Find the specific sequence
    target_sequence = f"IRI_{seq}"
    
    # Split content into trajectory blocks
    blocks = re.split(r'type\s+latitude\s+longitude\s+name\s+desc', content)
    blocks = [block.strip() for block in blocks if block.strip()]    
    
    # Find the block containing our target sequence
    target_block = None
    for block in blocks:
        if target_sequence in block:
            target_block = block
            break   
    if not target_block:
        raise ValueError(f"Sequence {target_sequence}")   
    
    # Extract all coordinate lines from the target block
    lines = target_block.strip().split('\n')
    coordinates = []    
    
    for line in lines:
        line = line.strip()
        if line and line.startswith('T'):
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    lat = float(parts[1])
                    lon = float(parts[2])
                    coordinates.append((lat, lon))
                except ValueError:
                    continue    
    # Calculate bounding box
    lats = [coord[0] for coord in coordinates]
    lons = [coord[1] for coord in coordinates]    
    max_lat = max(lats) + 0.0005
    min_lat = min(lats) - 0.0005
    max_lon = max(lons) + 0.0005
    min_lon = min(lons) - 0.0005 

    # Get initial point (first coordinate) and convert to UTM
    initial_lat, initial_lon = coordinates[0]
    initial_x, initial_y, zone_number, _ = latlon_to_utm(initial_lat, initial_lon)    
    
    # Convert max lat and lon convert to UTM
    max_lat, max_lon, _, _ = latlon_to_utm(max_lat, max_lon)   

    # Convert min lat and lon convert to UTM
    min_lat, min_lon, _, _ = latlon_to_utm(min_lat, min_lon) 
    
    # Calculate initial angle from first two points
    second_lat, second_lon = coordinates[1]
    second_x, second_y, _, _ = latlon_to_utm(second_lat, second_lon)    
    
    # Calculate the direction vector
    dx = second_x - initial_x
    dy = second_y - initial_y
    
    # Calculate angle from North (0 = North, clockwise positive)
    # atan2(dx, dy) gives angle from North (y-axis)
    initial_angle_rad = math.atan2(dx, dy)   
    
    # Convert to degrees
    initial_angle = math.degrees(initial_angle_rad)    
    
    # Normalize to [0, 360] range
    if initial_angle < 0:
        initial_angle += 360    
    
    return max_lat, min_lat, max_lon, min_lon, zone_number, (initial_x, initial_y), initial_angle, coordinates[0]

def street_segmentation(initial_point,zone,area=750):
    '''
    Function that checks if the point is on the graph and the distance to the closest edge of the graph

    Parameters:
        initial_point: (1,2) array of the coordinates (latitude, longitude) of the initial point of the sequence
        zone: UTM zone of the region  e.g. ("+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs") for KITTI sequence in Germany
        area: The radius of the area around the initial point that information is extracted.
    Returns:
        edges: Graph containing the center of the streets of an area around the initial point.
        road_area: Geoseries containg the area of the streets dimensioned to the number of lanes, if this is available
        walkable_area_gdf: Geoseries containg the walkable area (buildings minus streets)
    '''

    G = ox.graph_from_point(initial_point, dist=area, network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    # Check how to pass this info to the function.
    edges = edges.to_crs(zone)

    # Extract lane info, coerce -> invalid parsing will be set as NaN
    edges['lanes'] = pd.to_numeric(edges['lanes'], errors='coerce')

    # Assumes that if info is NaN there is 1 lane
    edges['lanes'] = pd.to_numeric(edges['lanes'], errors='coerce').fillna(1)

    # Creates the road space as a function of the number of lanes
    edges['buffer_size'] = edges['lanes'] * 2.5  # Assuming each lane is ~2.5 meters wide

    road_buffer = edges.geometry.buffer(edges['buffer_size'])
    road_area = gpd.GeoSeries(road_buffer, crs=edges.crs)

    buildings = ox.features_from_point(initial_point, tags={"building": True}, dist=area)
    buildings = buildings.to_crs(zone)
    building_area = gpd.GeoSeries(buildings.unary_union,crs=edges.crs)

    convex_hull = buildings.unary_union.convex_hull
    total_area = gpd.GeoSeries([convex_hull], crs=buildings.crs)

    # Identify intersections (crossings) in the graph using OpenStreetMap's crossing tag
    crossings = ox.features_from_point(initial_point, tags={"highway": "crossing"}, dist=area)
    crossings = crossings.to_crs(zone)

    # Buffer around crossings to represent their area
    # Match crossings to the nearest road segment and assign buffer size based on road width
    crossings = gpd.sjoin(crossings, edges[['geometry', 'buffer_size']], how='left', predicate='intersects')
    crossings['buffer_size'] = crossings['buffer_size'].fillna(5)  # Default to 5 meters if no match is found
    crossings_buffer = crossings.geometry.buffer(crossings['buffer_size']*1.4)
    crossings_area = gpd.GeoSeries(crossings_buffer.unary_union, crs=edges.crs)
    
    # Extract railway areas
    try:
        railway = ox.features_from_point(initial_point, tags={"railway": "tram"}, dist=area)
        railway = railway.to_crs(zone)
        railway_buffer = railway.geometry.buffer(3.5) # Assuming a buffer size of 3.5 meters for the railway
        railway_area = gpd.GeoSeries(railway_buffer.unary_union, crs=edges.crs)
    except InsufficientResponseError:
        railway_area = gpd.GeoSeries()

    # Extract grass, parks and grassland areas
    green_geometries = []
    
    # Define all the tags we want to check
    tags_to_check = [
        {"landuse": "grass"},
        {"natural": "grassland"},
        {"leisure": "garden"},
        {"leisure": "park"}, 
    ]
    
    # Try each tag combination
    for tags in tags_to_check:
        try:
            features = ox.features_from_point(initial_point, tags=tags, dist=area)
            if not features.empty:
                features = features.to_crs(zone)
                green_geometries.append(features.unary_union)
        except InsufficientResponseError:
            continue
    
    # If we found any green areas, combine them
    if green_geometries:
        # Filter out None geometries (just in case)
        valid_geometries = [geom for geom in green_geometries if geom is not None]
        
        if valid_geometries:
            # Combine all geometries
            combined = gpd.GeoSeries(valid_geometries, crs=zone).unary_union
            
            # Convert to final CRS and return as GeoSeries
            green_area = gpd.GeoSeries([combined], crs=edges.crs)

    # Subtract the building footprints from the buffered street area to get walkable space
    walkable_area = total_area.difference(building_area)
    walkable_area = walkable_area.difference(railway_area)
    unified_road = gpd.GeoSeries(road_buffer.unary_union, crs=edges.crs)
    walkable_area = walkable_area.difference(unified_road)

    # Convert to GeoDataFrame for easy plotting
    walkable_area_gdf = gpd.GeoDataFrame(geometry=walkable_area, crs=edges.crs)

    return (edges,road_area,walkable_area_gdf,building_area,crossings_area,railway_area,green_area)

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    rotation_matrix = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
    ])
    return rotation_matrix


def rotation_matrix_z(theta):
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta),  0],
        [0, 0, 1]
    ])

    return rot_z

# Create 4x4 transformation matrix
def create_transformation_matrix(translation, rotation_matrix):
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = translation
    return transformation