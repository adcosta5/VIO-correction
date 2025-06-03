import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.strtree import STRtree
from shapely.affinity import rotate
from shapely.ops import nearest_points, unary_union
from rtree import index
import random
import math


def boundary_correction():


    return 0

# Create spatial indices
def create_index(geoseries):
    idx = index.Index()
    for i, geom in enumerate(geoseries.geometry):
        idx.insert(i, geom.bounds)
    return idx

def point_correction(point, roads, buildings, railways, grass):

    # Convert points input to a Shaeply geometry
    point = Point(point)

    road_idx = create_index(roads)
    building_idx = create_index(buildings)
    railway_idx = create_index(railway)
    grass_idx = create_index(grass)

    

    return 0

def multipoint_correction():

    return 0

def calculate_distances_with_index(points, roads_gseries, buildings_gseries, railway_gseries):
    """
    Calculate perpendicular distances from points to nearest road/building features.
    
    Args:
        points: Can be either:
            - GeoDataFrame with point geometries
            - NumPy array of shape (n, 2) with (x,y) coordinates
            - List of shapely Point objects
        roads_gseries: GeoSeries containing road geometries
        buildings_gseries: GeoSeries containing building geometries
    
    Returns:
        Tuple of (results_df, nearest_points_gdf)
    """
    # Convert points input to consistent format
    if isinstance(points, np.ndarray):
        # Convert numpy array to GeoDataFrame
        geometry = [Point(xy) for xy in points]
        points_gdf = gpd.GeoDataFrame(geometry=geometry, crs=roads_gseries.crs)
    elif not hasattr(points, 'geometry'):  # Handle other non-geodataframe inputs
        points_gdf = gpd.GeoDataFrame(geometry=points, crs=roads_gseries.crs)
    else:
        points_gdf = points
    
    # Create spatial indices
    def create_index(geoseries):
        idx = index.Index()
        for i, geom in enumerate(geoseries.geometry):
            idx.insert(i, geom.bounds)
        return idx
    
    road_idx = create_index(roads_gseries)
    building_idx = create_index(buildings_gseries)
    railway_idx = create_index(railway_gseries)
    
    road_distances = []
    building_distances = []
    railway_distances = []
    nearest_road_points = []
    nearest_building_points = []
    nearest_railway_points = []
    
    for point in points_gdf.geometry:

        ## NEAREST ROAD POINTS USING SHAPELY DISTANCE
        #############################################################################################
        # Find nearest road
        road_candidates = list(road_idx.nearest(point.bounds, num_results=5))
        nearest_road = min(
            (nearest_points(point, roads_gseries.geometry.iloc[c])[1] for c in road_candidates),
            key=lambda p: point.distance(p)
        )
        road_distances.append(point.distance(nearest_road))
        nearest_road_points.append(nearest_road)

        ############################################################################################
        
        # Find nearest building
        building_candidates = list(building_idx.nearest(point.bounds, num_results=5))
        nearest_building = min(
            (nearest_points(point, buildings_gseries.geometry.iloc[c])[1] for c in building_candidates),
            key=lambda p: point.distance(p)
        )
        building_distances.append(point.distance(nearest_building))
        nearest_building_points.append(nearest_building)

        #Find nearest railway
        railway_candidates = list(railway_idx.nearest(point.bounds, num_results=5))
        nearest_railway = min(
            (nearest_points(point, railway_gseries.geometry.iloc[c])[1] for c in railway_candidates),
            key=lambda p: point.distance(p)
        )
        railway_distances.append(point.distance(nearest_railway))
        nearest_railway_points.append(nearest_railway)
    
    # Create results DataFrame
    results_df = points_gdf.copy()
    results_df['distance_to_road'] = road_distances
    results_df['distance_to_building'] = building_distances
    results_df['distance_to_railway'] = railway_distances
    
    nearest_data = {
        'geometry': nearest_road_points + nearest_building_points + nearest_railway_points,
        'type': ['road']*len(points_gdf) + ['building']*len(points_gdf) + ['railway']*len(points_gdf),
        'original_point_idx': list(range(len(points_gdf))) * 3
    }
    nearest_points_gdf = gpd.GeoDataFrame(nearest_data, crs=points_gdf.crs)
    
    return results_df, nearest_points_gdf