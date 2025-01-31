import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from itertools import combinations
from functools import lru_cache
import requests
from joblib import Parallel, delayed
import numpy as np
from IPython.display import IFrame
from sklearn.cluster import DBSCAN
import folium



map_center = [52.37, 4.89]
# Define central depot coordinates
DEPOT_LAT, DEPOT_LON = 52.333847, 4.865261  # Example coordinates
central_depot = (DEPOT_LAT, DEPOT_LON)

# OSRM Distance Function
@lru_cache(maxsize=None)
def get_osrm_distance(coord1, coord2, profile='driving'):
    """
    Calculate the distance between two coordinates using OSRM.
    """
    base_url = "http://router.project-osrm.org/route/v1"
    coords = f"{coord1[1]},{coord1[0]};{coord2[1]},{coord2[0]}"
    url = f"{base_url}/{profile}/{coords}"

    try:
        response = requests.get(url, params={"overview": "false"})
        response.raise_for_status()
        data = response.json()
        distance = data['routes'][0]['distance'] / 1000  # Distance in km
        return distance
    except requests.exceptions.RequestException as e:
        print(f"Error fetching OSRM data: {e}")
        return None

# Function to calculate bounding boxes
def calculate_bounding_boxes(data):
    bounding_boxes = {}
    for company in data['name'].unique():
        company_data = data[data['name'] == company]
        min_lat = company_data['lat'].min()
        max_lat = company_data['lat'].max()
        min_lon = company_data['lon'].min()
        max_lon = company_data['lon'].max()
        bounding_boxes[company] = {
            'min_lat': min_lat,
            'max_lat': max_lat,
            'min_lon': min_lon,
            'max_lon': max_lon
        }
    return bounding_boxes

# Function to calculate overlap area using shapely
def calculate_overlap_area(box1, box2):
    """
    Use shapely to calculate the overlap area between two bounding boxes.
    """
    polygon1 = Polygon([
        (box1['min_lon'], box1['min_lat']),
        (box1['max_lon'], box1['min_lat']),
        (box1['max_lon'], box1['max_lat']),
        (box1['min_lon'], box1['max_lat'])
    ])
    polygon2 = Polygon([
        (box2['min_lon'], box2['min_lat']),
        (box2['max_lon'], box2['min_lat']),
        (box2['max_lon'], box2['max_lat']),
        (box2['min_lon'], box2['max_lat'])
    ])
    if polygon1.intersects(polygon2):
        return polygon1.intersection(polygon2).area
    return 0

# Function to calculate overlap area using OSRM distances
def calculate_overlap_area_osrm(box1, box2):
    """
    Use OSRM distances to refine bounding box overlap.
    """
    # Calculate the corners of the overlapping bounding box
    overlap_min_lat = max(box1['min_lat'], box2['min_lat'])
    overlap_max_lat = min(box1['max_lat'], box2['max_lat'])
    overlap_min_lon = max(box1['min_lon'], box2['min_lon'])
    overlap_max_lon = min(box1['max_lon'], box2['max_lon'])

    # Check if there is an actual overlap
    if overlap_min_lat < overlap_max_lat and overlap_min_lon < overlap_max_lon:
        # Calculate OSRM distances for the diagonal of the overlap
        top_left = (overlap_max_lat, overlap_min_lon)
        bottom_right = (overlap_min_lat, overlap_max_lon)
        overlap_distance = get_osrm_distance(top_left, bottom_right)

        # Approximate the area as a square using this distance
        return (overlap_distance ** 2) / 2 if overlap_distance else 0
    else:
        return 0  # No overlap

# Function to evaluate a partnership
def rank_company_pairs_by_overlap(data):
    bounding_boxes = calculate_bounding_boxes(data)
    rankings = []

    for (company1, box1), (company2, box2) in combinations(bounding_boxes.items(), 2):
        # Calculate overlap area
        overlap_area = calculate_overlap_area_osrm(box1, box2)

        rankings.append((company1, company2, overlap_area))

    # Sort by overlap area in descending order (higher overlap is better)
    rankings.sort(key=lambda x: x[2], reverse=True)

    return pd.DataFrame(rankings, columns=['Company1', 'Company2', 'Overlap Area'])


def plot_dbscan_clusters(data, map_center):
    coords = np.radians(data[['lat', 'lon']].values)
    dbscan = DBSCAN(eps=0.3, min_samples=2, metric='haversine').fit(coords)
    data['cluster'] = dbscan.labels_
    m = folium.Map(location=map_center, zoom_start=12)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'lightblue', 'lightgreen', 'gray']

    for idx, row in data.iterrows():
        cluster = row['cluster']
        color = colors[cluster % len(colors)] if cluster != -1 else 'gray'
        folium.CircleMarker(
            location=(row['lat'], row['lon']),
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"Cluster: {cluster}"
        ).add_to(m)

    # Save the map as HTML
    map_file = "map.html"
    m.save(map_file)
    return map_file
