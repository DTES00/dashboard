from sklearn.neighbors import NearestNeighbors
import pandas as pd

"""# Bounding box"""

# Bounding box - Technique 2

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from itertools import combinations
from functools import lru_cache
import requests
import folium
from joblib import Parallel, delayed
from numba import jit
import numpy as np

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
        distance = data['routes'][0]['distance']  # Distance in meters
        return distance
    except requests.exceptions.RequestException as e:
        print(f"Error fetching OSRM data: {e}")
        return None

# Define central depot coordinates
DEPOT_LAT, DEPOT_LON = 52.333847, 4.865261  # Example coordinates
central_depot = (DEPOT_LAT, DEPOT_LON)

"""# Bounding Box"""
# Function to calculate bounding boxes
def calculate_bounding_boxes(data):
    """
    Calculate the bounding boxes for each company.
    """
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

# Function to calculate overlap area using bounding boxes
def calculate_overlap_area(box1, box2):
    """
    Calculate overlap area between two bounding boxes.
    """
    # Define bounding box polygons
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

    # Calculate intersection area
    if polygon1.intersects(polygon2):
        return polygon1.intersection(polygon2).area
    return 0

# Function to calculate bounding box area
def calculate_bounding_box_area(box):
    """
    Calculate the area of a bounding box.
    """
    width = box['max_lon'] - box['min_lon']
    height = box['max_lat'] - box['min_lat']
    return width * height

# Function to rank company pairs by overlap percentage
def rank_company_pairs_by_overlap_percentage(data):
    """
    Rank company pairs based on overlap percentage using bounding boxes.
    """
    # Calculate bounding boxes
    bounding_boxes = calculate_bounding_boxes(data)

    # Calculate rankings
    rankings = []
    for (company1, box1), (company2, box2) in combinations(bounding_boxes.items(), 2):
        # Calculate overlap area
        overlap_area = calculate_overlap_area(box1, box2)

        # Calculate bounding box areas
        area1 = calculate_bounding_box_area(box1)
        area2 = calculate_bounding_box_area(box2)
        total_area = area1 + area2

        # Calculate overlap percentage
        overlap_percentage = (overlap_area / total_area * 100) if total_area > 0 else 0

        # Append results
        rankings.append((company1, company2, overlap_percentage))

    # Sort by overlap percentage in descending order
    rankings.sort(key=lambda x: x[2], reverse=True)

    return pd.DataFrame(rankings, columns=['Company1', 'Company2', 'Overlap Percentage'])

# Function to visualize bounding boxes on a map
def visualize_bounding_boxes(data):
    """
    Visualize bounding boxes for each company on a map.
    """
    bounding_boxes = calculate_bounding_boxes(data)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    company_colors = {company: colors[i % len(colors)] for i, company in enumerate(bounding_boxes.keys())}

    # Create the map
    map_center = [data['lat'].mean(), data['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    # Add bounding boxes
    for company, box in bounding_boxes.items():
        bounds = [
            [box['min_lat'], box['min_lon']],
            [box['max_lat'], box['min_lon']],
            [box['max_lat'], box['max_lon']],
            [box['min_lat'], box['max_lon']],
            [box['min_lat'], box['min_lon']]
        ]
        folium.PolyLine(bounds, color=company_colors[company], weight=2, popup=company).add_to(m)

    # Add company points
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color=company_colors[row['name']],
            fill=True,
            popup=row['name']
        ).add_to(m)

    # Save the map
    m.save("bounding_boxes_map.html")
    return m



def get_best_partnerships(ranked_pairs):
    used_companies = set()
    best_partnerships = []

    for _, row in ranked_pairs.iterrows():
        company1, company2, overlap_percentage = row['Company1'], row['Company2'], row['Overlap Percentage']
        if company1 not in used_companies and company2 not in used_companies:
            best_partnerships.append(row)
            used_companies.add(company1)
            used_companies.add(company2)

    return pd.DataFrame(best_partnerships, columns=ranked_pairs.columns)

