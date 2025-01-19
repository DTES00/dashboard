from sklearn.cluster import DBSCAN
import folium
import pandas as pd
from matplotlib import colormaps
import random

def get_best_partnerships(ranked_pairs):
    used_companies = set()
    best_partnerships = []

    for _, row in ranked_pairs.iterrows():
      if 'Overlap Percentage' in row:
        company1, company2, overlap_percentage = row['Company1'], row['Company2'], row['Overlap Percentage']
      else:
          company1, company2, avg_distance = row['Company1'], row['Company2'], row['Average Distance']
      if company1 not in used_companies and company2 not in used_companies:
            best_partnerships.append(row)
            used_companies.add(company1)
            used_companies.add(company2)

    return pd.DataFrame(best_partnerships, columns=ranked_pairs.columns)



eps_values = {
    'mini': 50000,
    'medium': 20000,
    'many': 10000,
    'manyLarge': 5000,
    'Amsterdam': 1000
}


def get_clusters_for_file(distance_matrix):



    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=50000, min_samples=2, metric='precomputed')
    labels = dbscan.fit_predict(distance_matrix)

    return labels

# ranking partnerships

import numpy as np
from itertools import combinations
from geopy.distance import geodesic
import pandas as pd

# Function to rank partnerships using clusters
def rank_partnerships_using_clusters(data, labels, distance_matrix):
    # Add the cluster labels to the original data
    data['cluster'] = labels

    # Group data by cluster
    clustered_data = data.groupby('cluster')

    partnerships = []

    # For each cluster
    for cluster_id, group in clustered_data:
        if cluster_id == -1:
            continue  # Skip noise points

        # Extract unique companies in the cluster
        companies = group['name'].unique()

        # Calculate pairwise partnerships for companies in the cluster
        for company1, company2 in combinations(companies, 2):
            # Get indices for locations of each company
            indices1 = group[group['name'] == company1].index
            indices2 = group[group['name'] == company2].index

            # Compute distances based on the distance matrix
            distances = [
                distance_matrix[i, j] for i in indices1 for j in indices2
            ]

            # Calculate the heuristic (e.g., average distance)
            avg_distance = np.mean(distances)
            partnerships.append((cluster_id, company1, company2, avg_distance))

    # Sort partnerships by average distance (ascending)
    partnerships.sort(key=lambda x: x[3])
    return pd.DataFrame(partnerships, columns=['Cluster ID', 'Company1', 'Company2', 'Average Distance'])