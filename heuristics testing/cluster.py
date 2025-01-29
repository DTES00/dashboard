from sklearn.cluster import DBSCAN
import folium
import pandas as pd
from matplotlib import colormaps
import random
import numpy as np
from itertools import combinations
import itertools


def calculate_eps_k_distance_from_matrix(distance_matrix, k=3):
    # Sort distances for each point (row), excluding the diagonal (self-distance)
    sorted_distances = np.sort(distance_matrix, axis=1)
    
    # Extract the distances to the k-th nearest neighbor
    k_distances = sorted_distances[:, k]
    
    filtered_k_distances = k_distances[k_distances > 1000]
    eps = np.median(filtered_k_distances)

    return int(eps)

def get_best_partnerships(ranked_pairs):
    used_companies = set()
    best_partnerships = []

    for _, row in ranked_pairs.iterrows():
      if 'Overlap Percentage' in row:
        company1, company2, overlap_percentage = row['Company1'], row['Company2'], row['Overlap Percentage']
      else:
          company1, company2, avg_distance = row['Company1'], row['Company2'], row['Clustering Score']
      if company1 not in used_companies and company2 not in used_companies:
            best_partnerships.append(row)
            used_companies.add(company1)
            used_companies.add(company2)

    return pd.DataFrame(best_partnerships, columns=ranked_pairs.columns)



def get_clusters_for_file(distance_matrix):
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=calculate_eps_k_distance_from_matrix(distance_matrix), min_samples=2, metric='precomputed')
    labels = dbscan.fit_predict(distance_matrix)

    return labels


# outlier treatment

def assign_outliers_to_closest_cluster_using_matrix(data, distance_matrix):
    # Iterate over each row in the data DataFrame to check for outliers
    for idx, row in data.iterrows():
        if row['cluster'] == -1:  # If the location is an outlier
            company_name = row['name']
            print(company_name)
            outlier_location = idx
            
            # Get all the other locations for this company that are not outliers
            valid_location_names = data[(data['name'] == company_name) & (data['cluster'] != -1)].index.tolist()
            print(valid_location_names)
            
            # Create a list to hold distances from the outlier to the other valid locations
            distances = []
            
            for valid_location in valid_location_names:
                dist = distance_matrix[outlier_location, valid_location]
                distances.append((valid_location, dist))
            
            # Find the closest location
            closest_location = min(distances, key=lambda x: x[1])[0]
            
            # Get the cluster of the closest location
            closest_cluster = data.loc[closest_location, 'cluster']
            
            # Update the outlier's cluster to the closest cluster
            data.loc[outlier_location, 'cluster'] = closest_cluster
    
    return data

# ranking partnerships

def get_shared_cluster_locations(data, company1, company2):
    # Find clusters where both companies have at least one location
    clusters_with_both = data[data['name'] == company1]['cluster'].unique()
    
    # Now find clusters where both companies are present by checking company2's clusters as well
    clusters_with_both = data[(data['name'] == company2) & (data['cluster'].isin(clusters_with_both))]['cluster'].unique()

    clusters_with_both = [cluster for cluster in clusters_with_both if cluster != -1]
    
    
    # Filter locations that are in those shared clusters, and belong to both companies
    filtered_data = data[(data['cluster'].isin(clusters_with_both)) & 
                         (data['name'].isin([company1, company2]))]
    
    return filtered_data

def calculate_distances_within_clusters(shared_data, distance_matrix):
    # Initialize lists to store results
    avg_distances = []
    max_distances = []
    
    # Iterate through each cluster in the shared data
    clusters = shared_data['cluster'].unique()
    
    for cluster in clusters:
        # Get locations for this cluster
        cluster_data = shared_data[shared_data['cluster'] == cluster]
        
        # If there are more than one location in the cluster, calculate distances
        if len(cluster_data) > 1:
            # Extract the indices of the locations
            indices = cluster_data.index.tolist()
            
            # Initialize distance list for this cluster
            cluster_distances = []
            
            # Calculate pairwise distances
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):  # avoid repeating pairs
                    loc1_index = indices[i]
                    loc2_index = indices[j]
                    
                    # Get the distance from the distance matrix
                    distance = distance_matrix.iloc[loc1_index, loc2_index]
                    cluster_distances.append(distance)
            
            # Calculate average and maximum distances for this cluster
            avg_distances.append(np.mean(cluster_distances))
            max_distances.append(np.max(cluster_distances))
    
    # Calculate overall averages
    overall_avg_distance = np.mean(avg_distances) if avg_distances else 0
    overall_max_distance = np.max(max_distances) if max_distances else 0
    
    return overall_avg_distance, overall_max_distance


# Function to rank partnerships using clusters
def rank_partnerships_using_clusters(data, labels, distance_matrix):
    cluster_distance_matrix = pd.DataFrame(distance_matrix)
    cluster_distance_matrix=cluster_distance_matrix.iloc[:-1, :-1]
    labels=pd.DataFrame(labels)
    data['cluster']=labels
    #data = assign_outliers_to_closest_cluster_using_matrix(data,distance_matrix)
    data['cluster']=data['cluster'].iloc[:-1]
    
   
    results = []
    
  
    companies = data['name'].unique()

    # Generate all unique pairs of companies
    company_pairs = itertools.combinations(companies, 2)

    # Iterate through each pair of companies
    for company1, company2 in company_pairs:
        
      # Get the locations for both companies
      company1_locs = data[data['name'] == company1]
      company2_locs = data[data['name'] == company2]

      # Call the function to get the shared cluster locations
      merged = get_shared_cluster_locations(data, company1, company2)
      
      # Total locations from both companies
      total_locations = len(company1_locs) + len(company2_locs)
      shared_locations = len(merged)

      # Step 2: Calculate the distances within each cluster
      avg_distance, max_distance = calculate_distances_within_clusters(merged, cluster_distance_matrix)

      ranking = 0.5* shared_locations/total_locations + 0.5* (1-avg_distance/max_distance)
        
        
      # Append the result for this pair of companies
      results.append({
            'Company1': company1,
            'Company2': company2,
            'Clustering Score': ranking
        })

    
    # Convert resuls to a DataFrame for easy analysis
    ranking_df = pd.DataFrame(results)
    ranking_df = ranking_df.sort_values(by='Clustering Score', ascending=False)
    
    return ranking_df

