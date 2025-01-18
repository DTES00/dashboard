import pandas as pd
import requests
from networkx import DiGraph
from vrpy import VehicleRoutingProblem
from concurrent.futures import ThreadPoolExecutor


# Constants
#UNIVERSAL_DEPOT = (52.0, 5.0)


# def process_data(data):
#     """
#     Process input data to generate a list of locations with lat/lon and unique names.
#     """
#     locations = []
#     for i, row in data.iterrows():
#         loc = {
#             'lon': row['lon'],
#             'lat': row['lat'],
#             'name': row['name'],
#             'unique_name': f"{row['name']}_{i}"
#         }
#         locations.append(loc)

#     # Add the Universal Depot
#     locations.append({
#         'lon': UNIVERSAL_DEPOT[1],
#         'lat': UNIVERSAL_DEPOT[0],
#         'name': 'Universal Depot',
#         'unique_name': 'Universal_Depot'
#     })
#     return locations


def get_osrm_distance_submatrix(src_batch, dst_batch, base_url="http://localhost:5000", profile="driving"):
    """
    Send a table request to OSRM to get distances between src_batch and dst_batch.
    Returns a 2D list of distances or None if there's an error.
    """
    combined = src_batch + dst_batch
    coords_str = ";".join(f"{loc['lon']},{loc['lat']}" for loc in combined)

    src_indices = list(range(0, len(src_batch)))
    dst_indices = list(range(len(src_batch), len(src_batch) + len(dst_batch)))

    url = f"{base_url}/table/v1/{profile}/{coords_str}"
    params = {
        "annotations": "distance",
        "sources": ";".join(map(str, src_indices)),
        "destinations": ";".join(map(str, dst_indices))
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("distances"), src_batch, dst_batch
    except requests.exceptions.RequestException as e:
        print(f"Error with OSRM request: {e}")
        return None, src_batch, dst_batch

def create_batched_distance_matrix(locations, batch_size=100, max_workers=4):
    """
    Create a complete non-symmetric distance matrix by batching location pairs and parallelizing requests.
    """
    num_locations = len(locations)
    all_unique_names = [loc['unique_name'] for loc in locations]

    # Initialize full distance matrix with NaN
    full_matrix = pd.DataFrame(
        data=float("nan"),
        index=all_unique_names,
        columns=all_unique_names
    )

    # Prepare batch combinations
    args_list = []
    for i in range(0, num_locations, batch_size):
        src_batch = locations[i : i + batch_size]
        for j in range(0, num_locations, batch_size):
            dst_batch = locations[j : j + batch_size]
            args_list.append((src_batch, dst_batch, "http://localhost:5000", "driving"))

    # Use ThreadPoolExecutor for parallel requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda args: get_osrm_distance_submatrix(*args), args_list)

    # Populate the full matrix with results
    for distances, src_batch, dst_batch in results:
        if distances is not None:
            src_unique_names = [loc['unique_name'] for loc in src_batch]
            dst_unique_names = [loc['unique_name'] for loc in dst_batch]
            sub_df = pd.DataFrame(
                data=distances,
                index=src_unique_names,
                columns=dst_unique_names
            )
            full_matrix.loc[src_unique_names, dst_unique_names] = sub_df

    return full_matrix


def filter_company_distances(max_loc, distance_matrix, company_names, depot_name="Universal_Depot"):
    """
    Filter the distance matrix to include specific companies and the depot.
    """
    relevant_indices = [
        index for index in distance_matrix.index
        if any(company_name in index for company_name in company_names) or depot_name in index
    ]
    return distance_matrix.loc[relevant_indices, relevant_indices]


def solve_cvrp_numeric_ids(cost_per_truck, cost_per_km, timelimit, approx, nmbr_loc, distance_matrix, depot_name="Universal_Depot"):
    """
    Solve a Capacitated Vehicle Routing Problem using a distance matrix.
    """
    G = DiGraph()
    G.add_node("Source", demand=0)
    G.add_node("Sink", demand=0)

    labels = list(distance_matrix.index)
    customer_labels = [lab for lab in labels if lab != depot_name]

    label_to_id = {lab: idx for idx, lab in enumerate(customer_labels)}

    for lab, node_id in label_to_id.items():
        G.add_node(node_id, demand=1)

    for i_lab, i_id in label_to_id.items():
        for j_lab, j_id in label_to_id.items():
            if i_lab != j_lab:
                cost = distance_matrix.loc[i_lab, j_lab]
                G.add_edge(i_id, j_id, cost=cost)

    for lab, node_id in label_to_id.items():
        cost_from_depot = distance_matrix.loc[depot_name, lab]
        cost_to_depot = distance_matrix.loc[lab, depot_name]
        G.add_edge("Source", node_id, cost=cost_from_depot)
        G.add_edge(node_id, "Sink", cost=cost_to_depot)

    vrp = VehicleRoutingProblem(G)
    vrp.source = "Source"
    vrp.sink = "Sink"
    vrp.load_capacity = nmbr_loc
    vrp.exact = approx

    try:
        vrp.solve(time_limit=timelimit)
    except Exception as e:
        print("Error:", e)
        return None

    return {
        "Routes": vrp.best_routes,
        "Total Distance": vrp.best_value * (cost_per_km / 1000) + cost_per_truck * len(vrp.best_routes)
    }
def unique_company_names(distance_matrix, depot_name="Universal_Depot"):
    """
    Extracts unique company names from a distance matrix, excluding the depot.

    
    Parameters:
        distance_matrix (pd.DataFrame): Distance matrix with rows and columns as node labels.
        depot_name (str): The name of the depot to exclude (default is "Universal_Depot").
    
    Returns:
        list: A sorted list of unique company names.
    """
    # Extract row/column labels
    labels = distance_matrix.index.tolist()
    
    # Extract company names by splitting at the last underscore and excluding the depot
    company_names = {label.rsplit("_", 1)[0] for label in labels if label != depot_name}
    
    # Convert to a sorted list and return
    return sorted(company_names)



def all_filtered_matrices(distance_matrix):
    """
    Generate filtered matrices for all companies.
    """
    company_names = set(name.split('_')[0] for name in distance_matrix.index if name != "Universal_Depot")
    matrices = []

    for company in company_names:
        relevant_nodes = [
            node for node in distance_matrix.index
            if node.startswith(company) or node == "Universal_Depot"
        ]
        filtered_matrix = distance_matrix.loc[relevant_nodes, relevant_nodes]
        matrices.append(filtered_matrix)

    return matrices


def solo_routes(cost_per_truck, cost_per_km, timelimit, approx, nmbr_loc, distance_matrix):
    solo_results = []
    temp_matrices = all_filtered_matrices(distance_matrix)
    for matrices in temp_matrices:
        company_name = unique_company_names(matrices)
        cvrp_result = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km, timelimit, approx, nmbr_loc, matrices, depot_name="Universal_Depot")
        solo_results.append((company_name, cvrp_result))
    
    # Convert results into a DataFrame
    data = []
    for company, result in solo_results:
        data.append({
            'Company': company[0] if isinstance(company, list) else company,  # Extract company name if it's in a list
            'Routes': result['Routes'] if 'Routes' in result else None,
            'Total Distance': round(result['Total Distance'], 1) if 'Total Distance' in result else None  # Round distance
        })

    df = pd.DataFrame(data)
    return df


def get_all_pairings(name_list):
    """
    Generate all possible pairings from a list of names.
    """
    pairings = [(name_list[i], name_list[j]) for i in range(len(name_list)) for j in range(i + 1, len(name_list))]
    return pairings

def solve_vrp_for_all_pairs_in_dataframe(pairs_df, distance_matrix, cost_per_truck, cost_per_km, timelimit, approx, nmbr_loc):
    """
    Solve VRPs for all pairs of companies listed in a DataFrame.
    
    Parameters:
        pairs_df (pd.DataFrame): DataFrame with columns ["Company1", "Company2", "Overlap Percentage"].
        distance_matrix (pd.DataFrame): The original distance matrix with all locations.
        cost_per_truck (float): Fixed cost per truck.
        cost_per_km (float): Cost per kilometer.
        timelimit (int): Time limit for solving the VRP (in seconds).
        approx (bool): Whether to use an approximate solver.
        nmbr_loc (int): Maximum capacity of each truck.
    
    Returns:
        pd.DataFrame: DataFrame with VRP results for each pair.
    """
    results = []

    for _, row in pairs_df.iterrows():
        company1 = row["Company1"]
        company2 = row["Company2"]
        overlap_percentage = row["Overlap Percentage"]

        # Filter the distance matrix for the two companies and the depot
        relevant_nodes = [
            node for node in distance_matrix.index
            if node.startswith(company1) or node.startswith(company2) or node == "Universal_Depot"
        ]
        filtered_matrix = distance_matrix.loc[relevant_nodes, relevant_nodes]

        # Solve the VRP for the filtered matrix
        vrp_result = solve_cvrp_numeric_ids(
            cost_per_truck=cost_per_truck,
            cost_per_km=cost_per_km,
            timelimit=timelimit,
            approx=approx,
            nmbr_loc=nmbr_loc,
            distance_matrix=filtered_matrix,
            depot_name="Universal_Depot"
        )

        # Add the result to the list
        results.append({
            "Company1": company1,
            "Company2": company2,
            "Overlap Percentage": overlap_percentage,
            "Routes": vrp_result.get("Routes", None),
            "Total Distance": vrp_result.get("Total Distance", None)
        })

    # Convert the results into a DataFrame
    return pd.DataFrame(results)
