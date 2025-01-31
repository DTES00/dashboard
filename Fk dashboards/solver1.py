# solver.py

import pandas as pd
import requests
from networkx import DiGraph
from vrpy import VehicleRoutingProblem
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
from itertools import combinations

################################################################################
# Create the distance matrix with OSRM in batches
################################################################################

def get_osrm_distance_submatrix(src_batch, dst_batch,
                                base_url="http://router.project-osrm.org",
                                profile="driving"):
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


def create_batched_distance_matrix(locations, batch_size=10, max_workers=4,
                                   base_url="http://router.project-osrm.org",
                                   profile="driving"):
    """
    Create a complete non-symmetric distance matrix by batching location pairs
    and parallelizing OSRM table requests. 'profile' can be "driving", "cycling",
    etc., depending on your OSRM setup.
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
            # Pass the base_url and profile here
            args_list.append((src_batch, dst_batch, base_url, profile))

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


################################################################################
# Basic VRP solver function for a single distance matrix
################################################################################

from vrpy import VehicleRoutingProblem
from networkx import DiGraph

def solve_cvrp_numeric_ids(
    cost_per_truck,
    cost_per_km,
    timelimit,
    approx,
    nmbr_loc,
    distance_matrix,
    depot_name="Universal_Depot"
):
    """
    Solve a Capacitated Vehicle Routing Problem using a distance matrix.
    Returns the solution with the original location labels (rather than numeric IDs).

    This version incorporates cost_per_truck directly into the solver's objective
    by adding an equivalent distance penalty on Source->customer edges, then
    subtracting that penalty before computing the final total cost.
    """
    # Create directed graph
    G = DiGraph()
    G.add_node("Source", demand=0)
    G.add_node("Sink", demand=0)

    # All labels in the distance matrix
    labels = list(distance_matrix.index)
    customer_labels = [lab for lab in labels if lab != depot_name]

    # Map each label to an integer, and store the reverse mapping
    label_to_id = {lab: idx for idx, lab in enumerate(customer_labels)}
    id_to_label = {idx: lab for lab, idx in label_to_id.items()}

    # Add customer nodes with demand=1
    for lab, node_id in label_to_id.items():
        G.add_node(node_id, demand=1)

    # Add edges for customer <-> customer
    for i_lab, i_id in label_to_id.items():
        for j_lab, j_id in label_to_id.items():
            if i_lab != j_lab:
                cost = distance_matrix.loc[i_lab, j_lab]  # in meters
                G.add_edge(i_id, j_id, cost=cost)

    # If cost_per_km is nonzero, convert cost_per_truck to "meters"
    # so it fits the same cost scale as distances.
    if cost_per_km != 0:
        truck_cost_in_meters = cost_per_truck * 1000.0 / cost_per_km
    else:
        # If cost_per_km is zero, we cannot do a direct conversion;
        # handle it in a way that fits your application logic:
        truck_cost_in_meters = 0

    # Add edges from Source -> customer (with truck penalty) and customer -> Sink
    for lab, node_id in label_to_id.items():
        cost_from_depot = distance_matrix.loc[depot_name, lab] + truck_cost_in_meters
        cost_to_depot = distance_matrix.loc[lab, depot_name]
        G.add_edge("Source", node_id, cost=cost_from_depot)
        G.add_edge(node_id, "Sink", cost=cost_to_depot)

    # Setup the VRP
    vrp = VehicleRoutingProblem(G)
    vrp.source = "Source"
    vrp.sink = "Sink"
    vrp.load_capacity = nmbr_loc  # each customer has demand=1
    vrp.exact = approx            # toggles approximation or exact cspy

    # Solve with a time limit and chosen solver
    try:
        vrp.solve(
            cspy=True,
            pricing_strategy="Exact",
            time_limit=timelimit,
            solver="cbc"  # could be gurobi, glpk, etc.
        )
    except Exception as e:
        print("Error solving VRP:", e)
        return {"Routes": None, "Total Distance": float("inf")}

    # vrp.best_value includes:
    #   (sum_of_all_route_distances_in_meters) + (num_vehicles * truck_cost_in_meters)
    num_vehicles = len(vrp.best_routes)

    # Subtract the total truck penalty to get the "pure distance" in meters
    adjusted_best_value = vrp.best_value - num_vehicles * truck_cost_in_meters

    # Convert meters to kilometers
    dist_in_km = adjusted_best_value / 1000.0

    # Now compute final total cost in the exact same way:
    total_cost = dist_in_km * cost_per_km + num_vehicles * cost_per_truck

    # ----------------------------------------------------------------------
    # Remap numeric IDs in vrp.best_routes to the original location labels
    # ----------------------------------------------------------------------
    mapped_routes = {}
    for route_id, node_list in vrp.best_routes.items():
        new_nodes = []
        for node in node_list:
            if node == "Source" or node == "Sink":
                new_nodes.append(depot_name)
            else:
                new_nodes.append(id_to_label[node])
        mapped_routes[route_id] = new_nodes

    # Return the same dictionary structure
    return {
        "Routes": mapped_routes,
        "Total Distance": total_cost
    }




################################################################################
# Compute SOLO route solutions
################################################################################

def all_filtered_matrices(distance_matrix):
    """
    Generate filtered matrices for each single company + depot.
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


def unique_company_names(distance_matrix, depot_name="Universal_Depot"):
    """
    Return a sorted list of unique company names from distance_matrix.
    """
    labels = distance_matrix.index.tolist()
    companies = {label.rsplit("_", 1)[0] for label in labels if label != depot_name}
    return sorted(companies)


def solo_routes(cost_per_truck, cost_per_km, timelimit, approx, nmbr_loc, distance_matrix):
    """
    Solve VRP individually (solo) for each company.
    """
    solo_results = []
    temp_matrices = all_filtered_matrices(distance_matrix)

    for matrices in temp_matrices:
        company = unique_company_names(matrices)
        cvrp_result = solve_cvrp_numeric_ids(
            cost_per_truck, cost_per_km, timelimit, approx, nmbr_loc, matrices
        )
        solo_results.append((company, cvrp_result))

    data = []
    for company_list, result in solo_results:
        c = company_list[0] if isinstance(company_list, list) and company_list else company_list
        data.append({
            'Company': c,
            'Routes': result.get('Routes', None),
            'Total Distance': round(result.get('Total Distance', 0), 1)
        })

    df = pd.DataFrame(data)
    return df


################################################################################
# Solve VRP for a given set of pairs (Bounding Box or Cluster approach)
################################################################################

def solve_vrp_for_all_pairs_in_dataframe(pairs_df, distance_matrix,
                                         cost_per_truck, cost_per_km,
                                         timelimit, approx, nmbr_loc):
    """
    Solve VRP for only the pairs listed in pairs_df, returning a DataFrame
    with columns [Company1, Company2, Routes, Total Distance].
    """
    results = []

    for _, row in pairs_df.iterrows():
        company1 = row["Company1"]
        company2 = row["Company2"]

        # Filter matrix for these two companies + depot
        relevant_nodes = [
            node for node in distance_matrix.index
            if node.startswith(company1) or node.startswith(company2) or node == "Universal_Depot"
        ]
        filtered_matrix = distance_matrix.loc[relevant_nodes, relevant_nodes]

        # Solve VRP
        vrp_result = solve_cvrp_numeric_ids(
            cost_per_truck=cost_per_truck,
            cost_per_km=cost_per_km,
            timelimit=timelimit,
            approx=approx,
            nmbr_loc=nmbr_loc,
            distance_matrix=filtered_matrix,
            depot_name="Universal_Depot"
        )

        pair_cost = vrp_result.get("Total Distance", float("inf"))
        results.append({
            "Company1": company1,
            "Company2": company2,
            "Routes": vrp_result.get("Routes", None),
            "Total Distance": pair_cost
        })

    return pd.DataFrame(results)


################################################################################
# Solve VRP for ALL possible pairs (perfect matching approach)
################################################################################

def solve_vrp_for_all_possible_pairs(distance_matrix,
                                     cost_per_truck,
                                     cost_per_km,
                                     timelimit,
                                     approx,
                                     nmbr_loc):
    """
    Enumerate VRP for every possible pair of companies, then do a
    min-weight perfect matching to cover all companies in pairs
    (assuming an even number of companies, no solos).
    
    Returns:
        matching_pairs_df (pd.DataFrame): columns=["Company1","Company2","Total Distance"]
        total_cost (float): sum of VRP costs in the matching.
    """
    # 1) Unique companies
    unique_companies = sorted({
        name.split('_')[0]
        for name in distance_matrix.index
        if name != "Universal_Depot"
    })
    n_comp = len(unique_companies)
    if n_comp % 2 != 0:
        raise ValueError("Number of companies is odd; perfect matching requires even count.")

    # 2) Compute VRP cost for every 2-company pair
    pair_results = []
    for company1, company2 in combinations(unique_companies, 2):
        # Filter matrix for these two companies + depot
        relevant_nodes = [
            node for node in distance_matrix.index
            if node.startswith(company1) or node.startswith(company2) or node == "Universal_Depot"
        ]
        filtered_matrix = distance_matrix.loc[relevant_nodes, relevant_nodes]

        vrp_result = solve_cvrp_numeric_ids(
            cost_per_truck=cost_per_truck,
            cost_per_km=cost_per_km,
            timelimit=timelimit,
            approx=approx,
            nmbr_loc=nmbr_loc,
            distance_matrix=filtered_matrix,
            depot_name="Universal_Depot"
        )

        cost_val = vrp_result.get("Total Distance", float("inf"))
        pair_results.append((company1, company2, cost_val))

    all_pairs_df = pd.DataFrame(pair_results, columns=["Company1", "Company2", "Total Distance"])

    # 3) Build a graph for min-weight perfect matching
    G = nx.Graph()
    G.add_nodes_from(unique_companies)

    for _, row in all_pairs_df.iterrows():
        c1, c2, cost_val = row["Company1"], row["Company2"], row["Total Distance"]
        G.add_edge(c1, c2, weight=cost_val)

    # 4) Solve min-weight perfect matching
    matching = nx.algorithms.matching.min_weight_matching(G)
    chosen_pairs = []
    total_cost = 0.0
    for edge in matching:
        a, b = tuple(edge)
        cost_edge = G[a][b]['weight']
        total_cost += cost_edge
        # keep them in sorted order for consistency
        if a < b:
            chosen_pairs.append((a, b, cost_edge))
        else:
            chosen_pairs.append((b, a, cost_edge))

    matching_pairs_df = pd.DataFrame(chosen_pairs, columns=["Company1", "Company2", "Total Distance"])
    return matching_pairs_df, total_cost


def get_filtered_matrix_for_pair(distance_matrix, company1, company2):
    """
    Filters the distance matrix for the given companies and the universal depot.
    Considers all rows/columns that start with the company names or "Universal_Depot".
    """
    distance_matrix.index = distance_matrix.index.str.strip().str.replace('"', '').str.replace("'", '')
    distance_matrix.columns = distance_matrix.columns.str.strip().str.replace('"', '').str.replace("'", '')

    selected_companies = [
        idx for idx in distance_matrix.index
        if idx.startswith(company1) or idx.startswith(company2) or idx == "Universal_Depot"
    ]
    filtered_matrix = distance_matrix.loc[selected_companies, selected_companies]
    return filtered_matrix


def get_individual_company_matrices(company1, company2, distance_matrix):
    """
    Returns two distance matrices: one for 'company1' + depot, and one for 'company2' + depot.
    """
    distance_matrix.index = distance_matrix.index.str.strip().str.replace('"', '').str.replace("'", '')
    distance_matrix.columns = distance_matrix.columns.str.strip().str.replace('"', '').str.replace("'", '')

    def filter_for_company(company_name):
        relevant_nodes = [
            node for node in distance_matrix.index if node.startswith(company_name) or node == "Universal_Depot"
        ]
        return distance_matrix.loc[relevant_nodes, relevant_nodes]

    company1_matrix = filter_for_company(company1)
    company2_matrix = filter_for_company(company2)

    return company1_matrix, company2_matrix

# solver1.py

import pandas as pd
import requests
from networkx import DiGraph
from vrpy import VehicleRoutingProblem
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
from itertools import combinations

# ... (other existing imports and functions) ...


def solve_vrp_for_all_pairs(best_pairs_df, distance_matrix, cost_per_truck, cost_per_km, time_per_vrp, flag, nmbr_loc, algorithm):
    """
    Wrapper function to solve VRP for all pairs in the provided DataFrame.
    
    Parameters:
    - best_pairs_df (pd.DataFrame): DataFrame containing the best pairs with columns ['Company1', 'Company2']
    - distance_matrix (pd.DataFrame): Complete distance matrix
    - cost_per_truck (float): Cost per truck
    - cost_per_km (float): Cost per kilometer
    - time_per_vrp (int): Time limit per VRP in seconds
    - flag (bool): Additional flag if needed
    - nmbr_loc (int): Maximum number of locations per route
    - algorithm (str): The heuristic used ('Bounding Box' or 'Clustering')
    
    Returns:
    - pd.DataFrame: Results containing [Company1, Company2, Routes, Total Distance]
    """
    return solve_vrp_for_all_pairs_in_dataframe(
        best_pairs_df,
        distance_matrix,
        cost_per_truck,
        cost_per_km,
        time_per_vrp,
        flag,
        nmbr_loc
    )
