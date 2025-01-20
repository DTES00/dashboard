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
    # (Note: If your local OSRM is at http://localhost:5000, set base_url = "http://localhost:5000" above)
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

    # Add each customer node with demand=1
    for lab, node_id in label_to_id.items():
        G.add_node(node_id, demand=1)

    # Add edges for customer <-> customer
    for i_lab, i_id in label_to_id.items():
        for j_lab, j_id in label_to_id.items():
            if i_lab != j_lab:
                cost = distance_matrix.loc[i_lab, j_lab]
                G.add_edge(i_id, j_id, cost=cost)

    # Edges from Source -> customer, and customer -> Sink
    for lab, node_id in label_to_id.items():
        cost_from_depot = distance_matrix.loc[depot_name, lab]
        cost_to_depot = distance_matrix.loc[lab, depot_name]
        G.add_edge("Source", node_id, cost=cost_from_depot)
        G.add_edge(node_id, "Sink", cost=cost_to_depot)

    # Setup VRP
    vrp = VehicleRoutingProblem(G)
    vrp.source = "Source"
    vrp.sink = "Sink"
    vrp.load_capacity = nmbr_loc
    vrp.exact = approx

    try:
        vrp.solve(time_limit=timelimit)
    except Exception as e:
        print("Error solving VRP:", e)
        return {"Routes": None, "Total Distance": float("inf")}

    # vrp.best_value is the sum of edge costs in the solution (in meters).
    # Convert cost from meters to km and add cost per truck (# of vehicles).
    num_vehicles = len(vrp.best_routes)
    dist_in_km = vrp.best_value / 1000.0
    total_cost = dist_in_km * cost_per_km + num_vehicles * cost_per_truck

    return {
        "Routes": vrp.best_routes,
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
    
    Typically used for bounding-box or cluster-based pairs (not all possible).
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
    minimum-weight perfect matching to cover all companies in pairs
    (assuming an even number of companies, no solos).
    
    Returns:
        matching_pairs_df (pd.DataFrame): columns = ["Company1","Company2","Total Distance"]
        total_cost (float): sum of the VRP costs in the matching.
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

    # 2) Compute VRP cost for all 2-company pairs
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

    # Put into DataFrame
    all_pairs_df = pd.DataFrame(pair_results, columns=["Company1", "Company2", "Total Distance"])

    # 3) Build a graph for min-weight perfect matching
    G = nx.Graph()
    G.add_nodes_from(unique_companies)

    for _, row in all_pairs_df.iterrows():
        c1, c2, cost_val = row["Company1"], row["Company2"], row["Total Distance"]
        G.add_edge(c1, c2, weight=cost_val)

    # 4) Min-weight perfect matching
    matching = nx.algorithms.matching.min_weight_matching(G)

    # matching is a set of frozensets, e.g. {('A','B'), ('C','D')}
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
