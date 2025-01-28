# solver.py

import pandas as pd
import requests
import networkx as nx
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from gurobipy import Model, GRB, quicksum

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
# Gurobi-based CVRP solver
################################################################################

def solve_cvrp_numeric_ids(cost_per_truck,
                           cost_per_km,
                           timelimit,
                           approx,
                           nmbr_loc,
                           distance_matrix,
                           depot_name="Universal_Depot"):
    """
    Solve a Capacitated Vehicle Routing Problem using a Gurobi MIP model.

    Arguments:
        cost_per_truck: Fixed cost for using one vehicle (truck).
        cost_per_km: Cost per kilometer traveled.
        timelimit: Time limit (in seconds) for Gurobi to solve the model.
        approx: If True, allow a larger MIP gap for an approximate solution.
        nmbr_loc: The vehicle capacity (the max number of customer visits per route).
        distance_matrix: A pandas DataFrame containing distances (in meters).
        depot_name: The name/index of the universal depot in distance_matrix.

    Returns:
        {
            "Routes": {"Vehicle_1": [...], "Vehicle_2": [...], ...},
            "Total Distance": total_cost
        }
        or
        {
            "Routes": None,
            "Total Distance": float("inf")
        } in case of an error.
    """

    # --- 1) Build index mapping (depot -> 0, customers -> 1..n) ---
    labels = list(distance_matrix.index)
    if depot_name not in labels:
        print(f"Depot '{depot_name}' not found in distance matrix.")
        return {"Routes": None, "Total Distance": float("inf")}

    # Map each label to an integer index
    idx_map = {}
    node_idx = 1  # start customer indexing at 1, keep 0 for depot
    for lab in labels:
        if lab == depot_name:
            idx_map[lab] = 0
        else:
            idx_map[lab] = node_idx
            node_idx += 1

    # Invert idx_map for reverse lookup
    index_map = {v: k for k, v in idx_map.items()}

    n = len(labels) - 1  # number of customers
    # distance[i][j] in meters
    distance = {}
    for i_lab in labels:
        i = idx_map[i_lab]
        distance[i] = {}
        for j_lab in labels:
            j = idx_map[j_lab]
            distance[i][j] = distance_matrix.loc[i_lab, j_lab]

    # Demand: 1 unit for each customer, 0 for depot
    demand = [0]*(n+1)
    for lab in labels:
        if lab != depot_name:
            demand[idx_map[lab]] = 1

    # Capacity
    capacity = nmbr_loc

    # --- 2) Build Gurobi model ---
    try:
        m = Model("CVRP_Gurobi")

        # Set time limit
        m.Params.TimeLimit = timelimit

        # If approx == True, allow larger MIP gap (for a faster approximate solution)
        if approx:
            m.Params.MIPGap = 0.05  # 5% gap; adjust as needed

        # --- 3) Variables ---
        # x[i,j] = 1 if vehicle goes directly from i to j, 0 otherwise
        x = {}
        for i in range(n+1):
            for j in range(n+1):
                if i != j:
                    x[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # u[i] = load (or "flow") upon leaving node i (subtour elimination)
        # only for i=1..n (customers), i=0 is depot
        u = {}
        for i in range(1, n+1):
            u[i] = m.addVar(lb=0.0, ub=float(capacity), vtype=GRB.CONTINUOUS, name=f"u_{i}")

        # Number of vehicles used
        z = m.addVar(lb=0.0, ub=float(n), vtype=GRB.INTEGER, name="z")

        m.update()

        # --- 4) Constraints ---

        # (a) Sum of edges out of the depot = number of vehicles used
        m.addConstr(quicksum(x[(0, j)] for j in range(1, n+1)) == z,
                    name="vehicle_count_from_depot")

        # (b) Sum of edges into the depot = number of vehicles used
        m.addConstr(quicksum(x[(i, 0)] for i in range(1, n+1)) == z,
                    name="vehicle_count_to_depot")

        # (c) Each customer is visited exactly once: in-degree = 1
        for i in range(1, n+1):
            m.addConstr(quicksum(x[(j, i)] for j in range(n+1) if j != i) == 1,
                        name=f"in_degree_{i}")

        # (d) Each customer is left exactly once: out-degree = 1
        for i in range(1, n+1):
            m.addConstr(quicksum(x[(i, j)] for j in range(n+1) if j != i) == 1,
                        name=f"out_degree_{i}")

        # (e) Sub-tour elimination (flow-based):
        #     u[j] >= u[i] + demand[j] - capacity*(1 - x[i,j])
        for i in range(1, n+1):
            for j in range(1, n+1):
                if i != j:
                    m.addConstr(u[j] >= u[i] + demand[j] - capacity*(1 - x[(i, j)]),
                                name=f"flow_{i}_{j}")

        # Ensure u[i] >= demand[i]
        for i in range(1, n+1):
            m.addConstr(u[i] >= demand[i], name=f"u_lower_bound_{i}")

        # --- 5) Objective function ---
        # Total travel cost = sum(distance[i][j]* x[i,j]) * cost_per_km(meters->km)
        # + cost of trucks (# vehicles used * cost_per_truck)
        travel_cost = quicksum(distance[i][j] * cost_per_km * x[(i, j)]
                               for i in range(n+1) for j in range(n+1) if i != j)
        vehicle_cost = cost_per_truck * z
        m.setObjective(travel_cost + vehicle_cost, GRB.MINIMIZE)

        # --- 6) Solve ---
        m.optimize()

        # Check for an optimal or feasible solution
        if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            print("No feasible solution found by Gurobi.")
            return {"Routes": None, "Total Distance": float("inf")}

        # --- 7) Extract solution ---
        # Build routes by following x[i,j]=1 edges from depot outward.
        solution_edges = [(i, j) for i in range(n+1) for j in range(n+1)
                          if i != j and x[(i, j)].X > 0.5]

        # Number of vehicles used
        num_vehicles = int(round(z.X))

        # Reconstruct routes:
        # From the depot (0), find the next node and build the route until back to depot
        routes = []
        used_edges = set(solution_edges)  # to keep track of used edges

        for vehicle_num in range(1, num_vehicles + 1):
            route = [depot_name]
            current = 0  # start at depot

            while True:
                # Find the next node from current
                next_nodes = [j for (i, j) in used_edges if i == current]
                if not next_nodes:
                    break  # no further nodes

                next_node = next_nodes[0]
                route.append(index_map[next_node])
                used_edges.remove((current, next_node))
                current = next_node

                if current == 0:
                    break  # route completed

            routes.append(route)

        # Convert list of routes to a dictionary
        routes_as_dict = {f"Vehicle_{i}": route for i, route in enumerate(routes, 1)}

        # Compute total objective cost:
        total_cost = m.ObjVal  # This already includes cost_per_truck + travel cost

        return {
            "Routes": routes_as_dict,  # Now a dictionary
            "Total Distance": round(total_cost, 2)  # cost
        }

    except Exception as e:
        print("Error solving with Gurobi:", e)
        return {"Routes": None, "Total Distance": float("inf")}


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
    Solve VRP individually (solo) for each company using the Gurobi-based solver.
    """
    solo_results = []
    temp_matrices = all_filtered_matrices(distance_matrix)

    for filtered_matrix in temp_matrices:
        companies = unique_company_names(filtered_matrix)
        # Since it's solo, there should be only one company
        if len(companies) != 1:
            print(f"Unexpected number of companies in solo matrix: {companies}")
            continue
        company = companies[0]
        cvrp_result = solve_cvrp_numeric_ids(
            cost_per_truck=cost_per_truck,
            cost_per_km=cost_per_km,
            timelimit=timelimit,
            approx=approx,
            nmbr_loc=nmbr_loc,
            distance_matrix=filtered_matrix
        )
        solo_results.append((company, cvrp_result))

    data = []
    for company, result in solo_results:
        data.append({
            'Company': company,
            'Routes': result.get('Routes', None),
            'Total Distance': result.get('Total Distance', float("inf"))
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

        # Solve VRP with Gurobi
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
    (assuming an even number of companies).

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
    matching = nx.algorithms.matching.min_weight_matching(G, maxcardinality=True)
    chosen_pairs = []
    total_cost = 0.0
    for edge in matching:
        a, b = tuple(edge)
        cost_edge = G[a][b]['weight']
        total_cost += cost_edge
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
