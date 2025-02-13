# app.py

import itertools
import ast
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import folium
from folium import PolyLine
from folium.features import DivIcon
from folium.map import LayerControl

from bounding_box import get_best_partnerships as get_best_partnerships_bb
from bounding_box import rank_company_pairs_by_overlap_percentage

from cluster import get_best_partnerships as get_best_partnerships_cluster
from cluster import rank_partnerships_using_clusters, get_clusters_for_file

from combined import combine_heuristics_normalized
from combined import get_best_partnerships as get_best_partnerships_combined

from solver1 import (
    create_batched_distance_matrix,
    solve_vrp_for_all_pairs_in_dataframe,
    solve_cvrp_numeric_ids,
    get_filtered_matrix_for_pair,
    get_individual_company_matrices,
    solo_routes
)

# App functions

NETHERLANDS_BOUNDS = {
    "min_lat": 50.5,
    "max_lat": 53.7,
    "min_lon": 3.36,
    "max_lon": 7.22,
}

def is_within_netherlands(lat, lon):
    """Rough bounding box check for Netherlands coordinates."""
    return (
        NETHERLANDS_BOUNDS["min_lat"] <= lat <= NETHERLANDS_BOUNDS["max_lat"]
        and NETHERLANDS_BOUNDS["min_lon"] <= lon <= NETHERLANDS_BOUNDS["max_lon"]
    )

def convert_df(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes (for downloads)."""
    return df.to_csv(index=False).encode('utf-8')

def rename_source_sink(route_stops):
    """Rename 'Source' or 'Sink' to 'Depot' in a list of stops."""
    return ["Depot" if stop in ("Source", "Sink") else stop for stop in route_stops]

def prepare_single_pair_csv(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a single-pair VRP result DataFrame into a row-per-vehicle format
    with columns: [Company1, Company2, Vehicle ID, Route, Total Cost].
    """
    rows = []
    for idx, row in pair_df.iterrows():
        c1 = row.get("Company1", f"Company1_{idx}")
        c2 = row.get("Company2", f"Company2_{idx}")
        total_cost = row.get("Total Distance", 0)  # Actually total cost, not just distance
        routes_dict = row.get("Routes", {})

        if isinstance(routes_dict, str):
            # Convert stringified dict back to Python object
            routes_dict = ast.literal_eval(routes_dict)

        if routes_dict is None:
            continue

        # Each key in routes_dict is a vehicle ID, value is a list of route stops
        for vehicle_id, stops_list in routes_dict.items():
            route_str = " -> ".join(str(stop) for stop in stops_list)
            rows.append({
                "Company1": c1,
                "Company2": c2,
                "Vehicle ID": vehicle_id,
                "Route": route_str,
                "Total Cost": total_cost
            })

    return pd.DataFrame(rows)

def prepare_pairs_vrp_csv(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a multi-row VRP result DataFrame into a single CSV
    with one row per vehicle route.
    """
    all_rows = []
    for idx, row in pair_df.iterrows():
        c1 = row.get("Company1", f"Company1_{idx}")
        c2 = row.get("Company2", f"Company2_{idx}")
        total_cost = row.get("Total Distance", 0)  # Actually total cost
        routes_dict = row.get("Routes", {})

        if isinstance(routes_dict, str):
            routes_dict = ast.literal_eval(routes_dict)

        if routes_dict is None:
            continue

        for vehicle_id, stops_list in routes_dict.items():
            cleaned_stops = rename_source_sink(stops_list)
            route_str = " -> ".join(map(str, cleaned_stops))
            all_rows.append({
                "Company1": c1,
                "Company2": c2,
                "Vehicle ID": vehicle_id,
                "Route": route_str,
                "Total Cost": total_cost
            })
    return pd.DataFrame(all_rows)

def generate_route_map_fixed_with_legend(
    depot_loc,
    data: pd.DataFrame,
    solution_df: pd.DataFrame,
    label_to_coords: dict,
    osrm_url="http://router.project-osrm.org",
    profile="driving",
    zoom_start=7
):
    """
    Generates a Folium map that shows VRP routes with labeled stops.
    Works with route outputs that use string labels (e.g. 'CompanyA_0')
    rather than numeric indices.

    Parameters
    ----------
    depot_loc : tuple of float
        (latitude, longitude) for the depot location.
    data : pd.DataFrame
        A DataFrame (e.g. your uploaded data) with 'lat' and 'lon' columns
        used to determine the map center.
    solution_df : pd.DataFrame
        DataFrame that must include columns "Company" and "Routes".
        - "Company": string identifying the company name for display.
        - "Routes": dict {vehicle_id: [label1, label2, ...]} for each row.
    label_to_coords : dict
        A dictionary {string_label: (latitude, longitude)} mapping each VRP
        label to its coordinates. Must include an entry for each relevant label
        except "Universal_Depot", which uses `depot_loc`.
    osrm_url : str
        Base URL for the OSRM service (local or public).
    profile : str
        OSRM profile ("driving", "cycling", etc.).
    zoom_start : int
        Initial zoom level for the Folium map.

    Returns
    -------
    map_obj : folium.Map
        The generated Folium map object.
    map_file : str
        The path to the saved HTML map file ("routes_map.html").
    """

    # Center the map around the mean lat/lon of your data
    center_lat = data["lat"].mean()
    center_lon = data["lon"].mean()
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # If "Company" is missing in the solution, default to someting
    if "Company" not in solution_df.columns:
        solution_df["Company"] = "NoCompanyName"

    # Iterate over each row in the solution DataFrame
    for _, row in solution_df.iterrows():
        company_name = row["Company"]
        routes_dict = row.get("Routes", {})

        fg = folium.FeatureGroup(name=f"{company_name} Routes", show=True)

        # Cycle through colors 
        route_colors = itertools.cycle([
            "blue", "green", "purple", "orange", "darkred",
            "lightred", "beige", "darkblue", "darkgreen",
            "cadetblue", "pink", "lightblue", "lightgreen",
            "gray", "black"
        ])

        for vehicle_id, stops_list in routes_dict.items():
            route_color = next(route_colors)

            # Build the lat/lon for each stop in the route
            route_coords = []
            for stop_label in stops_list:
                if stop_label == "Universal_Depot":
                    # Use the depot coordinates
                    route_coords.append(depot_loc)
                else:
                    # Look up the location from label_to_coords
                    if stop_label in label_to_coords:
                        route_coords.append(label_to_coords[stop_label])
                    else:
                        print(f"[WARNING] Label '{stop_label}' not found in label_to_coords! "
                              "Using depot location as fallback.")
                        route_coords.append(depot_loc)

            # Build OSRM polylines
            real_path = []
            for i in range(len(route_coords) - 1):
                start_lat, start_lon = route_coords[i]
                end_lat, end_lon = route_coords[i + 1]
                coords_str = f"{start_lon},{start_lat};{end_lon},{end_lat}"
                url = f"{osrm_url}/route/v1/{profile}/{coords_str}?geometries=geojson&steps=true"

                try:
                    resp = requests.get(url)
                    resp.raise_for_status()
                    osrm_data = resp.json()
                    if osrm_data.get("routes"):
                        geometry = osrm_data["routes"][0]["geometry"]["coordinates"]
                        # geometry is a list of [lon, lat]
                        for (lon_, lat_) in geometry:
                            real_path.append((lat_, lon_))
                except requests.exceptions.RequestException as e:
                    print(f"[ERROR] OSRM request failed: {e}")
                    # If OSRM fails, fallback to direct lines
                    real_path.append((start_lat, start_lon))
                    real_path.append((end_lat, end_lon))

            # Draw the route polyline on the map
            if real_path:
                PolyLine(locations=real_path, color=route_color, weight=4, opacity=0.8).add_to(fg)

            # Add numbered markers for each stop
            for stop_order, (lat_, lon_) in enumerate(route_coords):
                label_number = stop_order + 1
                folium.map.Marker(
                    [lat_, lon_],
                    icon=DivIcon(
                        icon_size=(30, 30),
                        icon_anchor=(15, 15),
                        html=(
                            f"<div style='font-size:12pt;"
                            f"color:black;"
                            f"background-color:rgba(255,255,255,0.7);"
                            f"border:1px solid black;"
                            f"border-radius:15px;"
                            f"width:30px;height:30px;"
                            f"text-align:center;line-height:30px;'>"
                            f"{label_number}</div>"
                        )
                    ),
                    popup=f"Stop {label_number}"
                ).add_to(fg)

        fg.add_to(map_obj)

    LayerControl(collapsed=False).add_to(map_obj)

    map_file = "routes_map.html"
    map_obj.save(map_file)
    print(f"Map saved to {map_file}")

    return map_obj, map_file

# App begins here

# Initialize Streamlit session state variables
for key in [
    "uploaded_file",
    "uploaded_data",
    "distance_matrix_generated",
    "distance_matrix",
    "ranked_pairs",
    "best_partnerships",
    "pair_result_selected",
    "selected_pair_comparison",
    "pair_result_bb",
    "bb_comparison_table",
    "all_pairs_result",
    "all_pairs_total_cost",
    "pair_map_file",
    "pair_map_generated"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# Depot Location
st.sidebar.header("Depot Location")
lat_input = st.sidebar.number_input("Latitude", value=52.0, step=0.01, format="%.6f")
lon_input = st.sidebar.number_input("Longitude", value=5.0, step=0.01, format="%.6f")

#  Algorithm Selection
st.sidebar.header("Algorithm Selection")
algorithm_options = ["Bounding Box", "Clustering", "Combined"]
selected_algorithm = st.sidebar.selectbox("Select Algorithm", algorithm_options)

def create_distance_matrix(locations, batch_size, max_workers, base_url, profile):
    """Create the distance matrix for all locations using OSRM."""
    return create_batched_distance_matrix(locations, batch_size, max_workers, base_url, profile)

# Update depot_location in session state
st.session_state["depot_location"] = (lat_input, lon_input)

# Depot location check
if not is_within_netherlands(lat_input, lon_input):
    st.error("Depot location is outside the Netherlands. Please pick valid coordinates.")
    st.stop()

# Bike or car
st.sidebar.header("Routing Mode")
use_bicycle = st.sidebar.checkbox("Use Bicycle Routing", value=False)
profile = "cycling" if use_bicycle else "driving"

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    if st.session_state["uploaded_file"] != uploaded_file:
        # Reset states if a different file is uploaded
        st.session_state["uploaded_data"] = None
        st.session_state["distance_matrix_generated"] = False
        st.session_state["ranked_pairs"] = None
        st.session_state["best_partnerships"] = None
        st.session_state["pair_result_selected"] = None
        st.session_state["selected_pair_comparison"] = None
        st.session_state["pair_result_bb"] = None
        st.session_state["bb_comparison_table"] = None
        st.session_state["all_pairs_result"] = None
        st.session_state["all_pairs_total_cost"] = None
        st.session_state["pair_map_file"] = None
        st.session_state["pair_map_generated"] = False

    if st.session_state["uploaded_data"] is None or st.session_state["uploaded_file"] != uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = {"name", "lat", "lon"}
            if not required_columns.issubset(df.columns):
                st.error(f"CSV must have columns: {required_columns}")
                st.stop()
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["uploaded_data"] = df
            st.session_state["distance_matrix_generated"] = False
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

df = st.session_state.get("uploaded_data", None)

# VRP Parameters
st.sidebar.header("Solver Parameters")
nmbr_loc = st.sidebar.number_input("Maximum number of locations per route", min_value=1, value=4)
cost_per_km = st.sidebar.number_input("Cost per km (€)", min_value=0.0, value=1.0)
cost_per_truck = st.sidebar.number_input("Cost per truck (€)", min_value=0, value=800)
time_per_vrp = st.sidebar.number_input("Time limit per VRP (s)", min_value=0, value=10)

# Distance Matrix Generation
if df is not None and not st.session_state.get("distance_matrix_generated", False):
    try:
        # Prepare location list for OSRM
        locations = []
        for i, row in df.iterrows():
            locations.append({
                "lon": row["lon"],
                "lat": row["lat"],
                "name": row["name"],
                "unique_name": f"{row['name']}_{i}"
            })

        depot_lat, depot_lon = st.session_state["depot_location"]
        locations.append({
            "lon": depot_lon,  
            "lat": depot_lat,
            "name": "Universal Depot",
            "unique_name": "Universal_Depot"
        })

        # # Uncomment for Local OSRM if available:
        # if profile == "driving":
        #     base_url = "http://localhost:5000"
        # elif profile == "cycling":
        #     base_url = "http://localhost:5001"

        #Uncomment below and comment above for using public OSRM:
        base_url = "http://router.project-osrm.org"

        dm = create_distance_matrix(
            locations,
            batch_size=10,
            max_workers=4,
            base_url=base_url,
            profile=profile
        )

        st.session_state["distance_matrix"] = dm
        st.session_state["distance_matrix_generated"] = True

    except Exception as e:
        st.error(f"Error generating distance matrix: {e}")
        st.stop()

# Recompute partnerships if a new heuristic is selected
if df is not None and st.session_state.get("distance_matrix") is not None:
    # We have a valid distance matrix, so always recompute partnerships
    distance_matrix = st.session_state["distance_matrix"]

    if selected_algorithm == "Bounding Box":
        ranked_pairs = rank_company_pairs_by_overlap_percentage(df).reset_index(drop=True)
        st.session_state["ranked_pairs"] = ranked_pairs
        best_partnerships = get_best_partnerships_bb(ranked_pairs)
        best_partnerships = best_partnerships.reset_index(drop=True)
        st.session_state["best_partnerships"] = best_partnerships

    elif selected_algorithm == "Clustering":
        labels = get_clusters_for_file(distance_matrix)
        ranked_pairs = rank_partnerships_using_clusters(df, labels, distance_matrix).reset_index(drop=True)
        st.session_state["ranked_pairs"] = ranked_pairs
        best_partnerships = get_best_partnerships_cluster(ranked_pairs)
        best_partnerships = best_partnerships.reset_index(drop=True)
        st.session_state["best_partnerships"] = best_partnerships

    elif selected_algorithm == "Combined":
        overlap_df = rank_company_pairs_by_overlap_percentage(df)
        cluster_labels = get_clusters_for_file(distance_matrix)
        cluster_df = rank_partnerships_using_clusters(df, cluster_labels, distance_matrix)
        combined_df = combine_heuristics_normalized(overlap_df, cluster_df, 0.5, 0.5).reset_index(drop=True)
        st.session_state["ranked_pairs"] = combined_df
        best_partnerships = get_best_partnerships_combined(combined_df)
        best_partnerships = best_partnerships.reset_index(drop=True)
        st.session_state["best_partnerships"] = best_partnerships

# Show Ranked Partnerships
st.subheader("Ranked Partnerships by Overlap/Score")
if "ranked_pairs" in st.session_state and st.session_state["ranked_pairs"] is not None:
    if not st.session_state["ranked_pairs"].empty:
        st.dataframe(st.session_state["ranked_pairs"].reset_index(drop=True))
    else:
        st.warning("No ranked pairs found.")
else:
    st.info("Upload data.")



# Compute Selected Pair
if (
    df is not None
    and st.session_state.get("distance_matrix_generated", False)
    and "ranked_pairs" in st.session_state
    and st.session_state["ranked_pairs"] is not None
    and not st.session_state["ranked_pairs"].empty
):
    rp_data = st.session_state["ranked_pairs"]
    selected_pair_idx = st.selectbox("Select a pair to solve individually", rp_data.index)

    if st.button("Solve VRP for Selected Pair"):
        try:
            distance_matrix = st.session_state["distance_matrix"]
            row_sel = rp_data.loc[selected_pair_idx]
            c1, c2 = row_sel["Company1"], row_sel["Company2"]

            pair_df = row_sel.to_frame().T
            pair_result = solve_vrp_for_all_pairs_in_dataframe(
                pair_df,
                distance_matrix,
                cost_per_truck,
                cost_per_km,
                time_per_vrp,
                False,
                nmbr_loc
            )
            if pair_result.empty:
                st.error("No VRP solution for this pair.")
                st.stop()

            st.session_state["pair_result_selected"] = pair_result

            # Compare cost vs. solo
            filt = get_filtered_matrix_for_pair(distance_matrix, c1, c2)
            dm1, dm2 = get_individual_company_matrices(c1, c2, filt)
            solo1 = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km, time_per_vrp,
                                           False, nmbr_loc, dm1)
            solo2 = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km, time_per_vrp,
                                           False, nmbr_loc, dm2)

            paired_cost = pair_result["Total Distance"].sum()  # Actually total cost
            solo_cost_1 = solo1.get("Total Distance", 0)
            solo_cost_2 = solo2.get("Total Distance", 0)
            total_solo = solo_cost_1 + solo_cost_2
            savings = total_solo - paired_cost
            savings_pct = (savings / total_solo * 100) if total_solo > 0 else 0

            comp_data = {
                f"{c1} & {c2}": ["Paired Cost (€)", "Solo Cost Combined (€)", "Savings (€)", "Savings (%)"],
                "": [
                    f"{paired_cost:.2f}",
                    f"{total_solo:.2f}",
                    f"{savings:.2f}",
                    f"{savings_pct:.2f}%"
                ]
            }
            st.session_state["selected_pair_comparison"] = comp_data

        except Exception as e:
            st.error(f"Error solving pair VRP: {e}")

# Download for Selected VRP Route
if "selected_pair_comparison" in st.session_state:
    st.write("### Selected Pair Cost Comparison")
    st.table(pd.DataFrame(st.session_state["selected_pair_comparison"]))

    pair_df = st.session_state.get("pair_result_selected", None)
    if pair_df is not None and not pair_df.empty:
        
        c1 = pair_df["Company1"].iloc[0]
        c2 = pair_df["Company2"].iloc[0]

        single_pair_csv = prepare_single_pair_csv(pair_df)
        csv_data = convert_df(single_pair_csv)

        file_name = f"{c1}_{c2}_routes.csv"
        st.download_button(
            label="Download This Pair's Routes as CSV",
            data=csv_data,
            file_name=file_name,
            mime="text/csv"
        )

# Button to Create Map for the Selected Pair
if st.button("Create Map for Selected Pair"):
    pair_df = st.session_state.get("pair_result_selected", None)
    if pair_df is None or pair_df.empty:
        st.warning("No pair VRP result found. Solve VRP for a pair first.")
    else:
        try:
            label_to_coords = {}
            for i, row in df.iterrows():
                # labelling
                label = f"{row['name']}_{i}"
                label_to_coords[label] = (row["lat"], row["lon"])

            # Include the depot with a unique label
            label_to_coords["Universal_Depot"] = st.session_state["depot_location"]

            depot_loc = st.session_state["depot_location"]
            data_original = st.session_state["uploaded_data"]
            route_map, map_file = generate_route_map_fixed_with_legend(
                depot_loc,
                data_original,
                pair_df,
                label_to_coords,
                osrm_url="http://router.project-osrm.org",
                profile=profile
            )
            st.session_state["pair_map_file"] = map_file
            st.session_state["pair_map_generated"] = True
            st.success("Pair route map created successfully.")
        except Exception as e:
            st.error(f"Error generating pair map: {e}")

if "pair_map_generated" in st.session_state and st.session_state["pair_map_generated"]:
    if "pair_map_file" in st.session_state:
        try:
            with open(st.session_state["pair_map_file"], "r", encoding="utf-8") as f:
                map_html = f.read()
            components.html(map_html, height=700, width=1100)
        except Exception:
            st.warning("Could not render pair map inline. Please download instead.")

        with open(st.session_state["pair_map_file"], "rb") as f:
            st.download_button(
                label="Download Pair Map HTML",
                data=f,
                file_name="pair_routes_map.html",
                mime="text/html"
            )


# Potential Best Partnerships
st.subheader("Potentially Best Partnerships")
best_partnerships = st.session_state.get("best_partnerships", None)

if best_partnerships is not None and not best_partnerships.empty:
    # Determine which columns to display based on the selected algorithm
    if selected_algorithm == "Bounding Box":
        if "Overlap Percentage" in best_partnerships.columns:
            display_df = best_partnerships.drop(columns=["Overlap Percentage"])
        else:
            display_df = best_partnerships.copy()
    elif selected_algorithm == "Clustering":
        if "Clustering Score" in best_partnerships.columns:
            display_df = best_partnerships.drop(columns=["Clustering Score"])
        else:
            display_df = best_partnerships.copy()
    elif selected_algorithm == "Combined":
        # For combined, decide which columns to drop or keep
        columns_to_drop = [col for col in ["Clustering Score", "Overlap Percentage"] if col in best_partnerships.columns]
        display_df = best_partnerships.drop(columns=columns_to_drop)

    st.table(display_df.reset_index(drop=True))
else:
    st.warning("No potential best partnerships found.")

if df is not None and st.session_state.get("distance_matrix_generated", False):
    if st.button("Compute VRP for Potentially Best Partnerships"):
        try:
            distance_matrix = st.session_state["distance_matrix"]
            best_partnerships = st.session_state.get("best_partnerships", None)

            if best_partnerships is not None and not best_partnerships.empty:
                pair_res_bb = solve_vrp_for_all_pairs_in_dataframe(
                    best_partnerships,
                    distance_matrix,
                    cost_per_truck,
                    cost_per_km,
                    time_per_vrp,
                    False,
                    nmbr_loc
                )

                if not pair_res_bb.empty:
                    st.session_state["pair_result_bb"] = pair_res_bb

                    # Build comparison table for each row
                    comp_rows = []
                    for idx, row in pair_res_bb.iterrows():
                        c1 = row["Company1"]
                        c2 = row["Company2"]
                        paired_cost = row["Total Distance"]
                        filt = get_filtered_matrix_for_pair(distance_matrix, c1, c2)
                        dm1, dm2 = get_individual_company_matrices(c1, c2, filt)
                        solo1 = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km,
                                                       time_per_vrp, False, nmbr_loc, dm1)
                        solo2 = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km,
                                                       time_per_vrp, False, nmbr_loc, dm2)
                        cost1 = solo1.get("Total Distance", 0)
                        cost2 = solo2.get("Total Distance", 0)
                        total_solo = cost1 + cost2
                        savings = total_solo - paired_cost
                        savings_pct = (savings / total_solo * 100) if total_solo > 0 else 0

                        comp_rows.append({
                            "Company1": c1,
                            "Company2": c2,
                            "Paired Cost (€)": paired_cost,
                            "Solo Cost (C1)": cost1,
                            "Solo Cost (C2)": cost2,
                            "Total Solo (€)": total_solo,
                            "Savings (€)": savings,
                            "Savings (%)": savings_pct
                        })

                    bb_comp_df = pd.DataFrame(comp_rows)
                    st.session_state["bb_comparison_table"] = bb_comp_df
                else:
                    st.warning("No pair VRP solutions found.")
            else:
                st.warning("No best partnerships available.")
        except Exception as e:
            st.error(f"Error computing VRP: {e}")

if "bb_comparison_table" in st.session_state and st.session_state["bb_comparison_table"] is not None:
    st.write("### Heuristic VRP Comparison Table")
    df_bb_comp = st.session_state["bb_comparison_table"]
    st.dataframe(
        df_bb_comp.style.format({
            "Paired Cost (€)": "{:.2f}",
            "Solo Cost (C1)": "{:.2f}",
            "Solo Cost (C2)": "{:.2f}",
            "Total Solo (€)": "{:.2f}",
            "Savings (€)": "{:.2f}",
            "Savings (%)": "{:.2f}"
        })
    )

if "pair_result_bb" in st.session_state and st.session_state["pair_result_bb"] is not None:
    if not st.session_state["pair_result_bb"].empty:
        all_pairs_csv = prepare_pairs_vrp_csv(st.session_state["pair_result_bb"])
        csv_data = convert_df(all_pairs_csv)
        st.download_button(
            label="Download all Partnerships Routes CSV",
            data=csv_data,
            file_name="all_pairs_routes.csv",
            mime="text/csv"
        )

# Compute Solo Routes and Global Matching
st.subheader("Compare Solo Routes")

if st.button("Compute SOLO VRP"):
    try:
        distance_matrix = st.session_state["distance_matrix"]
        solo_df = solo_routes(
            cost_per_truck,
            cost_per_km,
            time_per_vrp,
            False,
            nmbr_loc,
            distance_matrix
        )
        if not solo_df.empty:
            st.session_state["vrp_result_solo"] = solo_df
        else:
            st.warning("No solo VRP solution found.")
    except Exception as e:
        st.error(f"Error computing SOLO VRP: {e}")


#uncomment for best possible result by solving all vrps option in dashboard
# if st.button("Compute Global Matching"):
#     distance_matrix = st.session_state["distance_matrix"]
#     try:
#         global_pairs_df, total_global_cost = solve_vrp_for_all_possible_pairs(
#             distance_matrix,
#             cost_per_truck,
#             cost_per_km,
#             time_per_vrp,
#             False,
#             nmbr_loc
#         )
#         st.session_state["all_pairs_result"] = global_pairs_df
#         st.session_state["all_pairs_total_cost"] = total_global_cost
#         st.success("Global minimal pairing found (all companies).")

#         st.write("### Matching Pairs")
#         st.dataframe(global_pairs_df)
#         st.write(f"**Total Cost**: €{total_global_cost:,.2f}")
#     except Exception as e:
#         st.error(f"Error solving global pairing: {e}")

# Cost Comparison
def get_total_cost_from_pairs(df: pd.DataFrame):
    """Sum the 'Total Distance' column (which actually represents total cost)."""
    if df is not None and not df.empty:
        return df["Total Distance"].sum()
    return None

solo_df = st.session_state.get("vrp_result_solo", None)
pairs_bb_df = st.session_state.get("pair_result_bb", None)
all_pairs_df = st.session_state.get("all_pairs_result", None)
all_pairs_cost = st.session_state.get("all_pairs_total_cost", None)

total_solo_cost = solo_df["Total Distance"].sum() if solo_df is not None and not solo_df.empty else None
total_bb_cost = get_total_cost_from_pairs(pairs_bb_df)
total_global_pairs = all_pairs_cost

comparison_data = {
    "Solution Type": [],
    "Total Cost (€)": []
}

if total_solo_cost is not None:
    comparison_data["Solution Type"].append("Solo Routes")
    comparison_data["Total Cost (€)"].append(total_solo_cost)

if total_bb_cost is not None:
    comparison_data["Solution Type"].append("Heuristic pairs")
    comparison_data["Total Cost (€)"].append(total_bb_cost)

if total_global_pairs is not None:
    comparison_data["Solution Type"].append("Global Perfect Matching (All in Pairs)")
    comparison_data["Total Cost (€)"].append(total_global_pairs)

if comparison_data["Solution Type"]:
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.style.format({"Total Cost (€)": "{:.2f}"}))

    if len(comparison_data["Solution Type"]) >= 2:
        min_cost_idx = comparison_df["Total Cost (€)"].idxmin()
        best_sol = comparison_df.loc[min_cost_idx, "Solution Type"]
        best_val = comparison_df.loc[min_cost_idx, "Total Cost (€)"]
        st.success(f"Best found solution: {best_sol} at €{best_val:.2f}")
else:
    st.info("Compute VRP solutions (solo, heuristic, global) to see final comparison.")

if "vrp_result_solo" in st.session_state and st.session_state["vrp_result_solo"] is not None:

    solo_csv_data = convert_df(st.session_state["vrp_result_solo"])
    st.download_button(
        label="Download SOLO VRP routes",
        data=solo_csv_data,
        file_name="solo_vrp.csv",
        mime="text/csv"
    )

st.write("---")
