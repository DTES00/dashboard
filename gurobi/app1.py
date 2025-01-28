# app.py

import itertools
import random
import ast

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import folium
from folium import Marker, PolyLine
from folium.features import DivIcon
from folium.map import LayerControl
from folium.plugins import FloatImage
from streamlit_folium import st_folium



import networkx as nx

# --- VRP + Overlap Logic ---
from bounding_box import get_best_partnerships as get_best_partnerships_bb
from bounding_box import rank_company_pairs_by_overlap_percentage

# --- VRP Solver Logic ---
from solver1 import (
    create_batched_distance_matrix,
    get_filtered_matrix_for_pair,
    get_individual_company_matrices,
    solo_routes,
    solve_cvrp_numeric_ids,
    solve_vrp_for_all_pairs_in_dataframe,
    solve_vrp_for_all_possible_pairs
)

###############################################################################
# GLOBALS & HELPERS
###############################################################################
NETHERLANDS_BOUNDS = {
    "min_lat": 0,
    "max_lat": 100,
    "min_lon": 0,
    "max_lon": 100.22,
}

def is_within_netherlands(lat, lon):
    """Rough bounding box check."""
    return (
        NETHERLANDS_BOUNDS["min_lat"] <= lat <= NETHERLANDS_BOUNDS["max_lat"]
        and NETHERLANDS_BOUNDS["min_lon"] <= lon <= NETHERLANDS_BOUNDS["max_lon"]
    )

def convert_df(df):
    """DataFrame -> CSV bytes."""
    return df.to_csv(index=False).encode('utf-8')

def rename_source_sink(route_stops):
    """'Source'/'Sink' -> 'Universal_Depot' in route stops."""
    return ["Universal_Depot" if stop in ("Source", "Sink") else stop for stop in route_stops]


def prepare_single_pair_csv(pair_df):
    """
    Convert a single-pair VRP result into row-per-vehicle. Columns: Company1, Company2, Routes, ...
    """
    rows = []
    for idx, row in pair_df.iterrows():
        c1 = row.get("Company1", f"Company1_{idx}")
        c2 = row.get("Company2", f"Company2_{idx}")
        total_cost = row.get("Total Distance", 0)
        routes_dict = row.get("Routes", {})
        if isinstance(routes_dict, str):
            import ast
            routes_dict = ast.literal_eval(routes_dict)
        if routes_dict is None:
            continue

        for vehicle_id, stops_list in routes_dict.items():
            cleaned = rename_source_sink(stops_list)
            route_str = " -> ".join(map(str, cleaned))
            rows.append({
                "Company1": c1,
                "Company2": c2,
                "Vehicle ID": vehicle_id,
                "Route": route_str,
                "Total Cost": total_cost
            })
    return pd.DataFrame(rows)


def prepare_pairs_vrp_csv(pair_df):
    """
    For multi-row bounding-box results, produce a single CSV with routes for each row.
    """
    all_rows = []
    for idx, row in pair_df.iterrows():
        c1 = row.get("Company1", f"Company1_{idx}")
        c2 = row.get("Company2", f"Company2_{idx}")
        total_cost = row.get("Total Distance", 0)
        routes_dict = row.get("Routes", {})
        if isinstance(routes_dict, str):
            import ast
            routes_dict = ast.literal_eval(routes_dict)
        if routes_dict is None:
            continue

        for vehicle_id, stops_list in routes_dict.items():
            cleaned = rename_source_sink(stops_list)
            route_str = " -> ".join(map(str, cleaned))
            all_rows.append({
                "Company1": c1,
                "Company2": c2,
                "Vehicle ID": vehicle_id,
                "Route": route_str,
                "Total Cost": total_cost
            })
    return pd.DataFrame(all_rows)

###############################################################################
# MAP FUNCTION (Numbered Route Stops)
###############################################################################
import folium
from folium import Marker, PolyLine
from folium.features import DivIcon
from folium.map import LayerControl

import folium
from folium.features import DivIcon
from streamlit_folium import st_folium
import itertools
import requests



def generate_route_map(
    depot_loc,
    data,
    routes_df,
    osrm_url="http://router.project-osrm.org",
    profile="driving"
):
    """
    Generates a Folium map visualizing the VRP routes for a selected pair.

    Parameters:
    - depot_loc (tuple): (latitude, longitude) of the depot.
    - data (pd.DataFrame): DataFrame containing 'name', 'lat', 'lon' of all stops.
    - routes_df (pd.DataFrame): DataFrame containing VRP solutions with 'Vehicle ID' and 'Route' columns.
    - osrm_url (str): Base URL for the OSRM server.
    - profile (str): Routing profile ('driving', 'cycling', etc.).

    Returns:
    - folium.Map: Folium map object with the plotted routes.
    """
    # Initialize Folium map centered around the depot
    map_center = depot_loc
    folium_map = folium.Map(location=map_center, zoom_start=10)

    # Add depot marker
    folium.Marker(
        location=depot_loc,
        popup="Depot",
        icon=folium.Icon(color="red", icon="home", prefix="fa")
    ).add_to(folium_map)

    # Define a color cycle for different vehicles
    color_cycle = itertools.cycle([
        "blue", "green", "purple", "orange", "darkred", "lightred",
        "beige", "darkblue", "darkgreen", "cadetblue", "pink",
        "lightblue", "lightgreen", "gray", "black"
    ])

    # Process each vehicle's route
    for idx, row in routes_df.iterrows():
        vehicle_id = row.get("Vehicle ID", f"Vehicle_{idx}")
        route_stops = row.get("Route", [])

        if not route_stops:
            continue  # Skip if no route data

        # Ensure 'Route' is a list
        if isinstance(route_stops, str):
            stops = route_stops.split(" -> ")
        elif isinstance(route_stops, list):
            stops = route_stops
        else:
            st.warning(f"Unexpected Route format for {vehicle_id}. Skipping.")
            continue

        stop_coords = []

        for stop in stops:
            if stop.lower() == "depot":
                stop_coords.append(depot_loc)
            else:
                # Find the stop in the data DataFrame
                stop_data = data[data['name'] == stop]
                if not stop_data.empty:
                    lat = stop_data.iloc[0]['lat']
                    lon = stop_data.iloc[0]['lon']
                    stop_coords.append((lat, lon))
                else:
                    # Handle missing stop data
                    st.warning(f"Stop '{stop}' not found in data. Skipping.")

        # Ensure the route starts and ends at the depot
        if stop_coords and stop_coords[0] != depot_loc:
            stop_coords.insert(0, depot_loc)
        if stop_coords and stop_coords[-1] != depot_loc:
            stop_coords.append(depot_loc)

        # Fetch the full route geometry from OSRM in one request
        if len(stop_coords) < 2:
            continue  # Need at least two points to form a route

        coords = ";".join([f"{lon},{lat}" for lat, lon in stop_coords])
        url = f"{osrm_url}/route/v1/{profile}/{coords}?overview=full&geometries=geojson"

        try:
            response = requests.get(url)
            response.raise_for_status()
            route_data = response.json()

            if "routes" in route_data and len(route_data["routes"]) > 0:
                geometry = route_data["routes"][0]["geometry"]["coordinates"]
                # Convert to (lat, lon) tuples
                route_path = [(lat, lon) for lon, lat in geometry]

                # Add the route to the map
                folium.PolyLine(
                    locations=route_path,
                    color=next(color_cycle),
                    weight=5,
                    opacity=0.7,
                    popup=f"{vehicle_id}"
                ).add_to(folium_map)
            else:
                st.warning(f"No route found for {vehicle_id}.")
        except requests.exceptions.RequestException as e:
            st.error(f"OSRM request failed for {vehicle_id}: {e}")
            continue

        # Add numbered markers for stops
        for order, (lat, lon) in enumerate(stop_coords, start=1):
            folium.Marker(
                location=(lat, lon),
                icon=DivIcon(
                    icon_size=(30, 30),
                    icon_anchor=(15, 15),
                    html=f"""
                        <div style="
                            font-size: 12pt;
                            color: black;
                            background-color: rgba(255, 255, 255, 0.7);
                            border: 1px solid black;
                            border-radius: 15px;
                            width: 30px;
                            height: 30px;
                            text-align: center;
                            line-height: 30px;">
                            {order}
                        </div>
                    """
                ),
                popup=f"Stop {order}: {stops[order-1]}"
            ).add_to(folium_map)

    # Add layer control
    folium.LayerControl().add_to(folium_map)

    return folium_map


###############################################################################
# STREAMLIT APP
###############################################################################


# Step 1: Depot
st.sidebar.header("Depot Location")
lon_input = st.sidebar.number_input("Latitude", value=52.0, step=0.01, format="%.6f")
lat_input = st.sidebar.number_input("Longitude", value=5.0, step=0.01, format="%.6f")

if "depot_location" not in st.session_state:
    st.session_state["depot_location"] = None
st.session_state["depot_location"] = (lat_input, lon_input)

if not is_within_netherlands(lat_input, lon_input):
    st.error("Depot location is outside the Netherlands. Please pick valid coords.")
    st.stop()



# Step 2: Routing Mode
st.sidebar.header("Routing Mode")
use_bicycle = st.sidebar.checkbox("Use Bicycle Routing", value=False)
profile = "cycling" if use_bicycle else "driving"

# Step 3: Upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    if "uploaded_file" in st.session_state and st.session_state["uploaded_file"] != uploaded_file:
        st.session_state["uploaded_data"] = None
        st.session_state["distance_matrix_generated"] = False
        st.session_state["ranked_pairs"] = None
        st.session_state["pair_result_selected"] = None
        st.session_state["vrp_result_solo"] = None
        st.session_state["pair_result_bb"] = None
        st.session_state["bb_comparison_table"] = None
        st.session_state["all_pairs_result"] = None
        st.session_state["all_pairs_total_cost"] = None

    if "uploaded_data" not in st.session_state or st.session_state.get("uploaded_file") != uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            req_cols = {"name","lat","lon"}
            if not req_cols.issubset(df.columns):
                st.error(f"CSV must have columns {req_cols}")
                st.stop()
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["uploaded_data"] = df
            st.session_state["distance_matrix_generated"] = False
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

df = st.session_state.get("uploaded_data", None)

# Step 4: VRP Parameters
st.sidebar.header("Solver Params")
nmbr_loc = st.sidebar.number_input("Max # locations/route", min_value=1, value=4)
cost_per_km = st.sidebar.number_input("Cost per km (€)", min_value=0, value=1)
cost_per_truck = st.sidebar.number_input("Cost per truck (€)", min_value=0, value=800)
time_per_vrp = st.sidebar.number_input("Time limit per VRP (s)", min_value=0, value=10)

# Step 5: Generate Distance Matrix
if df is not None and not st.session_state.get("distance_matrix_generated", False):
    try:
        # Prepare
        locations = []
        for i, row in df.iterrows():
            locations.append({
                "lon": row["lon"],
                "lat": row["lat"],
                "name": row["name"],
                "unique_name": f"{row['name']}_{i}"
            })
        depot_lon, depot_lat = st.session_state["depot_location"]
        locations.append({
            "lon": depot_lon,
            "lat": depot_lat,
            "name": "Universal Depot",
            "unique_name": "Universal_Depot"
        })



        # Example: Inspecting the coordinates
        for loc in locations:
            if loc['unique_name'] == "Universal_Depot":
                print(f"Depot Coordinates: Latitude = {loc['lat']}, Longitude = {loc['lon']}")

        # Create matrix
        dm = create_batched_distance_matrix(
            locations,
            batch_size=100,
            max_workers=4,
            base_url="http://router.project-osrm.org",
            profile=profile
        )
        st.session_state["distance_matrix"] = dm
        st.session_state["distance_matrix_generated"] = True
        st.session_state["distance_matrix"] = (st.session_state["distance_matrix"] / 1000).round(2)

        # bounding-box logic
        ranked_pairs = rank_company_pairs_by_overlap_percentage(df)
        st.session_state["ranked_pairs"] = ranked_pairs
        best_bb = get_best_partnerships_bb(ranked_pairs)
        st.session_state["best_partnerships_bb"] = best_bb

        st.success("Distance matrix + bounding-box pairs generated.")
    except Exception as e:
        st.error(f"Error generating matrix: {e}")
        st.stop()



# Show Partnerships
st.subheader("Ranked Partnerships by Overlap")
if "ranked_pairs" in st.session_state and st.session_state["ranked_pairs"] is not None:
    if not st.session_state["ranked_pairs"].empty:
        st.dataframe(st.session_state["ranked_pairs"].reset_index(drop=True))
    else:
        st.warning("No ranked pairs found.")
else:
    st.info("Upload data + generate the distance matrix first.")

st.subheader("Potential Best Partnerships (Bounding-Box)")
best_partnerships_bb = st.session_state.get("best_partnerships_bb", None)
if best_partnerships_bb is not None and not best_partnerships_bb.empty:
    st.table(best_partnerships_bb.drop(columns=["Overlap Percentage"]).reset_index(drop=True))
else:
    st.warning("No best bounding-box partnerships found.")


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

            solo1 = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km, time_per_vrp, False, nmbr_loc, dm1)
            solo2 = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km, time_per_vrp, False, nmbr_loc, dm2)

            paired_cost = pair_result["Total Distance"].sum()
            solo_cost_1 = solo1.get("Total Distance", 0)
            solo_cost_2 = solo2.get("Total Distance", 0)
            total_solo = solo_cost_1 + solo_cost_2
            savings = total_solo - paired_cost
            savings_pct = (savings / total_solo * 100) if total_solo > 0 else 0

            comp_data = {
                "Metric": ["Paired Cost (€)", "Solo Cost Combined (€)", "Savings (€)", "Savings (%)"],
                "Value": [
                    f"{paired_cost:.2f}",
                    f"{total_solo:.2f}",
                    f"{savings:.2f}",
                    f"{savings_pct:.2f}%"
                ]
            }
            st.session_state["selected_pair_comparison"] = comp_data

        except Exception as e:
            st.error(f"Error solving pair VRP: {e}")

if "selected_pair_comparison" in st.session_state:
    st.write("### Selected Pair Cost Comparison")
    st.table(pd.DataFrame(st.session_state["selected_pair_comparison"]))

    # Download single pair CSV
    pair_df = st.session_state.get("pair_result_selected", None)
    if pair_df is not None and not pair_df.empty:
        single_pair_csv = prepare_single_pair_csv(pair_df)
        csv_data = convert_df(single_pair_csv)
        st.download_button(
            label="Download This Pair's Routes as CSV",
            data=csv_data,
            file_name="selected_pair_routes.csv",
            mime="text/csv"
        )

from plot_routes import generate_route_map_fixed_with_legend 
# Map for single pair
# Map for single pair
if st.button("Create Map for Selected Pair"):
    pair_df = st.session_state.get("pair_result_selected", None)
    if pair_df is None or pair_df.empty:
        st.warning("No pair VRP result found. Solve VRP for a pair first.")
    else:
        try:
            # Preprocess Routes column by handling dicts or strings
            company_routes = []
            for _, row in pair_df.iterrows():
                # Check if 'Routes' is a string; if so, parse it. Otherwise, use it directly.
                if isinstance(row["Routes"], str):
                    try:
                        routes = ast.literal_eval(row["Routes"])  # Convert JSON string to dictionary
                    except (ValueError, SyntaxError) as e:
                        st.error(f"Error parsing Routes for Company {row['Company1']}: {e}")
                        continue
                elif isinstance(row["Routes"], dict):
                    routes = row["Routes"]
                else:
                    st.error(f"Unexpected Routes format for Company {row['Company1']}.")
                    continue

                # Ensure routes is a dictionary
                if not isinstance(routes, dict):
                    st.error(f"Routes should be a dictionary for Company {row['Company1']}.")
                    continue

                for vehicle_id, stops in routes.items():
                    company_routes.append({
                        "Company": row["Company1"],
                        "Vehicle": vehicle_id,
                        "Stops": stops
                    })

            # Create a DataFrame with parsed routes
            routes_df = pd.DataFrame(company_routes)

            # Pass to the map generation function
            depot_lat, depot_lon = st.session_state["depot_location"]
            data_original = st.session_state["uploaded_data"]
            route_map, map_file = generate_route_map_fixed_with_legend(
                (depot_lat, depot_lon),
                data_original,
                routes_df,  # Pass the parsed routes DataFrame
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
        except Exception as e:
            st.warning(f"Could not render pair map inline: {e}. Please download instead.")

        with open(st.session_state["pair_map_file"], "rb") as f:
            st.download_button(
                label="Download Pair Map HTML",
                data=f,
                file_name="pair_routes_map.html",
                mime="text/html"
            )



###############################################################################
# COMPUTE VRP FOR ALL BEST PARTNERSHIPS
###############################################################################
st.subheader("Solve VRP for Best Partnerships")

if df is not None and st.session_state.get("distance_matrix_generated", False):
    if st.button("Compute VRP for Best Partnerships"):
        try:
            distance_matrix = st.session_state["distance_matrix"]
            best_bb_df = st.session_state.get("best_partnerships_bb", None)
            if best_bb_df is not None and not best_bb_df.empty:
                pair_res_bb = solve_vrp_for_all_pairs_in_dataframe(
                    best_bb_df,
                    distance_matrix,
                    cost_per_truck,
                    cost_per_km,
                    time_per_vrp,
                    False,
                    nmbr_loc
                )
                if not pair_res_bb.empty:
                    st.session_state["pair_result_bb"] = pair_res_bb
                    
                    # Now build the comparison table for each row
                    comp_rows = []
                    for idx, row in pair_res_bb.iterrows():
                        c1 = row["Company1"]
                        c2 = row["Company2"]
                        paired_cost = row["Total Distance"]

                        # Solve individually
                        # filter the matrix
                        filt = get_filtered_matrix_for_pair(distance_matrix, c1, c2)
                        dm1, dm2 = get_individual_company_matrices(c1, c2, filt)
                        solo1 = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km, time_per_vrp, False, nmbr_loc, dm1)
                        solo2 = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km, time_per_vrp, False, nmbr_loc, dm2)
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
                    st.warning("No bounding-box pair VRP solutions found.")
            else:
                st.warning("No best partnerships available.")
        except Exception as e:
            st.error(f"Error computing bounding-box VRP: {e}")

# Show bounding-box routes CSV
if "pair_result_bb" in st.session_state and st.session_state["pair_result_bb"] is not None:
    if not st.session_state["pair_result_bb"].empty:
        st.write("### Download Full Bounding-Box Partnerships VRP Routes as CSV")
        all_pairs_csv = prepare_pairs_vrp_csv(st.session_state["pair_result_bb"])
        csv_data = convert_df(all_pairs_csv)
        st.download_button(
            label="Download Bounding-Box Partnerships Routes CSV",
            data=csv_data,
            file_name="bounding_box_all_pairs_routes.csv",
            mime="text/csv"
        )

# Show bounding-box comparison table
if "bb_comparison_table" in st.session_state and st.session_state["bb_comparison_table"] is not None:
    st.write("### Bounding-Box VRP Comparison Table")
    df_bb_comp = st.session_state["bb_comparison_table"]
    st.dataframe(df_bb_comp.style.format({"Paired Cost (€)": "{:.2f}",
                                          "Solo Cost (C1)": "{:.2f}",
                                          "Solo Cost (C2)": "{:.2f}",
                                          "Total Solo (€)": "{:.2f}",
                                          "Savings (€)": "{:.2f}",
                                          "Savings (%)": "{:.2f}"}))


###############################################################################
# SOLO + GLOBAL MATCHING
###############################################################################
st.subheader("Compute SOLO VRP & Global Perfect Matching")

colA, colB = st.columns(2)

with colA:
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
                st.success("Individual (solo) VRP solution computed.")
            else:
                st.warning("No solo VRP solution found.")
        except Exception as e:
            st.error(f"Error computing SOLO VRP: {e}")

with colB:
    if st.button("Compute Global Matching"):
        distance_matrix = st.session_state["distance_matrix"]
        try:
            global_pairs_df, total_global_cost = solve_vrp_for_all_possible_pairs(
                distance_matrix,
                cost_per_truck,
                cost_per_km,
                time_per_vrp,
                False,
                nmbr_loc
            )
            st.session_state["all_pairs_result"] = global_pairs_df
            st.session_state["all_pairs_total_cost"] = total_global_cost
            st.success("Global minimal pairing found (all companies).")

            st.write("### Matching Pairs")
            st.dataframe(global_pairs_df)
            st.write(f"**Total Cost**: €{total_global_cost:,.2f}")
        except Exception as e:
            st.error(f"Error solving global pairing: {e}")

# Download SOLO results
if "vrp_result_solo" in st.session_state and st.session_state["vrp_result_solo"] is not None:
    st.write("### Download SOLO VRP CSV")
    solo_csv_data = convert_df(st.session_state["vrp_result_solo"])
    st.download_button(
        label="Download SOLO VRP",
        data=solo_csv_data,
        file_name="solo_vrp.csv",
        mime="text/csv"
    )

###############################################################################
# FINAL COST COMPARISON
###############################################################################


def get_total_cost_from_pairs(df):
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
    comparison_data["Solution Type"].append("Bounding-Box Pairs")
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
        st.success(f"Optimal solution so far: {best_sol} at €{best_val:.2f}")
else:
    st.info("Compute VRP solutions (solo, bounding-box, global) to see final comparison.")

st.write("---")
st.write("End of App.")
