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

# Cluster
from cluster import get_best_partnerships as get_best_partnerships_cluster
from cluster import rank_partnerships_using_clusters, get_clusters_for_file

# --- VRP Solver Logic ---
from solver1 import (
    create_batched_distance_matrix,

    get_filtered_matrix_for_pair,
    get_individual_company_matrices,
    solo_routes,
    solve_cvrp_numeric_ids,
    solve_vrp_for_all_pairs_in_dataframe,
    solve_vrp_for_all_possible_pairs,
    solve_vrp_for_all_pairs  # Newly added function
)

###############################################################################
# GLOBALS & HELPERS
###############################################################################
NETHERLANDS_BOUNDS = {
    "min_lat": 50.5,
    "max_lat": 53.7,
    "min_lon": 3.36,
    "max_lon": 7.22,
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
    """'Source'/'Sink' -> 'Depot' in route stops."""
    return ["Depot" if stop in ("Source", "Sink") else stop for stop in route_stops]

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
    For multi-row bounding-box or clustering results, produce a single CSV with routes for each row.
    """
    all_rows = []
    for idx, row in pair_df.iterrows():
        c1 = row.get("Company1", f"Company1_{idx}")
        c2 = row.get("Company2", f"Company2_{idx}")
        total_cost = row.get("Total Distance", 0)
        routes_dict = row.get("Routes", {})
        if isinstance(routes_dict, str):
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
def generate_route_map_fixed_with_legend(
    depot_loc,
    data,
    test,
    osrm_url="http://router.project-osrm.org",
    profile="driving"
):
    # Center on avg lat/lon
    center_lat = data["lat"].mean()
    center_lon = data["lon"].mean()
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=8)

    if "Company" not in test.columns:
        test["Company"] = "NoCompanyName"

    for _, row in test.iterrows():
        company_name = row["Company"]
        routes_dict = row.get("Routes", {})
        fg = folium.FeatureGroup(name=f"{company_name} Routes", show=True)

        if depot_loc:
            Marker(
                location=depot_loc,
                popup=f"{company_name} - Depot",
                icon=folium.Icon(color="red", icon="home", prefix="fa")
            ).add_to(fg)

        route_colors = itertools.cycle([
            "blue","green","purple","orange","darkred","lightred","beige","darkblue",
            "darkgreen","cadetblue","pink","lightblue","lightgreen","gray","black"
        ])

        for vehicle_id, stops_list in routes_dict.items():
            route_color = next(route_colors)
            numeric_stops = [s for s in stops_list if isinstance(s, int)]
            route_coords = []
            for stop_idx in numeric_stops:
                if 0 <= stop_idx < len(data):
                    row_data = data.iloc[stop_idx]
                    route_coords.append((row_data["lat"], row_data["lon"]))

            if depot_loc and route_coords:
                route_coords = [depot_loc] + route_coords + [depot_loc]

            real_path = []
            for i in range(len(route_coords) - 1):
                start_lat, start_lon = route_coords[i]
                end_lat, end_lon     = route_coords[i+1]
                coords_str = f"{start_lon},{start_lat};{end_lon},{end_lat}"
                url = f"{osrm_url}/route/v1/{profile}/{coords_str}?geometries=geojson&steps=true"
                try:
                    resp = requests.get(url)
                    resp.raise_for_status()
                    osrm_data = resp.json()
                    if osrm_data.get("routes"):
                        geometry = osrm_data["routes"][0]["geometry"]["coordinates"]
                        for (lon_, lat_) in geometry:
                            real_path.append((lat_, lon_))
                except Exception as e:
                    print(f"OSRM error {start_lat},{start_lon} -> {end_lat},{end_lon}: {e}")
                    continue

            if real_path:
                PolyLine(locations=real_path, color=route_color, weight=3, opacity=0.5).add_to(fg)

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

###############################################################################
# STREAMLIT APP
###############################################################################

st.set_page_config(page_title="VRP Partnership Optimizer", layout="wide")

st.title("VRP Partnership Optimizer")

# Step 1: Depot Location
st.sidebar.header("Depot Location")
lat_input = st.sidebar.number_input("Latitude", value=52.0, step=0.01, format="%.6f")
lon_input = st.sidebar.number_input("Longitude", value=5.0, step=0.01, format="%.6f")

# Step 2: Routing Mode
st.sidebar.header("Routing Mode")
use_bicycle = st.sidebar.checkbox("Use Bicycle Routing", value=False)
profile = "cycling" if use_bicycle else "driving"

# Step 3: Upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    if "uploaded_file" in st.session_state and st.session_state["uploaded_file"] != uploaded_file:
        # Reset session state variables if a new file is uploaded
        st.session_state["uploaded_data"] = None
        st.session_state["distance_matrix_generated"] = False
        st.session_state["ranked_pairs_bb"] = None
        st.session_state["ranked_pairs_clust"] = None
        st.session_state["best_partnerships_bb"] = None
        st.session_state["best_partnerships_clust"] = None
        st.session_state["pair_result_selected"] = None
        st.session_state["vrp_result_solo"] = None
        st.session_state["pair_result_bb"] = None
        st.session_state["pair_result_clust"] = None
        st.session_state["bb_comparison_table"] = None
        st.session_state["clust_comparison_table"] = None
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
            st.success("File uploaded successfully.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

df = st.session_state.get("uploaded_data", None)

# Step 4: VRP Parameters
st.sidebar.header("Solver Parameters")
nmbr_loc = st.sidebar.number_input("Max # Locations/Route", min_value=1, value=4, step=1)
cost_per_km = st.sidebar.number_input("Cost per km (€)", min_value=0.0, value=1.0, step=0.1)
cost_per_truck = st.sidebar.number_input("Cost per Truck (€)", min_value=0.0, value=800.0, step=50.0)
time_per_vrp = st.sidebar.number_input("Time Limit per VRP (s)", min_value=1, value=10, step=1)

# Step 5: Generate Distance Matrix
if df is not None and not st.session_state.get("distance_matrix_generated", False):
    try:
        # Validate depot location
        depot_lat, depot_lon = lat_input, lon_input
        if not is_within_netherlands(depot_lat, depot_lon):
            st.error("Depot location is outside the Netherlands. Please pick valid coordinates.")
            st.stop()

        # Prepare locations
        locations = []
        for i, row in df.iterrows():
            locations.append({
                "lon": row["lon"],
                "lat": row["lat"],
                "name": row["name"],
                "unique_name": f"{row['name']}_{i}"
            })
        locations.append({
            "lon": depot_lon,
            "lat": depot_lat,
            "name": "Universal Depot",
            "unique_name": "Universal_Depot"
        })

        # Verify dimensions
        assert len(locations) == len(df) + 1, "Mismatch in locations and dataframe dimensions"

        with st.spinner("Generating distance matrix..."):
            # Create distance matrix
            dm = create_batched_distance_matrix(
                locations,
                batch_size=100,
                max_workers=4,
                base_url="http://router.project-osrm.org",
                profile=profile
            )

        if dm is None or dm.isnull().all().all():
            st.error("Failed to generate a valid distance matrix. Please check your data and try again.")
            st.stop()

        st.session_state["distance_matrix"] = dm
        st.session_state["distance_matrix_generated"] = True

        # Compute Bounding Box Partnerships
        with st.spinner("Computing Bounding Box partnerships..."):
            ranked_pairs_bb = rank_company_pairs_by_overlap_percentage(df)
            st.session_state["ranked_pairs_bb"] = ranked_pairs_bb
            best_bb = get_best_partnerships_bb(ranked_pairs_bb)
            st.session_state["best_partnerships_bb"] = best_bb

        # Compute Clustering Partnerships
        with st.spinner("Computing Clustering partnerships..."):
            distance_matrix_clusters = dm.iloc[:-1, :-1].values
            labels = get_clusters_for_file(distance_matrix_clusters)
            if labels is None:
                st.error("Clustering failed. Please check the distance matrix and clustering parameters.")
                st.stop()
            ranked_pairs_clust = rank_partnerships_using_clusters(df, labels, distance_matrix_clusters)
            st.session_state["ranked_pairs_clust"] = ranked_pairs_clust
            best_clust = get_best_partnerships_cluster(ranked_pairs_clust)
            st.session_state["best_partnerships_clust"] = best_clust

        st.success("Distance matrix and partnerships generated successfully.")

    except Exception as e:
        st.error(f"Error generating matrix: {e}")
        st.stop()

# Show Partnerships
st.header("Ranked Partnerships by Heuristic")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Bounding Box Heuristic")
    if "ranked_pairs_bb" in st.session_state and st.session_state["ranked_pairs_bb"] is not None:
        if not st.session_state["ranked_pairs_bb"].empty:
            st.dataframe(st.session_state["ranked_pairs_bb"].reset_index(drop=True))
        else:
            st.warning("No ranked pairs found for Bounding Box heuristic.")
    else:
        st.info("Bounding Box partnerships not generated yet.")

with col2:
    st.subheader("Clustering Heuristic")
    if "ranked_pairs_clust" in st.session_state and st.session_state["ranked_pairs_clust"] is not None:
        if not st.session_state["ranked_pairs_clust"].empty:
            st.dataframe(st.session_state["ranked_pairs_clust"].reset_index(drop=True))
        else:
            st.warning("No ranked pairs found for Clustering heuristic.")
    else:
        st.info("Clustering partnerships not generated yet.")

st.header("Potential Best Partnerships")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Bounding Box Best Partnerships")
    best_partnerships_bb = st.session_state.get("best_partnerships_bb", None)
    if best_partnerships_bb is not None and not best_partnerships_bb.empty:
        st.table(best_partnerships_bb.drop(columns=["Overlap Percentage"], errors='ignore').reset_index(drop=True))
    else:
        st.warning("No best Bounding Box partnerships found.")

with col4:
    st.subheader("Clustering Best Partnerships")
    best_partnerships_clust = st.session_state.get("best_partnerships_clust", None)
    if best_partnerships_clust is not None and not best_partnerships_clust.empty:
        st.table(best_partnerships_clust.reset_index(drop=True))
    else:
        st.warning("No best Clustering partnerships found.")

###############################################################################
# SINGLE PAIR SOLUTION
###############################################################################
st.header("Single Pair VRP Solution")

if (
    df is not None
    and st.session_state.get("distance_matrix_generated", False)
    and (("ranked_pairs_bb" in st.session_state and not st.session_state["ranked_pairs_bb"].empty) or
         ("ranked_pairs_clust" in st.session_state and not st.session_state["ranked_pairs_clust"].empty))
):
    # Combine both heuristics for selection
    combined_ranked_pairs = pd.DataFrame()

    if "ranked_pairs_bb" in st.session_state and not st.session_state["ranked_pairs_bb"].empty:
        ranked_bb = st.session_state["ranked_pairs_bb"].copy()
        ranked_bb['Heuristic'] = 'Bounding Box'
        combined_ranked_pairs = pd.concat([combined_ranked_pairs, ranked_bb], ignore_index=True)

    if "ranked_pairs_clust" in st.session_state and not st.session_state["ranked_pairs_clust"].empty:
        ranked_clust = st.session_state["ranked_pairs_clust"].copy()
        ranked_clust['Heuristic'] = 'Clustering'
        combined_ranked_pairs = pd.concat([combined_ranked_pairs, ranked_clust], ignore_index=True)

    if not combined_ranked_pairs.empty:
        # Display the combined ranked pairs with index numbers
        st.subheader("Combined Ranked Partnerships")
        st.dataframe(combined_ranked_pairs.reset_index(drop=True))

        # List of pairs with their index numbers
        st.subheader("Available Pairs with Index Numbers")
        pairs_list = combined_ranked_pairs.reset_index().apply(
            lambda row: f"Index {row['index']}: {row['Company1']} & {row['Company2']} ({row['Heuristic']})",
            axis=1
        )
        st.write(pairs_list.to_list())

        # Allow user to select a pair based on index number
        st.subheader("Select a Pair to Solve VRP")
        pair_selection = st.selectbox(
            "Select a pair by its index number:",
            options=combined_ranked_pairs.reset_index()['index'].tolist(),
            format_func=lambda x: f"Index {x}"
        )

        if st.button("Solve VRP for Selected Pair"):
            try:
                selected_pair = combined_ranked_pairs.loc[pair_selection]
                c1, c2 = selected_pair["Company1"], selected_pair["Company2"]
                heuristic = selected_pair["Heuristic"]

                st.write(f"### Solving VRP for Pair: **{c1} & {c2}** (Heuristic: {heuristic})")

                # Filter distance matrix for the selected pair
                distance_matrix = st.session_state["distance_matrix"]
                filt = get_filtered_matrix_for_pair(distance_matrix, c1, c2)
                dm_filtered = filt.copy()

                # Solve VRP for the pair
                pair_df = pd.DataFrame([selected_pair])
                pair_result = solve_vrp_for_all_pairs(
                    best_pairs_df=pair_df,
                    distance_matrix=distance_matrix,
                    cost_per_truck=cost_per_truck,
                    cost_per_km=cost_per_km,
                    time_per_vrp=time_per_vrp,
                    flag=False,
                    nmbr_loc=nmbr_loc,
                    algorithm=heuristic
                )

                if pair_result.empty:
                    st.error("No VRP solution found for this pair.")
                else:
                    st.session_state["pair_result_selected"] = pair_result

                    # Compare cost vs. solo
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
                st.error(f"Error solving VRP for the selected pair: {e}")

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

        # Map for single pair
        st.subheader("Map for Selected Pair Routes")
        if st.button("Create Map for Selected Pair"):
            pair_df = st.session_state.get("pair_result_selected", None)
            if pair_df is None or pair_df.empty:
                st.warning("No pair VRP result found. Solve VRP for a pair first.")
            else:
                try:
                    depot_loc = (depot_lat, depot_lon)  # Correctly pass depot location as tuple
                    data_original = st.session_state["uploaded_data"]
                    route_map, map_file = generate_route_map_fixed_with_legend(
                        depot_loc,
                        data_original,
                        pair_df,
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

###############################################################################
# COMPUTE VRP FOR ALL BEST PARTNERSHIPS
###############################################################################
st.header("Solve VRP for Best Partnerships")

if df is not None and st.session_state.get("distance_matrix_generated", False):
    if st.button("Compute VRP for Best Partnerships"):
        try:
            distance_matrix = st.session_state["distance_matrix"]

            # Compute VRP for Bounding Box partnerships
            best_bb_df = st.session_state.get("best_partnerships_bb", None)
            if best_bb_df is not None and not best_bb_df.empty:
                with st.spinner("Solving VRP for Bounding Box partnerships..."):
                    pair_res_bb = solve_vrp_for_all_pairs(
                        best_pairs_df=best_bb_df,
                        distance_matrix=distance_matrix,
                        cost_per_truck=cost_per_truck,
                        cost_per_km=cost_per_km,
                        time_per_vrp=time_per_vrp,
                        flag=False,
                        nmbr_loc=nmbr_loc,
                        algorithm="Bounding Box"
                    )

                if not pair_res_bb.empty:
                    st.session_state["pair_result_bb"] = pair_res_bb

                    # Build comparison table for Bounding Box partnerships
                    comp_rows_bb = []
                    for idx, row in pair_res_bb.iterrows():
                        c1 = row["Company1"]
                        c2 = row["Company2"]
                        paired_cost = row["Total Distance"]

                        # Solve individually
                        filt = get_filtered_matrix_for_pair(distance_matrix, c1, c2)
                        dm1, dm2 = get_individual_company_matrices(c1, c2, filt)

                        solo1 = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km, time_per_vrp, False, nmbr_loc, dm1)
                        solo2 = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km, time_per_vrp, False, nmbr_loc, dm2)

                        solo_cost_1 = solo1.get("Total Distance", 0)
                        solo_cost_2 = solo2.get("Total Distance", 0)
                        total_solo = solo_cost_1 + solo_cost_2
                        savings = total_solo - paired_cost
                        savings_pct = (savings / total_solo * 100) if total_solo > 0 else 0

                        comp_rows_bb.append({
                            "Company1": c1,
                            "Company2": c2,
                            "Paired Cost (€)": paired_cost,
                            "Solo Cost (C1)": solo_cost_1,
                            "Solo Cost (C2)": solo_cost_2,
                            "Total Solo (€)": total_solo,
                            "Savings (€)": savings,
                            "Savings (%)": savings_pct
                        })
                    bb_comp_df = pd.DataFrame(comp_rows_bb)
                    st.session_state["bb_comparison_table"] = bb_comp_df
                else:
                    st.warning("No Bounding Box pair VRP solutions found.")
            else:
                st.warning("No best Bounding Box partnerships available.")

            # Compute VRP for Clustering partnerships
            best_clust_df = st.session_state.get("best_partnerships_clust", None)
            if best_clust_df is not None and not best_clust_df.empty:
                with st.spinner("Solving VRP for Clustering partnerships..."):
                    pair_res_clust = solve_vrp_for_all_pairs(
                        best_pairs_df=best_clust_df,
                        distance_matrix=distance_matrix,
                        cost_per_truck=cost_per_truck,
                        cost_per_km=cost_per_km,
                        time_per_vrp=time_per_vrp,
                        flag=False,
                        nmbr_loc=nmbr_loc,
                        algorithm="Clustering"
                    )

                if not pair_res_clust.empty:
                    st.session_state["pair_result_clust"] = pair_res_clust

                    # Build comparison table for Clustering partnerships
                    comp_rows_clust = []
                    for idx, row in pair_res_clust.iterrows():
                        c1 = row["Company1"]
                        c2 = row["Company2"]
                        paired_cost = row["Total Distance"]

                        # Solve individually
                        filt = get_filtered_matrix_for_pair(distance_matrix, c1, c2)
                        dm1, dm2 = get_individual_company_matrices(c1, c2, filt)

                        solo1 = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km, time_per_vrp, False, nmbr_loc, dm1)
                        solo2 = solve_cvrp_numeric_ids(cost_per_truck, cost_per_km, time_per_vrp, False, nmbr_loc, dm2)

                        solo_cost_1 = solo1.get("Total Distance", 0)
                        solo_cost_2 = solo2.get("Total Distance", 0)
                        total_solo = solo_cost_1 + solo_cost_2
                        savings = total_solo - paired_cost
                        savings_pct = (savings / total_solo * 100) if total_solo > 0 else 0

                        comp_rows_clust.append({
                            "Company1": c1,
                            "Company2": c2,
                            "Paired Cost (€)": paired_cost,
                            "Solo Cost (C1)": solo_cost_1,
                            "Solo Cost (C2)": solo_cost_2,
                            "Total Solo (€)": total_solo,
                            "Savings (€)": savings,
                            "Savings (%)": savings_pct
                        })
                    clust_comp_df = pd.DataFrame(comp_rows_clust)
                    st.session_state["clust_comparison_table"] = clust_comp_df
                else:
                    st.warning("No Clustering pair VRP solutions found.")
            else:
                st.warning("No best Clustering partnerships available.")

            st.success("VRP computations for best partnerships completed.")

        except Exception as e:
            st.error(f"Error computing VRP for best partnerships: {e}")

# Show Bounding Box routes CSV
if "pair_result_bb" in st.session_state and st.session_state["pair_result_bb"] is not None:
    if not st.session_state["pair_result_bb"].empty:
        st.subheader("Download Bounding Box Partnerships VRP Routes as CSV")
        all_pairs_csv_bb = prepare_pairs_vrp_csv(st.session_state["pair_result_bb"])
        csv_data_bb = convert_df(all_pairs_csv_bb)
        st.download_button(
            label="Download Bounding Box Routes CSV",
            data=csv_data_bb,
            file_name="bounding_box_all_pairs_routes.csv",
            mime="text/csv"
        )

# Show Clustering routes CSV
if "pair_result_clust" in st.session_state and st.session_state["pair_result_clust"] is not None:
    if not st.session_state["pair_result_clust"].empty:
        st.subheader("Download Clustering Partnerships VRP Routes as CSV")
        all_pairs_csv_clust = prepare_pairs_vrp_csv(st.session_state["pair_result_clust"])
        csv_data_clust = convert_df(all_pairs_csv_clust)
        st.download_button(
            label="Download Clustering Routes CSV",
            data=csv_data_clust,
            file_name="clustering_all_pairs_routes.csv",
            mime="text/csv"
        )

# Show Bounding Box comparison table
if "bb_comparison_table" in st.session_state and st.session_state["bb_comparison_table"] is not None:
    st.subheader("Bounding Box VRP Comparison Table")
    df_bb_comp = st.session_state["bb_comparison_table"]
    st.dataframe(df_bb_comp.style.format({
        "Paired Cost (€)": "{:.2f}",
        "Solo Cost (C1)": "{:.2f}",
        "Solo Cost (C2)": "{:.2f}",
        "Total Solo (€)": "{:.2f}",
        "Savings (€)": "{:.2f}",
        "Savings (%)": "{:.2f}"
    }))

# Show Clustering comparison table
if "clust_comparison_table" in st.session_state and st.session_state["clust_comparison_table"] is not None:
    st.subheader("Clustering VRP Comparison Table")
    df_clust_comp = st.session_state["clust_comparison_table"]
    st.dataframe(df_clust_comp.style.format({
        "Paired Cost (€)": "{:.2f}",
        "Solo Cost (C1)": "{:.2f}",
        "Solo Cost (C2)": "{:.2f}",
        "Total Solo (€)": "{:.2f}",
        "Savings (€)": "{:.2f}",
        "Savings (%)": "{:.2f}"
    }))

###############################################################################
# SOLO + GLOBAL MATCHING
###############################################################################
st.header("Compute SOLO VRP & Global Perfect Matching")

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
            with st.spinner("Solving global perfect matching..."):
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

            st.subheader("Matching Pairs")
            st.dataframe(global_pairs_df)
            st.write(f"**Total Cost**: €{total_global_cost:,.2f}")
        except Exception as e:
            st.error(f"Error solving global pairing: {e}")

# Download SOLO results
if "vrp_result_solo" in st.session_state and st.session_state["vrp_result_solo"] is not None:
    st.subheader("Download SOLO VRP CSV")
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

# Retrieve all relevant data
solo_df = st.session_state.get("vrp_result_solo", None)
pairs_bb_df = st.session_state.get("pair_result_bb", None)
pairs_clust_df = st.session_state.get("pair_result_clust", None)
all_pairs_df = st.session_state.get("all_pairs_result", None)
all_pairs_cost = st.session_state.get("all_pairs_total_cost", None)

# Calculate total costs
total_solo_cost = solo_df["Total Distance"].sum() if solo_df is not None and not solo_df.empty else None
total_bb_cost = get_total_cost_from_pairs(pairs_bb_df)
total_clust_cost = get_total_cost_from_pairs(pairs_clust_df)
total_global_pairs = all_pairs_cost

# Prepare comparison data
comparison_data = {
    "Solution Type": [],
    "Total Cost (€)": []
}
if total_solo_cost is not None:
    comparison_data["Solution Type"].append("All Routes Solo")
    comparison_data["Total Cost (€)"].append(total_solo_cost)

if total_bb_cost is not None:
    comparison_data["Solution Type"].append("Bounding Box Pairs")
    comparison_data["Total Cost (€)"].append(total_bb_cost)

if total_clust_cost is not None:
    comparison_data["Solution Type"].append("Clustering Pairs")
    comparison_data["Total Cost (€)"].append(total_clust_cost)

if total_global_pairs is not None:
    comparison_data["Solution Type"].append("Global Perfect Matching (All in Pairs)")
    comparison_data["Total Cost (€)"].append(total_global_pairs)

# Display Final Cost Comparison
if comparison_data["Solution Type"]:
    st.header("Final Cost Comparison")
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.style.format({"Total Cost (€)": "{:.2f}"}))
    if len(comparison_data["Solution Type"]) >= 2:
        min_cost_idx = comparison_df["Total Cost (€)"].idxmin()
        best_sol = comparison_df.loc[min_cost_idx, "Solution Type"]
        best_val = comparison_df.loc[min_cost_idx, "Total Cost (€)"]
        st.success(f"Optimal solution so far: {best_sol} at €{best_val:.2f}")
else:
    st.info("Compute VRP solutions (solo, bounding box, clustering, global) to see final comparison.")

st.write("---")
st.write("End of App.")
