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

# Combined heuristic
from combined import combine_heuristics_normalized
from combined import get_best_partnerships as get_best_partnerships_combined

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

# Step 1: Depot Location
st.sidebar.header("Depot Location")
lat_input = st.sidebar.number_input("Latitude", value=52.0, step=0.01, format="%.6f")
lon_input = st.sidebar.number_input("Longitude", value=5.0, step=0.01, format="%.6f")

# Step 1: Algorithm Selection
st.sidebar.header("Algorithm Selection")
algorithm_options = ["Bounding Box", "Clustering", "Combined"]
selected_algorithm = st.sidebar.selectbox("Select Algorithm", algorithm_options)

def create_distance_matrix(locations, batch_size, max_workers, base_url, profile, algorithm):
    if algorithm == "Bounding Box":
        return create_batched_distance_matrix(locations, batch_size, max_workers, base_url, profile)
    else:
        return create_batched_distance_matrix(locations, batch_size, max_workers, base_url, profile)



def solve_vrp_for_all_pairs(best_bb_df, distance_matrix, cost_per_truck, cost_per_km, time_per_vrp, flag, nmbr_loc, algorithm):
    return solve_vrp_for_all_pairs_in_dataframe(best_bb_df, distance_matrix, cost_per_truck, cost_per_km, time_per_vrp, flag, nmbr_loc)

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
            "lon": depot_lat,
            "lat": depot_lon,
            "name": "Universal Depot",
            "unique_name": "Universal_Depot"
        })

        # Verify dimensions
        assert len(locations) == len(df) + 1, "Mismatch in locations and dataframe dimensions"

        # Create matrix based on selected algorithm
        dm = create_distance_matrix(
            locations,
            batch_size=100,
            max_workers=4,
            base_url="http://router.project-osrm.org",
            profile=profile,
            algorithm=selected_algorithm
        )

        st.session_state["distance_matrix"] = dm
        st.session_state["distance_matrix_generated"] = True

        # Algorithm-specific logic
        if selected_algorithm == "Bounding Box":
            ranked_pairs = rank_company_pairs_by_overlap_percentage(df)
            st.session_state["ranked_pairs"] = ranked_pairs
            best_bb = get_best_partnerships_bb(ranked_pairs)
            st.session_state["best_partnerships_bb"] = best_bb
        elif selected_algorithm == "Clustering":
            labels = get_clusters_for_file(dm)
            ranked_pairs = rank_partnerships_using_clusters(df, labels, dm)
            st.session_state["ranked_pairs"] = ranked_pairs
            best_bb = get_best_partnerships_cluster(ranked_pairs)
            st.session_state["best_partnerships_bb"] = best_bb
        elif selected_algorithm == "Combined":
            ranked_pairs = combine_heuristics_normalized(rank_company_pairs_by_overlap_percentage(df), rank_partnerships_using_clusters(df, get_clusters_for_file(dm), dm), weight_overlap=0.5, weight_cluster=0.5)
            st.session_state["ranked_pairs"] = ranked_pairs
            best_bb = get_best_partnerships_combined(ranked_pairs)
            st.session_state["best_partnerships_bb"] = best_bb
        st.success("Distance matrix + partnerships generated.")
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

st.subheader("Potential Best Partnerships")
best_partnerships_bb = st.session_state.get("best_partnerships_bb", None)
if best_partnerships_bb is not None and not best_partnerships_bb.empty:
    if selected_algorithm == "Bounding Box":
        st.table(best_partnerships_bb.drop(columns=["Overlap Percentage"]).reset_index(drop=True))
else:
    st.warning("No best partnerships found.")

###############################################################################
# SINGLE PAIR SOLUTION
###############################################################################
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

# Map for single pair
if st.button("Create Map for Selected Pair"):
    pair_df = st.session_state.get("pair_result_selected", None)
    if pair_df is None or pair_df.empty:
        st.warning("No pair VRP result found. Solve VRP for a pair first.")
    else:
        try:
            depot_loc = st.session_state["depot_location"]
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
st.subheader("Solve VRP for Best Partnerships")
# Solve VRP for Best Partnerships
if df is not None and st.session_state.get("distance_matrix_generated", False):
    if st.button("Compute VRP for Best Partnerships"):
        try:
            distance_matrix = st.session_state["distance_matrix"]
            best_bb_df = st.session_state.get("best_partnerships_bb", None)
            if best_bb_df is not None and not best_bb_df.empty:
                pair_res_bb = solve_vrp_for_all_pairs(
                    best_bb_df,
                    distance_matrix,
                    cost_per_truck,
                    cost_per_km,
                    time_per_vrp,
                    False,
                    nmbr_loc,
                    algorithm=selected_algorithm
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
                    st.warning("No pair VRP solutions found.")
            else:
                st.warning("No best partnerships available.")
        except Exception as e:
            st.error(f"Error computing VRP: {e}")

# Show bounding-box routes CSV
if "pair_result_bb" in st.session_state and st.session_state["pair_result_bb"] is not None:
    if not st.session_state["pair_result_bb"].empty:
        st.write("### Download Full Partnerships VRP Routes as CSV")
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
    st.write("### Heuristic VRP Comparison Table")
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
    comparison_data["Solution Type"].append("Heursitics pairs")
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
    st.info("Compute VRP solutions (solo, heuristic, global) to see final comparison.")

st.write("---")
st.write("End of App.")
