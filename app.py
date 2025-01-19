import folium
from folium import Marker, PolyLine, CircleMarker
from folium.map import LayerControl
import itertools
from folium.plugins import FloatImage
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import random


# --- Imports for Overlap & Bounding Box Logic ---
from bounding_box import (
    rank_company_pairs_by_overlap_percentage,
    visualize_bounding_boxes,
    get_best_partnerships as get_best_partnerships_bb,  # Renamed for clarity
)

# --- Imports for Clustering ---
from cluster import (
    get_clusters_for_file,
    rank_partnerships_using_clusters,
    get_best_partnerships as get_best_partnerships_clust,
)

# --- Imports for VRP ---
from solver import create_batched_distance_matrix, solo_routes, solve_vrp_for_all_pairs_in_dataframe

# --- Plotting of Routes ---
from plot_routes import generate_route_map_fixed_with_legend

# -------------
# GLOBALS
# -------------
map_center = [52.37, 4.95]

# Define bounding box for the Netherlands
NETHERLANDS_BOUNDS = {
    "min_lat": 50.5,
    "max_lat": 53.7,
    "min_lon": 3.36,
    "max_lon": 7.22,
}

# -------------
# FUNCTIONS
# -------------
def is_within_netherlands(lat, lon):
    """Validate if a location is within the Netherlands (rough bounding box)."""
    return (
        NETHERLANDS_BOUNDS["min_lat"] <= lat <= NETHERLANDS_BOUNDS["max_lat"]
        and NETHERLANDS_BOUNDS["min_lon"] <= lon <= NETHERLANDS_BOUNDS["max_lon"]
    )

# -------------
# STREAMLIT APP
# -------------

# Title
st.title("Dashboard")

# ----------------------------------
# STEP 1: SELECT DEPOT LOCATION
# ----------------------------------
if "depot_location" not in st.session_state:
    st.session_state["depot_location"] = None

st.subheader("Step 1: Select a Central Depot Location")

m = folium.Map(location=map_center, zoom_start=10)
m.add_child(folium.LatLngPopup())  # Enable clicking to get lat/lon
map_data = st_folium(m, width=1100, height=700)

# Sidebar for manual depot location
st.sidebar.header("Manual Depot Location Input")
manual_lat = st.sidebar.number_input("Enter Latitude", value=52.0, step=0.01, format="%.6f")
manual_lon = st.sidebar.number_input("Enter Longitude", value=5.0, step=0.01, format="%.6f")
use_manual_location = st.sidebar.checkbox("Use manual location", value=False)

if use_manual_location:
    selected_lat = manual_lat
    selected_lon = manual_lon
    st.session_state["depot_location"] = (selected_lat, selected_lon)
    st.write(f"Manual Central Depot Location: Latitude = {selected_lat}, Longitude = {selected_lon}")
elif map_data is not None and "last_clicked" in map_data and map_data["last_clicked"] is not None:
    selected_lat = map_data["last_clicked"]["lat"]
    selected_lon = map_data["last_clicked"]["lng"]
    st.session_state["depot_location"] = (selected_lat, selected_lon)
    st.write(f"Selected Central Depot Location: Latitude = {selected_lat}, Longitude = {selected_lon}")
else:
    st.write("Click on the map or use manual inputs to select a central depot location.")

# Validate Depot location
if st.session_state["depot_location"] is not None:
    selected_lat, selected_lon = st.session_state["depot_location"]
    if not is_within_netherlands(selected_lat, selected_lon):
        st.error("The selected depot location is outside the Netherlands. Please select a valid location.")
        st.stop()
else:
    st.error("Please select a depot location.")
    st.stop()

# ----------------------------------
# STEP 2: UPLOAD DATA
# ----------------------------------
st.subheader("Step 2: Upload a File")
uploaded_file = st.file_uploader("Upload file", type=["csv"])

if uploaded_file is not None:
    # Check if the uploaded file is new
    if "uploaded_file" in st.session_state and st.session_state["uploaded_file"] != uploaded_file:
        st.session_state.clear()  # Clear all session state variables when a new file is uploaded

    if "uploaded_data" not in st.session_state or st.session_state.get("uploaded_file") != uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = {"name", "lat", "lon"}
            if not required_columns.issubset(df.columns):
                st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
                st.stop()

            # Store the new file and data in session state
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["uploaded_data"] = df
            st.session_state["distance_matrix_generated"] = False  # Reset calculations
            
            st.success("File uploaded successfully. Processing the data...")

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()


df = st.session_state.get("uploaded_data", None)

if df is not None:
    st.write("Uploaded Data (Preview):")
    st.dataframe(df)

# ----------------------------------
# STEP 3: VRP PARAMETERS (SIDEBAR)
# ----------------------------------
st.sidebar.header("Parameters Solver")
nmbr_loc = st.sidebar.number_input("Max number of locations per route", min_value=0, value=4)
cost_per_km = st.sidebar.number_input("Cost (euro) per kilometer", min_value=0, value=1)
cost_per_truck = st.sidebar.number_input("Cost (euro) per truck", min_value=0, value=800)
exact_solution = st.sidebar.checkbox("Use Exact Solution", value=False)
time_per_VRP = st.sidebar.number_input("Maximum time spent per VRP (seconds)", min_value=0, value=10)

# ----------------------------------
# STEP 4: GENERATE DISTANCE MATRIX
#      and CALCULATE PARTNERSHIPS
# ----------------------------------


if df is not None and not st.session_state.get("distance_matrix_generated", False):
    try:


        # Prepare list of locations (including depot)
        locations = [
            {"lon": row["lon"], "lat": row["lat"], "name": row["name"], "unique_name": f"{row['name']}_{i}"}
            for i, row in df.iterrows()
        ]
        locations.append({
            "lon": selected_lon,
            "lat": selected_lat,
            "name": "Universal Depot",
            "unique_name": "Universal_Depot"
        })

        # 1) Automatically generate the distance matrix
        distance_matrix = create_batched_distance_matrix(locations)
        st.session_state["distance_matrix"] = distance_matrix
        st.session_state["distance_matrix_generated"] = True
 

        # 2) Automatically perform Bounding Box–based Pair Ranking
        ranked_pairs = rank_company_pairs_by_overlap_percentage(df)
        st.session_state["ranked_pairs"] = ranked_pairs
        best_partnerships_bb = get_best_partnerships_bb(ranked_pairs)
        st.session_state["best_partnerships_bb"] = best_partnerships_bb


        # 3) Automatically perform Cluster–based Pair Ranking
        distance_matrix_for_clusters = distance_matrix.iloc[:-1, :-1].values
        labels = get_clusters_for_file(distance_matrix_for_clusters)

        # Rank partnerships by cluster
        ranked_partnerships_cluster = rank_partnerships_using_clusters(df, labels, distance_matrix_for_clusters)
        st.session_state["ranked_partnerships_cluster"] = ranked_partnerships_cluster

        # Identify best cluster-based partnerships
        best_partnerships_clust_ = get_best_partnerships_clust(ranked_partnerships_cluster)
        st.session_state["best_partnerships_clust"] = best_partnerships_clust_

    except Exception as e:
        st.error(f"Error generating distance matrix or performing pair calculations: {e}")

st.write("## Potential Best Partnerships")

if "best_partnerships_bb" in st.session_state and "best_partnerships_clust" in st.session_state:
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Bounding-Box")
        st.table(st.session_state["best_partnerships_bb"])

    with col2:
        st.write("### Cluster-Based")
        cluster_data_without_id = st.session_state["best_partnerships_clust"].drop(columns=["Cluster ID"])
        st.table(cluster_data_without_id)
else:
    st.warning("Pairing data not available. Please upload a file to generate pairings.")


# ----------------------------------
# STEP 5: CALCULATE VRP (SOLO + PAIRS)
# ----------------------------------

if df is not None and st.session_state.get("distance_matrix_generated", False):
    if st.button("Calculate VRP Solutions"):
        try:
            distance_matrix = st.session_state["distance_matrix"]
            progress_bar = st.progress(0)

            # Solve SOLO VRP
            result_solo = solo_routes(
                cost_per_truck, cost_per_km, time_per_VRP, exact_solution, nmbr_loc, distance_matrix
            )
            if isinstance(result_solo, pd.DataFrame) and not result_solo.empty:
                st.session_state["vrp_result_solo"] = result_solo
                st.session_state["vrp_generated"] = True
                st.success("Individual VRP solution calculated.")
            else:
                st.error("No Individual VRP solution found.")
 
            progress_bar.progress(50) 

            # Solve VRP for bounding box pairs
            if "best_partnerships_bb" in st.session_state:
                best_partnerships_bb = st.session_state["best_partnerships_bb"]
                pair_result_bb = solve_vrp_for_all_pairs_in_dataframe(
                    best_partnerships_bb,
                    distance_matrix,
                    cost_per_truck,
                    cost_per_km,
                    time_per_VRP,
                    exact_solution,
                    nmbr_loc
                )
                if isinstance(pair_result_bb, pd.DataFrame) and not pair_result_bb.empty:
                    st.session_state["pair_result_bb"] = pair_result_bb
                    st.success("Pair VRP solution (Overlap) calculated.")
                else:
                    st.error("No pair VRP solution found (Overlap).")
            progress_bar.progress(75) 
            # Solve VRP for cluster-based pairs
            if "best_partnerships_clust" in st.session_state:

                best_partnerships_cl = st.session_state["best_partnerships_clust"]
                pair_result_cl = solve_vrp_for_all_pairs_in_dataframe(
                    best_partnerships_cl,
                    distance_matrix,
                    cost_per_truck,
                    cost_per_km,
                    time_per_VRP,
                    exact_solution,
                    nmbr_loc
                ) 
 
            

                if isinstance(pair_result_cl, pd.DataFrame) and not pair_result_cl.empty:
                    st.session_state["pair_result_cl"] = pair_result_cl
                    st.success("Pair VRP solution (Cluster-Based) calculated.")
                    progress_bar.progress(100) 
                else:
                    st.error("No pair VRP solution found (Cluster-Based).")
                

        except Exception as e:
            st.error(f"Error processing VRP: {e}")


# ----------------------------------
# STEP 6: COMPARISON TABLES
# ----------------------------------
st.subheader("Comparison of Solo vs. Paired Routes")

if "vrp_result_solo" in st.session_state:
    # Get solo route costs
    solo_costs = st.session_state["vrp_result_solo"].copy()
    solo_costs = solo_costs[["Company", "Total Distance"]]
    solo_costs.rename(columns={"Total Distance": "Solo Route Cost (€)"}, inplace=True)

    # ------ Overlap-Based Pairs Comparison ------
    if "pair_result_bb" in st.session_state:
        pair_costs_bb = st.session_state["pair_result_bb"].copy()
        pair_costs_bb = pair_costs_bb[["Company1", "Company2", "Total Distance"]]
        pair_costs_bb.rename(columns={"Total Distance": "Paired Route Cost (€)"}, inplace=True)

        # Merge solo costs for Company1 and Company2
        pair_costs_merged_bb = pd.merge(
            pair_costs_bb,
            solo_costs,
            left_on="Company1",
            right_on="Company",
            how="left"
        ).rename(columns={"Solo Route Cost (€)": "Solo Cost (Company1)"})

        pair_costs_merged_bb = pd.merge(
            pair_costs_merged_bb,
            solo_costs,
            left_on="Company2",
            right_on="Company",
            how="left"
        ).rename(columns={"Solo Route Cost (€)": "Solo Cost (Company2)"})

        # Drop duplicate columns safely
        pair_costs_merged_bb.drop(columns=[col for col in ["Company_x", "Company_y"] if col in pair_costs_merged_bb.columns], inplace=True)

        # Calculate potential savings
        pair_costs_merged_bb["Total Solo Cost (€)"] = (
            pair_costs_merged_bb["Solo Cost (Company1)"] + pair_costs_merged_bb["Solo Cost (Company2)"]
        )
        pair_costs_merged_bb["Savings (€)"] = (
            pair_costs_merged_bb["Total Solo Cost (€)"] - pair_costs_merged_bb["Paired Route Cost (€)"]
        )

        # Format the columns to show 2 decimal places
        pair_costs_merged_bb = pair_costs_merged_bb.round(2)

    # ------ Cluster-Based Pairs Comparison ------
    if "pair_result_cl" in st.session_state:
        pair_costs_cl = st.session_state["pair_result_cl"].copy()
        pair_costs_cl = pair_costs_cl[["Company1", "Company2", "Total Distance"]]
        pair_costs_cl.rename(columns={"Total Distance": "Paired Route Cost (€)"}, inplace=True)

        # Merge solo costs
        pair_costs_merged_cl = pd.merge(
            pair_costs_cl,
            solo_costs,
            left_on="Company1",
            right_on="Company",
            how="left"
        ).rename(columns={"Solo Route Cost (€)": "Solo Cost (Company1)"})

        pair_costs_merged_cl = pd.merge(
            pair_costs_merged_cl,
            solo_costs,
            left_on="Company2",
            right_on="Company",
            how="left"
        ).rename(columns={"Solo Route Cost (€)": "Solo Cost (Company2)"})

        # Drop duplicate columns safely
        pair_costs_merged_cl.drop(columns=[col for col in ["Company_x", "Company_y"] if col in pair_costs_merged_cl.columns], inplace=True)

        # Calculate potential savings
        pair_costs_merged_cl["Total Solo Cost (€)"] = (
            pair_costs_merged_cl["Solo Cost (Company1)"] + pair_costs_merged_cl["Solo Cost (Company2)"]
        )
        pair_costs_merged_cl["Savings (€)"] = (
            pair_costs_merged_cl["Total Solo Cost (€)"] - pair_costs_merged_cl["Paired Route Cost (€)"]
        )

        # Format the columns to show 2 decimal places
        pair_costs_merged_cl = pair_costs_merged_cl.round(2)

    # Ensure tables are aligned properly
    st.markdown(
        """
        <style>
        div.block-container {padding-top: 1rem;}
        table {
            width: 100%;
            text-align: center;
        }
        th {
            background-color: #f4f4f4;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the tables side by side in equal-width columns
    col1, col2 = st.columns([1, 1])

    with col1:
        if "pair_result_bb" in st.session_state:
            st.markdown("### Solo vs Bounding-Box")
            st.table(pair_costs_merged_bb.set_index(["Company1", "Company2"]))
        else:
            st.write("Bounding-Box data not available.")

    with col2:
        if "pair_result_cl" in st.session_state:
            st.markdown("### Solo vs Cluster-Based")
            st.table(pair_costs_merged_cl.set_index(["Company1", "Company2"]))
        else:
            st.write("Cluster-Based data not available.")

    # Calculate total costs
    total_solo_cost = solo_costs["Solo Route Cost (€)"].sum()
    total_bb_cost = pair_costs_merged_bb["Paired Route Cost (€)"].sum() if "pair_result_bb" in st.session_state else 0.0
    total_cl_cost = pair_costs_merged_cl["Paired Route Cost (€)"].sum() if "pair_result_cl" in st.session_state else 0.0

    # Display the totals below the tables
    st.markdown("### Total Route Costs Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Total Solo Route Cost (€)", value=f"{total_solo_cost:.2f}")

    with col2:
        st.metric(label="Total Bounding-Box Route Cost (€)", value=f"{total_bb_cost:.2f}")

    with col3:
        st.metric(label="Total Cluster-Based Route Cost (€)", value=f"{total_cl_cost:.2f}")





# ----------------------------------
# STEP 7: DISPLAY VRP RESULTS
# ----------------------------------


# --- 7A: SOLO VRP ---
if "vrp_result_solo" in st.session_state:
    solo_df = st.session_state["vrp_result_solo"]
    st.write("## Solo VRP Solution:")
    for idx, row in solo_df.iterrows():
        company_name = row.get("Company", f"Company {idx + 1}")
        st.subheader(f"Route for {company_name}")

        # Display routes
        routes = row.get("Routes", {})
        if isinstance(routes, str):
            import ast
            routes = ast.literal_eval(routes)
        for vehicle_id, route in routes.items():
            st.write(f"Vehicle {vehicle_id}: {' -> '.join(map(str, route))}")

        # Display total cost
        total_distance = row.get("Total Distance", None)
        if total_distance is not None:
            st.write(f"Total Cost: {total_distance:.1f} €")

# --- 7B: BOUNDING BOX PAIRS VRP ---
if "pair_result_bb" in st.session_state:
    pair_result_bb = st.session_state["pair_result_bb"]
    st.write("## Pair VRP Solution (Overlap-Based):")
    for idx, row in pair_result_bb.iterrows():
        company1 = row.get("Company1")
        company2 = row.get("Company2")
        st.subheader(f"Route for: {company1} & {company2}")

        # Display routes
        routes = row.get("Routes", {})
        if isinstance(routes, str):
            import ast
            routes = ast.literal_eval(routes)
        for vehicle_id, route in routes.items():
            st.write(f"Vehicle {vehicle_id}: {' -> '.join(map(str, route))}")

        # Display total cost
        total_distance = row.get("Total Distance", None)
        if total_distance is not None:
            st.write(f"Total Cost: {total_distance:.1f} €")

# --- 7C: CLUSTER PAIRS VRP ---
if "pair_result_cl" in st.session_state:
    pair_result_cl = st.session_state["pair_result_cl"]
    st.write("## Pair VRP Solution (Cluster-Based):")
    for idx, row in pair_result_cl.iterrows():
        company1 = row.get("Company1")
        company2 = row.get("Company2")
        st.subheader(f"Route for: {company1} & {company2}")

        # Display routes
        routes = row.get("Routes", {})
        if isinstance(routes, str):
            import ast
            routes = ast.literal_eval(routes)
        for vehicle_id, route in routes.items():
            st.write(f"Vehicle {vehicle_id}: {' -> '.join(map(str, route))}")

        # Display total cost
        total_distance = row.get("Total Distance", None)
        if total_distance is not None:
            st.write(f"Total Cost: {total_distance:.1f} €")





# ----------------------------------
# STEP 8: ROUTE MAP VISUALIZATION
# ----------------------------------
st.subheader("Route Visualization (Solo Example)")
# Check if VRP results are generated
if st.session_state.get("vrp_generated", False):
    if st.button("Generate Route Visualization"):
        vrp_result = st.session_state.get("vrp_result_solo", None)
        if vrp_result is not None and "route_map" not in st.session_state:
            try:
                depot_location = st.session_state["depot_location"]
                route_map, map_file = generate_route_map_fixed_with_legend(depot_location, df, vrp_result)
                st.session_state["route_map"] = route_map
                st.session_state["map_file"] = map_file
                st.success("Route visualization generated successfully.")
            except Exception as e:
                st.error(f"Error generating the map: {e}")

    # Display the map if available
    if "map_file" in st.session_state:
        try:
            with open(st.session_state["map_file"], "r", encoding="utf-8") as map_file:
                map_html = map_file.read()
            components.html(map_html, height=700, width=1100)
        except Exception:
            st.warning("The interactive map could not be rendered inline. Please download the map instead.")

        # Download button for the map file
        with open(st.session_state["map_file"], "rb") as file:
            st.download_button(
                label="Download Map as HTML",
                data=file,
                file_name="routes_map.html",
                mime="text/html",
            )
else:
    st.warning("Please generate VRP solutions first.")


# ----------------------------------
# FINAL DOWNLOAD BUTTON (Placeholder)
# ----------------------------------
st.write("Download data")
st.download_button(
    label="Download data (placeholder)",
    data="no_data",
    file_name="empty.csv",
    mime="text/csv"
)
