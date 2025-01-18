import folium
from folium import Marker, PolyLine, CircleMarker
from folium.map import LayerControl
import itertools
from folium.plugins import FloatImage
import requests
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from ranking import rank_company_pairs_by_overlap
from solver import create_batched_distance_matrix, solo_routes
from plot_routes import generate_route_map_fixed_with_legend

# If you need to add variables, do it here for clarity
map_center = [52.37, 4.95]

# Dashboard title
st.title("Dashboard")

# Define bounding box for the Netherlands
NETHERLANDS_BOUNDS = {
    "min_lat": 50.5,
    "max_lat": 53.7,
    "min_lon": 3.36,
    "max_lon": 7.22,
}

# Function to validate if a location is within the Netherlands
def is_within_netherlands(lat, lon):
    return (
        NETHERLANDS_BOUNDS["min_lat"] <= lat <= NETHERLANDS_BOUNDS["max_lat"]
        and NETHERLANDS_BOUNDS["min_lon"] <= lon <= NETHERLANDS_BOUNDS["max_lon"]
    )

# Step 1: Select Depot Location
if "depot_location" not in st.session_state:
    st.session_state["depot_location"] = None

st.subheader("Step 1: Select a Central Depot Location")
m = folium.Map(location=map_center, zoom_start=10)
m.add_child(folium.LatLngPopup())  # Enable clicking to get lat/lon
map_data = st_folium(m, width=1100, height=700)

# Sidebar inputs for exact latitude and longitude
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

if st.session_state["depot_location"] is not None:
    selected_lat, selected_lon = st.session_state["depot_location"]
    if not is_within_netherlands(selected_lat, selected_lon):
        st.error("The selected depot location is outside the Netherlands. Please select a valid location.")
        st.stop()
else:
    st.error("Please select a depot location.")
    st.stop()

# Step 2: Upload File
st.subheader("Step 2: Upload a File")
uploaded_file = st.file_uploader("Upload file", type=["csv"])

if uploaded_file is not None:
    if "uploaded_data" not in st.session_state:
        try:
            df = pd.read_csv(uploaded_file)
            # Validate required columns
            required_columns = {"name", "lat", "lon"}
            if not required_columns.issubset(df.columns):
                st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
                st.stop()

            st.session_state["uploaded_data"] = df
            st.session_state["vrp_generated"] = False  # Reset VRP when a new file is uploaded
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

    df = st.session_state["uploaded_data"]
    #st.write("Uploaded Data:")
    #st.dataframe(df)

    # Rank company pairs
    try:
        ranked_pairs = rank_company_pairs_by_overlap(df)
        st.subheader("Ranking of Company Pairs")
        st.dataframe(ranked_pairs)
    except Exception as e:
        st.error(f"Error ranking company pairs: {e}")

    # Sidebar inputs for VRP solver
    st.sidebar.header("Parameters Solver")
    nmbr_loc = st.sidebar.number_input("Capacity of truck", min_value=0, value=4)
    cost_per_km = st.sidebar.number_input("Cost per kilometer", min_value=0, value=1)
    cost_per_truck = st.sidebar.number_input("Cost per truck", min_value=0, value=800)
    exact_solution = st.sidebar.checkbox("Use Exact Solution", value=False)
    time_per_VRP = st.sidebar.number_input("Maximum time spent per VRP", min_value=0, value=10)

    # Button to calculate VRP
    if st.sidebar.button("Calculate VRP Solution"):
        try:
            locations = [
                {"lon": row["lon"], "lat": row["lat"], "name": row["name"], "unique_name": f"{row['name']}_{i}"}
                for i, row in df.iterrows()
            ]

            # Add depot
            locations.append(
                {"lon": selected_lon, "lat": selected_lat, "name": "Universal Depot", "unique_name": "Universal_Depot"}
            )

            # Create distance matrix
            distance_matrix = create_batched_distance_matrix(locations)
            st.session_state["distance_matrix"] = distance_matrix  # Save matrix to session state
            st.write("Distance Matrix:")
            st.dataframe(distance_matrix)

            # Solve CVRP
            result = solo_routes(cost_per_truck, cost_per_km, time_per_VRP, exact_solution, nmbr_loc, distance_matrix)
            if isinstance(result, pd.DataFrame) and not result.empty:
                st.session_state["vrp_result"] = result
                st.session_state["vrp_generated"] = True
                st.success("VRP solution calculated.")
            else:
                st.error("No VRP solution found.")
        except Exception as e:
            st.error(f"Error processing VRP: {e}")

    # Display results, distance matrix, and map
    if st.session_state.get("vrp_generated", False):
        vrp_result = st.session_state["vrp_result"]

        # # Display distance matrix
        # if "distance_matrix" in st.session_state:
        #     st.write("Distance Matrix:")
        #     st.dataframe(st.session_state["distance_matrix"])

        # Display textual results
        st.write("VRP Solution:")
        for idx, row in vrp_result.iterrows():
            company_name = row.get("Company", f"Company {idx + 1}")
            st.subheader(f"Route for {company_name}")

            # Display routes
            routes = row.get("Routes", {})
            if isinstance(routes, str):
                import ast

                routes = ast.literal_eval(routes)
            for vehicle_id, route in routes.items():
                st.write(f"Vehicle {vehicle_id}: {' -> '.join(map(str, route))}")

            # Display total distance
            total_distance = row.get("Total Distance", None)
            if total_distance is not None:
                st.write(f"Total Cost: {total_distance:.1f} €")


        # Generate the map automatically after VRP calculation
        import streamlit.components.v1 as components  # Required to render HTML iframe

        # Generate the map automatically after VRP calculation
        if "route_map" not in st.session_state:
            try:
                depot_location = st.session_state["depot_location"]
                route_map, map_file = generate_route_map_fixed_with_legend(depot_location, df, vrp_result)
                st.session_state["route_map"] = route_map
                st.session_state["map_file"] = map_file
            except Exception as e:
                st.error(f"Error generating the map: {e}")

        # Provide a subheader
        st.subheader("Generated Routes Map (Interactive)")

        # Render the map using an HTML iframe
        if "map_file" in st.session_state:
            try:
                # Display the map using an iframe
                with open(st.session_state["map_file"], "r", encoding="utf-8") as map_file:
                    map_html = map_file.read()
                components.html(map_html, height=700, width=1100)
            except Exception as e:
                st.warning("The interactive map could not be rendered inline. Please download the map instead.")

        # Provide a download button for the map file
        if "map_file" in st.session_state:
            with open(st.session_state["map_file"], "rb") as file:
                st.download_button(
                    label="Download Map as HTML",
                    data=file,
                    file_name="routes_map.html",
                    mime="text/html",
                )




# Download button
st.write("Download data")
st.download_button("Download data", data="no_data", file_name="empty.csv", mime="text/csv")
