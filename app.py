import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ranking import rank_company_pairs_by_overlap
from ranking import plot_dbscan_clusters
from solver import unique_company_names, create_batched_distance_matrix, solve_cvrp_numeric_ids, filter_company_distances, solo_routes

# If you need to add variables, do it here for clarity
map_center = [52.37, 4.95]

# Dashboard title
st.title('Dashboard')

# Define bounding box for the Netherlands
NETHERLANDS_BOUNDS = {
    "min_lat": 50.5,
    "max_lat": 53.7,
    "min_lon": 3.36,
    "max_lon": 7.22
}

# Function to validate if a location is within the Netherlands
def is_within_netherlands(lat, lon):
    return (
        NETHERLANDS_BOUNDS["min_lat"] <= lat <= NETHERLANDS_BOUNDS["max_lat"] and
        NETHERLANDS_BOUNDS["min_lon"] <= lon <= NETHERLANDS_BOUNDS["max_lon"]
    )

# Step 1: Pick Depot Location
st.subheader("Step 1: Select a Central Depot Location")
m = folium.Map(location=map_center, zoom_start=10)
m.add_child(folium.LatLngPopup())  # Enable clicking to get lat/lon
map_data = st_folium(m, width=1100, height=700)

# Sidebar inputs for exact latitude and longitude
st.sidebar.header("Manual Depot Location Input")
manual_lat = st.sidebar.number_input("Enter Latitude", value=52.0, step=0.01, format="%.6f")
manual_lon = st.sidebar.number_input("Enter Longitude", value=5.0, step=0.01, format="%.6f")
use_manual_location = st.sidebar.checkbox("Use manual location", value=False)

# Sidebar inputs for VRP solver
st.sidebar.header('Parameters Solver')
nmbr_loc = st.sidebar.number_input('Capacity of truck', min_value=0, value=4)
cost_per_km = st.sidebar.number_input('Cost per kilometer', min_value=0, value=1)
cost_per_truck = st.sidebar.number_input('Cost per truck', min_value=0, value=800)
exact_solution = st.sidebar.checkbox('Use Exact Solution', value=False)
time_per_VRP = st.sidebar.number_input('Maximum time spent per VRP', min_value=0, value=10)

# Capture the pin location from the map or use manual input
if use_manual_location:
    selected_lat = manual_lat
    selected_lon = manual_lon
    st.write(f"Manual Central Depot Location: Latitude = {selected_lat}, Longitude = {selected_lon}")
elif map_data is not None and "last_clicked" in map_data and map_data["last_clicked"] is not None:
    selected_lat = map_data["last_clicked"]["lat"]
    selected_lon = map_data["last_clicked"]["lng"]
    st.write(f"Selected Central Depot Location: Latitude = {selected_lat}, Longitude = {selected_lon}")
else:
    selected_lat, selected_lon = 52.0, 5.0  # Default values
    st.write("Click on the map or use manual inputs to select a central depot location.")

# Validate depot location
if not is_within_netherlands(selected_lat, selected_lon):
    st.error("The selected depot location is outside the Netherlands. Please select a valid location.")
    st.stop()

# Step 2: Upload File
st.subheader("Step 2: Upload a File")
uploaded_file = st.file_uploader("Upload file", type=["csv"])

# Process uploaded file
if uploaded_file is not None:
    # Read the uploaded file
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Validate required columns
    required_columns = {'name', 'lat', 'lon'}
    if not required_columns.issubset(df.columns):
        st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
        st.stop()

    # Process valid data
    st.write("Uploaded Data:")
    st.dataframe(df)

    # Rank company pairs
    try:
        ranked_pairs = rank_company_pairs_by_overlap(df)
        st.subheader('Ranking of Company Pairs')
        st.dataframe(ranked_pairs)
    except Exception as e:
        st.error(f"Error ranking company pairs: {e}")

    # Plot clusters and add depot marker
    try:
        # Create a Folium map centered around the average location of the uploaded data
        cluster_map = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=10)

        # Add depot location marker as a circle
        folium.CircleMarker(
            location=[selected_lat, selected_lon],
            radius=10,  # Size of the circle
            color="red",  # Circle border color
            fill=True,
            fill_color="red",
            popup="Depot Location"
        ).add_to(cluster_map)

        # Add company locations to the map as circles
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,  # Size of the circle
                color="blue",  # Circle border color
                fill=True,
                fill_color="blue",
                popup=row['name']
            ).add_to(cluster_map)

        # Display the map with the depot and company locations
        st.write("Cluster Map with Depot Location:")
        st_folium(cluster_map, width=700, height=500)
    except Exception as e:
        st.error(f"Error generating map: {e}")



    # Use the selected pin for depot location if available
    if selected_lat is not None and selected_lon is not None:
        UNIVERSAL_DEPOT = (selected_lat, selected_lon)
    else:
        UNIVERSAL_DEPOT = (52.0, 5.0)  # Default depot location

    # Button to trigger VRP calculations
    if st.sidebar.button('Calculate VRP Solution'):
        try:
            # Prepare data for VRP solver
            locations = [{
                'lon': row['lon'],
                'lat': row['lat'],
                'name': row['name'],
                'unique_name': f"{row['name']}_{i}"
            } for i, row in df.iterrows()]

            # Add Universal Depot
            locations.append({
                'lon': UNIVERSAL_DEPOT[1],
                'lat': UNIVERSAL_DEPOT[0],
                'name': 'Universal Depot',
                'unique_name': 'Universal_Depot'
            })

            # Create distance matrix
            distance_matrix = create_batched_distance_matrix(locations)
            st.write("Distance Matrix:")
            st.dataframe(distance_matrix)

            # Solve CVRP
            result = solo_routes(cost_per_truck, cost_per_km, time_per_VRP, exact_solution, nmbr_loc, distance_matrix)

            # Display results
            if isinstance(result, pd.DataFrame) and not result.empty:
                st.write("VRP Solution:")
                for idx, row in result.iterrows():
                    company_name = row.get("Company", f"Company {idx + 1}")
                    st.subheader(f"Result for {company_name}")

                    # Display routes
                    routes = row.get("Routes", {})
                    if isinstance(routes, str):  # Parse stringified routes
                        import ast
                        routes = ast.literal_eval(routes)
                    for vehicle_id, route in routes.items():
                        st.write(f"Vehicle {vehicle_id}: {' -> '.join(map(str, route))}")

                    # Display total distance
                    total_distance = row.get("Total Distance", None)
                    if total_distance is not None:
                        st.write(f"Total Distance: {total_distance:.1f} meters")
            else:
                st.error("No VRP solution found.")
        except Exception as e:
            st.error(f"Error processing VRP: {e}")

# Download button
st.write('Download data')
st.download_button('Download data', data="no_data", file_name='empty.csv', mime='text/csv')
