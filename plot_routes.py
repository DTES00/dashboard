import folium
from folium import Marker, PolyLine, CircleMarker
from folium.map import LayerControl
import itertools
from folium.plugins import FloatImage
import requests


def generate_route_map_fixed_with_legend(depot_loc, data, test, osrm_url="http://localhost:5000"):
    # Create a base map centered around the coordinates
    center_lat = data['lat'].mean()
    center_lon = data['lon'].mean()
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=8)

    # List of distinct colors for companies
    colors = itertools.cycle([
        "blue", "green", "purple", "orange", "darkred",
        "lightred", "beige", "darkblue", "darkgreen", "cadetblue",
        "pink", "lightblue", "lightgreen", "gray", "black"
    ])

    # Create a mapping of company to color
    company_colors = {}
    for company in test['Company'].unique():
        company_colors[company] = next(colors)

    def plot_routes_for_company(company, routes):
        # Create a feature group for the company
        company_group = folium.FeatureGroup(name=f"{company} Routes", show=True)
        
        # Filter company-specific data
        company_data = data[data['name'] == company]
        coords = company_data[['lat', 'lon']].values.tolist()

        # Define the shared Source/Sink location (Depot)
        source_sink_location = depot_loc

        # Add a marker for the depot location
        if source_sink_location:
            Marker(
                location=source_sink_location,
                popup=f"{company} - Depot",
                icon=folium.Icon(color="red", icon="home", prefix="fa")
            ).add_to(company_group)

        # Add a circle for every location visited by the company
        for location in coords:
            CircleMarker(
                location=location,
                radius=6,
                color=company_colors[company],  # Match the company's color
                fill=True,
                fill_color=company_colors[company],
                fill_opacity=0.5,
                popup=f"{company} - Location"
            ).add_to(company_group)

        for route_id, stops in routes.items():
            # Extract the indices from the route (ignore 'Source' and 'Sink')
            route_indices = [s for s in stops if isinstance(s, int)]
            route_coords = [coords[i] for i in route_indices]

            # Ensure the route starts and ends at Source/Sink
            if source_sink_location:
                route_coords = [source_sink_location] + route_coords + [source_sink_location]

            # Generate realistic paths using OSRM
            real_path = []
            for i in range(len(route_coords) - 1):
                start = route_coords[i]
                end = route_coords[i + 1]

                try:
                    # Format the coordinates for OSRM API with higher precision
                    coordinates = f"{start[1]},{start[0]};{end[1]},{end[0]}"
                    url = f"{osrm_url}/route/v1/driving/{coordinates}?geometries=geojson&steps=true"

                    # Make the API request to OSRM
                    response = requests.get(url)
                    response.raise_for_status()

                    osrm_data = response.json()

                    # Extract the route geometry
                    if 'routes' in osrm_data and osrm_data['routes']:
                        geometry = osrm_data['routes'][0]['geometry']['coordinates']
                        real_path.extend([(lat, lon) for lon, lat in geometry])  # GeoJSON is [lon, lat]
                    else:
                        raise ValueError(f"No routes found between {start} and {end}")

                except Exception as e:
                    # Log the error and skip this segment
                    print(f"Error plotting route between {start} and {end}: {e}")
                    continue

            # Add the realistic path as a polyline to the company's layer
            if real_path:
                PolyLine(
                    real_path,
                    color=company_colors[company],  # Use the unique color for this company
                    weight=2.5,
                    opacity=0.6
                ).add_to(company_group)

        # Add the company feature group to the map
        company_group.add_to(map_obj)

    # Iterate through the test dataframe to plot routes for each company
    for _, row in test.iterrows():
        company = row['Company']
        routes = row['Routes']  # Directly use the dictionary if it's already in dictionary format
        plot_routes_for_company(company, routes)

    # Add layer control to toggle companies on/off
    LayerControl(collapsed=False).add_to(map_obj)

    # Save the map to an HTML file
    map_file = "routes_map.html"
    map_obj.save(map_file)
    print(f"Map saved to {map_file}")

    return map_obj, map_file
