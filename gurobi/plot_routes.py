import folium
from folium import Marker, PolyLine, CircleMarker
from folium.map import LayerControl
import itertools
from folium.plugins import FloatImage
import requests


def generate_route_map_fixed_with_legend(depot_loc, data, test, osrm_url="http://localhost:5000", profile="driving"):
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

    def plot_routes_for_company(company, routes, data, depot_loc, osrm_url, profile, company_colors, company_group):
        for _, row in routes.iterrows():
            stops = row["Stops"]

            # Map stops to coordinates
            route_coords = []
            for stop in stops:
                if stop in ["Universal_Depot", "Depot"]:  # Updated condition
                    route_coords.append(depot_loc)  # Map depot to depot_loc
                else:
                    stop_data = data[data["name"] == stop]
                    if not stop_data.empty:
                        route_coords.append((stop_data.iloc[0]["lat"], stop_data.iloc[0]["lon"]))
                    else:
                        print(f"Warning: Stop '{stop}' not found in data. Skipping.")

            # Ensure the route has at least two points (for a PolyLine)
            if len(route_coords) < 2:
                print(f"Insufficient points for route: {stops}")
                continue

            # Generate realistic paths using OSRM
            real_path = []
            for i in range(len(route_coords) - 1):
                start = route_coords[i]
                end = route_coords[i + 1]

                try:
                    # Format the coordinates for OSRM API with higher precision
                    coordinates = f"{start[1]},{start[0]};{end[1]},{end[0]}"
                    url = f"{osrm_url}/route/v1/{profile}/{coordinates}?geometries=geojson&steps=true"

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

                except requests.exceptions.RequestException as e:
                    # Log the error and skip this segment
                    print(f"Request error plotting route between {start} and {end}: {e}")
                    continue
                except ValueError as ve:
                    print(ve)
                    continue
                except Exception as ex:
                    print(f"Unexpected error: {ex}")
                    continue

            # Add the realistic path as a polyline to the company's layer
            if real_path:
                PolyLine(
                    real_path,
                    color=company_colors[company],  # Use the unique color for this company
                    weight=2.5,
                    opacity=0.6
                ).add_to(company_group)

        company_group.add_to(map_obj)

    # Iterate through the test dataframe to plot routes for each company
    for company in test['Company'].unique():
        # Filter routes for the current company
        company_routes = test[test['Company'] == company]

        # Create a Folium FeatureGroup for the company
        company_group = folium.FeatureGroup(name=f"{company} Routes", show=True)

        # Plot routes for this company
        plot_routes_for_company(company, company_routes, data, depot_loc, osrm_url, profile, company_colors, company_group)

    # Add layer control to toggle companies on/off
    LayerControl(collapsed=False).add_to(map_obj)

    # Save the map to an HTML file
    map_file = "routes_map.html"
    map_obj.save(map_file)
    print(f"Map saved to {map_file}")

    return map_obj, map_file

