# import folium
# from folium import Marker, PolyLine
# from folium.features import DivIcon
# from folium.map import LayerControl
# import itertools
# import requests

# def generate_route_map_fixed_with_legend(
#     depot_loc,
#     data,
#     test,
#     osrm_url="http://router.project-osrm.org",
#     profile="driving"
# ):
#     """
#     Generate a Folium map of routes using an OSRM server for realistic polylines,
#     labeling each delivery stop with its index (order of visitation).

#     Parameters
#     ----------
#     depot_loc : tuple (lat, lon)
#         The latitude and longitude of the depot.
#     data : pd.DataFrame
#         Must contain columns ['lat','lon'].
#         The row index in 'data' must align with the integer stops in 'test'.
#     test : pd.DataFrame
#         Must contain:
#           - 'Company': a str identifying which company/pair this row belongs to.
#           - 'Routes': a dict {vehicle_id: [list_of_stops]} where each stop
#             is an integer (index in 'data') or a string like "Source"/"Sink".
#     osrm_url : str
#         Base URL of your OSRM server, e.g. "http://localhost:5000".
#     profile : str
#         The OSRM profile to use (e.g. "driving", "cycling").

#     Returns
#     -------
#     map_obj : folium.Map
#         The generated Folium map object.
#     map_file : str
#         The name of the saved HTML file containing this map.
#     """

#     # Create a base map centered on the average lat/lon of all data points
#     center_lat = data["lat"].mean()
#     center_lon = data["lon"].mean()
#     map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=8)

#     # Ensure there's a 'Company' column
#     if "Company" not in test.columns:
#         test["Company"] = "NoCompanyName"

#     # We'll group routes by company, so each company has a FeatureGroup
#     for _, row in test.iterrows():
#         company_name = row["Company"]
#         routes_dict = row.get("Routes", {})

#         # Each company's set of routes goes in one FeatureGroup
#         fg = folium.FeatureGroup(name=f"{company_name} Routes", show=True)

#         # Mark the depot if available
        

#         # Create a distinct color cycle for each vehicle route
#         route_colors = itertools.cycle([
#             "blue", "green", "purple", "orange", "darkred",
#             "lightred", "beige", "darkblue", "darkgreen", "cadetblue",
#             "pink", "lightblue", "lightgreen", "gray", "black"
#         ])

#         # For each vehicle route in "Routes"
#         for vehicle_id, stops_list in routes_dict.items():
#             route_color = next(route_colors)
#             if depot_loc:Marker(
#                 location=depot_loc,
#                 popup=f"{company_name} - Depot",
#                 icon=folium.Icon(color="red", icon="home", prefix="fa")
#             ).add_to(fg)

#             # Filter out non-integer entries ("Source"/"Sink")
#             numeric_stops = [s for s in stops_list if isinstance(s, int)]

#             # Convert these numeric stops to lat/lon using 'data'
#             route_coords = []
#             for stop_idx in numeric_stops:
#                 if 0 <= stop_idx < len(data):
#                     row_data = data.iloc[stop_idx]
#                     route_coords.append((row_data["lat"], row_data["lon"]))

#             # If we want to start/end at the depot
#             if depot_loc and route_coords:
#                 route_coords = [depot_loc] + route_coords + [depot_loc]

#             # Build the OSRM path
#             real_path = []
#             for i in range(len(route_coords) - 1):
#                 start_lat, start_lon = route_coords[i]
#                 end_lat, end_lon     = route_coords[i + 1]

#                 # OSRM route call: 'lon,lat;lon,lat'
#                 coords_str = f"{start_lon},{start_lat};{end_lon},{end_lat}"
#                 url = f"{osrm_url}/route/v1/{profile}/{coords_str}?geometries=geojson&steps=true"

#                 try:
#                     resp = requests.get(url)
#                     resp.raise_for_status()
#                     osrm_data = resp.json()

#                     if "routes" in osrm_data and osrm_data["routes"]:
#                         geometry = osrm_data["routes"][0]["geometry"]["coordinates"]
#                         # geometry is [[lon, lat], [lon, lat], ...]
#                         for (lon_, lat_) in geometry:
#                             real_path.append((lat_, lon_))
#                 except Exception as e:
#                     print(f"OSRM error building path from {start_lat},{start_lon} to {end_lat},{end_lon}: {e}")
#                     continue

#             # Draw the OSRM polyline
#             if real_path:
#                 PolyLine(
#                     locations=real_path,
#                     color=route_color,
#                     weight=3,
#                     opacity=0.5
#                 ).add_to(fg)

#             # Now place a numbered marker at each stop in the route, showing the order
#             for stop_order, (lat_, lon_) in enumerate(route_coords):
#                 # We'll label them 1-based; if you prefer 0-based, use stop_order directly
#                 label_number = stop_order + 1

#                 # We'll use Folium's DivIcon to create a small text label
#                 folium.map.Marker(
#                     [lat_, lon_],
#                     icon=folium.features.DivIcon(
#                         icon_size=(30, 30),
#                         icon_anchor=(15, 15),
#                         html=(
#                             f"<div style='font-size:12pt;"
#                             f"color:black;"
#                             f"background-color:rgba(255,255,255,0.7);"
#                             f"border:1px solid black;"
#                             f"border-radius:15px;"
#                             f"width:30px;height:30px;"
#                             f"text-align:center;line-height:30px;'>"
#                             f"{label_number}</div>"
#                         )
#                     ),
#                     popup=f"Stop {label_number}"
#                 ).add_to(fg)

#         # Add FeatureGroup
#         fg.add_to(map_obj)

#     # Enable toggling layers
#     LayerControl(collapsed=False).add_to(map_obj)

#     # Save map to HTML
#     map_file = "routes_map.html"
#     map_obj.save(map_file)
#     print(f"Map saved to {map_file}")

#     return map_obj, map_file
