from pykml import parser
from shapely.geometry import Point, Polygon
import xml.etree.ElementTree as ET

# Function to parse KML and get Polygon objects
def parse_kml(kml_file):
    with open(kml_file, 'r') as f:
        doc = parser.parse(f)
    root = doc.getroot()
    polygons = []
    
    # Iterate over each Placemark and extract Polygon data
    for placemark in root.Document.findall('.//kml:Placemark', namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}):
        polygon = placemark.find('.//kml:Polygon', namespaces = {'kml': 'http://www.opengis.net/kml/2.2'})
        
        if polygon is not None:
            # Extract coordinates for the polygon
            coordinates = polygon.find('.//kml:coordinates', namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}).text.strip()
            coords = []
            
            # Convert coordinates into a list of tuples (longitude, latitude)
            for coord in coordinates.split():
                lon, lat = map(float, coord.split(','))
                coords.append((lon, lat))
            
            # Create a Shapely polygon object
            shapely_polygon = Polygon(coords)
            polygons.append(shapely_polygon)
    
    return polygons

"""# Function to check if point is inside any polygon
def check_point_in_polygon(point, polygons):
    # Create a Shapely Point object
    point = Point(point)  # point = (longitude, latitude)
    
    # Check if the point lies within any polygon
    for i, polygon in enumerate(polygons):
        if polygon.contains(point):
            print(f"Point is inside Polygon {i+1}")
            return i + 1  # Return the index of the polygon
    print("Point is outside all polygons")
    return None

# Example usage
kml_file = 'your_file.kml'  # Replace with your .kml file path
latitude = 37.7749          # Replace with your latitude
longitude = -122.4194       # Replace with your longitude

# Parse the KML file and extract polygons
polygons = parse_kml(kml_file)

# Check if the point (longitude, latitude) is inside any polygon
check_point_in_polygon((longitude, latitude), polygons)"""


kml_file = '../../data/Building_Footprint.kml'

idx = 100
p = parse_kml(kml_file)
print(len(p), "\n")
print(p[idx], "\n")
print(p[idx].is_closed, "\n")
print(p[idx].area, "\n")
print(p[idx].length, "\n")
print(p[idx].point_on_surface(), "\n")