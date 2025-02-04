from numpy import argmin, array
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

def associate_point_to_polygon(input_points, polygons):
    """
    Function to associate a given (Lon, Lat) to a Polygon object. 
    Its closeness is defined as the closest Polygon's centroid. The 
    function returns a (N, 2) array containing the area and perimeter, 
    respectively, of the Polygon associated with (Lon, Lat).
    ------------------------------------------------------------
    - points: (N, 2) numpy array representing (Lon, Lat) points. 
    - polygons: (M, 2) list containing all the Polygons objects 
    defined from the Bulding_Footprint.kml file.
    ------------------------------------------------------------
    """
    
    # Create Shapely Point objects
    points = [Point(coord) for coord in input_points]
    
    # Areas and perimeters list
    area_and_perimeter = []
    
    # Check which point is closest to the centroid of a polygon
    for point in points:
        # Compute the distance between the point and the centroids
        d = array([p.centroid.distance(point) for p in polygons])

        # Check which distance is the smallest
        idx_min = argmin(d)
        
        # Get area and perimeter from the corresponding polygon
        polygon_area, polygon_perimeter = polygons[idx_min].area, polygons[idx_min].length
        
        # Append the area and perimeter values
        area_and_perimeter.append([polygon_area, polygon_perimeter])
        
    # Convert the list of areas and perimeters into a numpy array
    area_and_perimeter = array(area_and_perimeter)
    
    return area_and_perimeter


kml_file = '../../data/Building_Footprint.kml'

idx = 100
p = parse_kml(kml_file)
print(len(p), "\n")
print(p[idx], "\n")
#print(p[idx].is_closed, "\n")
print(p[idx].area, "\n")
#print(p[idx].length, "\n")
#print(p[idx].point_on_surface(), "\n")
#for c in p[idx].exterior.coords:
#    print(c)
print(p[idx].centroid)
print(p[idx:idx + 3].centroid.distance(p[idx + 5].centroid))