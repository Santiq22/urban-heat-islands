# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from numpy import argmin, array, concatenate, pi, float64, sum, where
from pandas import DataFrame, read_csv
from pykml import parser
from shapely.geometry import Point, Polygon
from tqdm import tqdm
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataIngestionConfig:
    def __init__(self, dataset):
        # Path to output datasets
        self.data_path: str = os.path.join('../../data', dataset+'_building_footprint_data.csv')
    
class DataIngestion:
    def __init__(self, input_points, kml_path, radius, threshold, type_of_dataset):
        # This variable will consist in the input I need to initialize
        self.ingestion_config = DataIngestionConfig(type_of_dataset)
        
        # (N, 2) array of (Lon, Lat) points to associate to the given polygons
        self.input_points = input_points*pi/180.0                                         # Radians
        
        # Path to kml data
        self.kml_path = kml_path
        
        # Radius used to compute the building density
        self.radius = radius
        
        # Radius used as threshold to associate a point to a polygon
        self.threshold = threshold
        
    def initiate_data_ingestion(self):
        # Here needs to be the code to read the data (from local, database, etc)
        logging.info("Entered the data ingestion method or component from building footprint data")
        
        try:
            """
            Associate a given (Lon, Lat) to a Polygon object. Its closeness 
            is defined as the closest Polygon's centroid, with threshold the
            value given as input variable. If there's no centroid inside that
            radius, a value of 0 will be returned. 
            
            The idea is to return a (N, 3) array containing the area, 
            perimeter, and building density, respectively, of the Polygon 
            associated with (Lon, Lat).
            -------------------------------------------------------------
            - points: (N, 2) numpy array representing (Lon, Lat) points. 
            - polygons: lenght M list containing all the Polygons objects 
            defined from the Bulding_Footprint.kml file.
            -------------------------------------------------------------
            """
            
            # Parse KML and get Polygon objects
            with open(self.kml_path, 'r') as f:
                doc = parser.parse(f)
            
            logging.info("kml file parsed")
            
            root = doc.getroot()
        
            # Set empty list to store the polygons
            polygons = []
        
            # Iterate over each Placemark and extract Polygon data
            for placemark in root.Document.findall('.//kml:Placemark', 
                                                   namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}):
                
                polygon = placemark.find('.//kml:Polygon', 
                                         namespaces = {'kml': 'http://www.opengis.net/kml/2.2'})
            
                if polygon is not None:
                    # Extract coordinates for the polygon
                    coordinates = polygon.find('.//kml:coordinates', 
                                               namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}).text.strip()
                    
                    # Set empty list to store coordinates
                    coords = []
                    
                    # Convert coordinates into a list of tuples (longitude, latitude)
                    for coord in coordinates.split():
                        lon, lat = map(float, coord.split(','))
                        
                        # Transformation from degrees to radians
                        lon, lat = lon*pi/180.0, lat*pi/180.0                             # Radians
                        
                        # Append longitude and latitude values
                        coords.append((lon, lat))
                    
                    # Create a Shapely polygon object
                    shapely_polygon = Polygon(coords)
                    polygons.append(shapely_polygon)
                    
            logging.info("Polygon data correctly extracted")
            
            # Create Shapely Point objects
            points = [Point(coord) for coord in self.input_points]
            
            logging.info("Shapely Point objects correctly created")
            
            # Areas, perimeters, and densities list
            area_perimeter_density = []
            
            # Earth radius
            earth_radius = 6378000.0                        # Meters
            
            # Check which point is closest to the centroid of a polygon taking into account the 
            # threshold radius
            for point in tqdm(points, total = len(points), desc = "Mapping values"):
                # Compute the distance between the point and the centroids
                d = array([p.centroid.distance(point) for p in polygons])

                # Check which distance is the smallest
                idx_min = argmin(d)
                
                # Number of buildings inside a circle of radius equal to self.radius
                polygon_density = sum(where(earth_radius*d <= self.radius, True, False), dtype = float64)
                
                # Get area and perimeter from the corresponding polygon subject to the threshold
                """ To obtain the perimeter in physical units just multiply by the Earth radius 
                in the desired units. It is due to the small angle aproximation and neglecting 
                the curvature of the working surface. The same argument can be applied to the 
                area. It is important in this case to represent the angles in radians. """
                if earth_radius*d[idx_min] <= self.threshold:
                    polygon_area, polygon_perimeter = polygons[idx_min].area, polygons[idx_min].length
                else:
                    polygon_area, polygon_perimeter = 0.0, 0.0
                
                # Append the area, perimeter, and density values
                area_perimeter_density.append([polygon_area, polygon_perimeter, polygon_density])
                
            # Convert the list of areas, perimeters, and densities into a numpy array
            area_perimeter_density = array(area_perimeter_density)
            
            logging.info("Areas, perimeters, and densities correctly associated to each point")
            
            # Concatenate arrays of points, areas, perimeters, and densities
            dataset = concatenate((self.input_points, area_perimeter_density), axis = 1)
            
            # Convert data to pandas DataFrame object
            dataset = DataFrame(dataset, columns = ['Longitude', 'Latitude', 'polygon_area', 'polygon_perimeter', 'polygon_density'])
            
            logging.info("Data concatenated and transformed to pandas DataFrame")
            
            # Save the dataset
            dataset.to_csv(self.ingestion_config.data_path, index=False)
            
            return self.ingestion_config.data_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    df = read_csv('../../data/Training_data_uhi_index_UHI2025-v2.csv')
    #df = read_csv('../../data/Test_data_uhi_index_UHI2025-v2.csv')
    
    lon_lat = df[['Longitude', 'Latitude']].to_numpy()
    kml_file_path = '../../data/Building_Footprint.kml'
    density_radius = 250.0                                # Meters
    threshold_radius = 100.0                              # Meters
    dataset_type = 'training'
    
    # Instantiate DataIngestion object
    obj = DataIngestion(lon_lat, kml_file_path, density_radius, threshold_radius, dataset_type)
    
    data = obj.initiate_data_ingestion()