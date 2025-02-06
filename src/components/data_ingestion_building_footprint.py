# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from numpy import argmin, array, concatenate
from pandas import DataFrame, read_csv
from pykml import parser
from shapely.geometry import Point, Polygon
import xml.etree.ElementTree as ET
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataIngestionConfig:
    # Path to output datasets
    data_path: str = os.path.join('../../data', 'building_footprint_data.csv')
    
class DataIngestion:
    def __init__(self, input_points):
        # This variable will consist in the input I need to initialize
        self.ingestion_config = DataIngestionConfig()
        
        # (N, 2) array of (Lon, Lat) points to associate to the given polygons
        self.input_points = input_points
        
    def initiate_data_ingestion(self):
        # Here needs to be the code to read the data (from local, database, etc)
        logging.info("Entered the data ingestion method or component from building footprint data")
        
        try:
            """
            Associate a given (Lon, Lat) to a Polygon object. Its closeness 
            is defined as the closest Polygon's centroid. The idea is 
            to return a (N, 2) array containing the area and perimeter, 
            respectively, of the Polygon associated with (Lon, Lat).
            ------------------------------------------------------------
            - points: (N, 2) numpy array representing (Lon, Lat) points. 
            - polygons: (M, 2) list containing all the Polygons objects 
            defined from the Bulding_Footprint.kml file.
            ------------------------------------------------------------
            """
            
            # Parse KML and get Polygon objects
            with open('../../data/Building_Footprint.kml', 'r') as f:
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
                    coords = []
                    
                    # Convert coordinates into a list of tuples (longitude, latitude)
                    for coord in coordinates.split():
                        lon, lat = map(float, coord.split(','))
                        """ Here it goes the transformation from degrees to radians. """
                        coords.append((lon, lat))
                    
                    # Create a Shapely polygon object
                    shapely_polygon = Polygon(coords)
                    polygons.append(shapely_polygon)
                    
            logging.info("Polygon data correctly extracted")
            
            # Create Shapely Point objects
            points = [Point(coord) for coord in self.input_points]
            
            logging.info("Shapely Point objects correctly created")
            
            # Areas and perimeters list
            area_and_perimeter = []
            
            # Check which point is closest to the centroid of a polygon
            for i, point in enumerate(points):
                # Compute the distance between the point and the centroids
                d = array([p.centroid.distance(point) for p in polygons])

                # Check which distance is the smallest
                idx_min = argmin(d)
                
                # Get area and perimeter from the corresponding polygon
                """ To obtain the perimeter in physical units just multiply by the Earth radius 
                in the desired units. It is due to the small angle aproximation and neglecting 
                the curvature of the working surface. The same argument can be applied to the 
                area. It is important in this case to represent the angles in radians. """
                polygon_area, polygon_perimeter = polygons[idx_min].area, polygons[idx_min].length
                
                # Append the area and perimeter values
                area_and_perimeter.append([polygon_area, polygon_perimeter])
                
                print("{} th area and perimeter computed".format(i))
                
            # Convert the list of areas and perimeters into a numpy array
            area_and_perimeter = array(area_and_perimeter)
            
            logging.info("Areas and perimeters correctly associated to each point")
            
            # Concatenate arrays of points, areas, and perimeters
            dataset = concatenate((self.input_points, area_and_perimeter), axis = 1)
            
            # Convert data to pandas dataframe
            dataset = DataFrame(dataset, columns = ['Longitude', 'Latitude', 'polygon_area', 'polygon_perimeter'])
            
            logging.info("Data concatenated and transformed to pandas dataframe")
            
            # Save the dataset
            dataset.to_csv(self.ingestion_config.data_path)
            
            return self.ingestion_config.data_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    df = read_csv('../../data/Training_data_uhi_index_UHI2025-v2.csv')
    
    lon_lat = df[['Longitude', 'Latitude']].to_numpy()
    
    # Instantiate DataIngestion object
    obj = DataIngestion(lon_lat)
    
    data = obj.initiate_data_ingestion()
    
    #data_transformation = DataTransformation()
    #_ = data_transformation.initiate_data_transformation(raw_data)