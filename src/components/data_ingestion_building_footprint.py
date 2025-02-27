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

# ===================================== External functions ====================================== #
def assign_hag(polygon_centroid, list_of_hag_points, array_of_hag_values):
    """ Assigns a HAG value to the given polygon """
    
    # Compute the distance between the polygon centroid and the list of points
    d = array([p.centroid.distance(polygon_centroid) for p in list_of_hag_points])

    # Check which distance is the smallest
    idx_min = argmin(d)
    
    # HAG index corresponding to the closest polygon
    hag = array_of_hag_values[idx_min]
    return hag
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataIngestionConfig:
    def __init__(self, dataset):
        # Path to output datasets
        self.data_path: str = os.path.join('../../data/initial_datasets/', dataset+'_building_footprint_data.csv')
    
class DataIngestion:
    def __init__(self, input_points, hag_path, pluto_path, census_path, kml_path, radius, threshold, type_of_dataset):
        # This variable will consist in the input I need to initialize
        self.ingestion_config = DataIngestionConfig(type_of_dataset)
        
        # (N, 2) array of (Lon, Lat) points to associate to the given polygons
        self.input_points = input_points*pi/180.0                                         # Radians
        self.input_points_degrees = input_points                                           # Degrees
        
        # Path to the dataset storing the HAG indeces
        self.hag_path = hag_path
        
        # Path to the dataset containing pluto data
        self.pluto_path = pluto_path
        
        # Path to the dataset containing census data
        self.census_path = census_path
        
        # Path to kml data
        self.kml_path = kml_path
        
        # Radius used to compute the building and population density
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
            
            The idea is to return a (N, 6) array containing the area, 
            perimeter, building density, floors on a building, number of
            units on a building, and the population density respectively, of
            the input points (Lon, Lat).
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
            
            # Create Shapely Point objects over the input (Lon, Lat) points
            points = [Point(coord) for coord in self.input_points]
            
            logging.info("Shapely Point objects correctly created")
            
            # --------------------------------- Load the datasets ---------------------------------
            #df_hag = read_csv(self.hag_path)
            df_pluto = read_csv(self.pluto_path)
            df_census = read_csv(self.census_path)
            # -------------------------------------------------------------------------------------
            
            # ------------ Create lists of Shapely Point objects for the above datasets -----------
            #hag_points = [Point(coord) for coord in df_hag[['Longitude', 'Latitude']].values]
            pluto_points = [Point(coord) for coord in df_pluto[['Longitude', 'Latitude']].values*pi/180.0]
            census_points = [Point(coord) for coord in df_census[['Longitude', 'Latitude']].values*pi/180.0]
            # -------------------------------------------------------------------------------------
            
            # Areas, perimeters, densities, floors, units
            output_data = []
            
            # Earth radius
            earth_radius = 6378000.0                        # Meters
            
            # Check which point is closest to the centroid of a polygon taking into account the 
            # threshold radius
            for point in tqdm(points, total = len(points), desc = "Mapping values"):
                # ---------------------------- Building footprint data ----------------------------
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
                    # Area and perimeter
                    polygon_area, polygon_perimeter = polygons[idx_min].area, polygons[idx_min].length
                    
                    # HAG
                    #polygon_hag = assign_hag(polygons[idx_min].centroid, hag_points, df_hag['HAG'].values)
                else:
                    # Area and perimeter
                    polygon_area, polygon_perimeter = 0.0, 0.0
                    
                    # HAG
                    #polygon_hag = 0.0
                # ---------------------------------------------------------------------------------
                
                # ----------------------------------- Pluto data ----------------------------------
                # Compute the distance between the point and the points in the pluto dataset
                d = array([p.distance(point) for p in pluto_points])
                
                # Check distances smallers than self.radius
                idx = where(earth_radius*d <= self.radius, True, False)
                
                # Mean number of floors inside self.radius
                mean_number_of_floors = df_pluto['numfloors'].values[idx].mean()
                
                # Total number of units inside self.radius
                units_density = sum(df_pluto['unitstotal'].values[idx], dtype = float64)
                # ---------------------------------------------------------------------------------
                
                # ---------------------------------- Census data ----------------------------------
                # Compute the distance between the point and the block centroids
                d = array([p.distance(point) for p in census_points])

                # Check distances smallers than self.radius
                idx = where(earth_radius*d <= self.radius, True, False)
                
                # Population inside a circle of radius equal to self.radius
                population_density = sum(df_census['Decennial Population Count'].values[idx], dtype = float64)
                # ---------------------------------------------------------------------------------
                
                # Append the area, perimeter, density values, floors, and units
                output_data.append([polygon_area, polygon_perimeter, polygon_density, 
                                    mean_number_of_floors, units_density, population_density])
                
            # Convert the list above into a numpy array
            output_data = array(output_data)
            
            logging.info("Areas, perimeters, densities, and floors correctly associated to each point")
            
            # Concatenate arrays
            dataset = concatenate((self.input_points_degrees, output_data), axis = 1)
            
            # Convert data to pandas DataFrame object
            dataset = DataFrame(dataset, columns = ['Longitude', 
                                                    'Latitude', 
                                                    'polygon_area', 
                                                    'polygon_perimeter', 
                                                    'polygon_density_'+str(self.radius),
                                                    'mean_number_of_floors_'+str(self.radius),
                                                    'units_density_'+str(self.radius),
                                                    'population_density_'+str(self.radius)])
            
            logging.info("Data concatenated and transformed to pandas DataFrame")
            
            # Save the dataset
            dataset.to_csv(self.ingestion_config.data_path, index=False)
            
            return self.ingestion_config.data_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    df = read_csv('../../data/initial_datasets/Training_data_uhi_index_2025-02-18.csv')
    #df = read_csv('../../data/initial_datasets/Test_data_uhi_index_UHI2025-v2.csv')
    
    lon_lat = df[['Longitude', 'Latitude']].values
    path_to_hag_data = '../../data/initial_datasets/hag_data.csv'
    path_to_pluto_data = '../../data/initial_datasets/pluto_data.csv'
    path_to_census_data = '../../data/initial_datasets/population_count_data.csv'
    kml_file_path = '../../data/initial_datasets/Building_Footprint.kml'
    density_radius = 250                                  # Meters
    threshold_radius = 100.0                              # Meters
    dataset_type = 'training'
    #dataset_type = 'test'
    
    # Instantiate DataIngestion object
    obj = DataIngestion(lon_lat, 
                        path_to_hag_data, 
                        path_to_pluto_data, 
                        path_to_census_data,
                        kml_file_path, 
                        density_radius, 
                        threshold_radius, 
                        dataset_type)
    
    data = obj.initiate_data_ingestion()