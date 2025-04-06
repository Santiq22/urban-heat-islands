# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from pandas import read_csv
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataIngestionConfig:
    def __init__(self, file_name):
        # Path to output dataset
        self.file_path = os.path.join('../../data/initial_datasets/', file_name)
        
class DataIngestion:
    def __init__(self, area_of_interest, input_data_path, file_name):
        # This variable will consist in the input I need to initialize
        self.ingestion_config = DataIngestionConfig(file_name)
        
        # Limits defining the area of interest
        self.area_of_interest = area_of_interest
        
        # Path to the input dataset
        self.input_data_path = input_data_path
        
    def initiate_data_ingestion(self):
        # Here needs to be the code to read the data (from local, database, etc)
        logging.info("Entered the data ingestion method or component")
        
        try:
            # Read the dataset
            df = read_csv(self.input_data_path)
            
            logging.info("Dataset loaded")
            
            # Select important variables
            df = df[['Centroid Longitude', 'Centroid Latitude', 'Decennial Population Count']]
            
            # Cut the dataset to consider the area of interest
            df = df[df['Centroid Longitude'] >= self.area_of_interest[0][0]]
            df = df[df['Centroid Longitude'] <= self.area_of_interest[0][1]]
            df = df[df['Centroid Latitude'] >= self.area_of_interest[1][0]]
            df = df[df['Centroid Latitude'] <= self.area_of_interest[1][1]]
            
            logging.info("Region of interest correctly selected")
            
            # Rename for consistency
            df.rename(columns = {'Centroid Longitude': "Longitude", 'Centroid Latitude': "Latitude"}, inplace = True)
            
            # Save the dataset
            df.to_csv(self.ingestion_config.file_path, index = False)
            
            logging.info("Dataset saved")
            
            return self.ingestion_config.file_path
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    # Area: ((min_lon, max_lon), (min_lat, max_lat))
    area = ((-74.01, -73.86), (40.75, 40.88))
    input = '../../data/initial_datasets/us_census_blocks_data.csv'
    name = 'population_count_data.csv'
    
    obj = DataIngestion(area, input, name)
    data = obj.initiate_data_ingestion()