# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

# Data Science
import pandas as pd

# Geospatial raster data handling
import rioxarray as rxr

# Coordinate transformations
from pyproj import Proj, Transformer

# Others
from tqdm import tqdm

#from data_transformation import DataTransformation
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataGeneratorConfig:
    # Path to output datasets
    data_path: str = os.path.join('../../data', 'training_dataset.csv')
    
class DataGenerator:
    def __init__(self, sentinel_data, landsat_data, building_data, weather_data):
        # This variable will consist in the input I need to initialize
        self.generator_config = DataGeneratorConfig()
        
        # Path to sentinel .csv data
        self.sentinel_data = sentinel_data
        
        # Path to landsat .csv data
        self.landsat_data = landsat_data
        
        # Path to building footprint .csv data
        self.building_data = building_data
        
        # Path to weather .csv data
        self.weather_data = weather_data
        
    def initiate_data_convertion(self):
        
        logging.info("Entered the data convertion method or component")
        
        try:
            
            
            # Save the DataFrame object as csv
            df.to_csv(self.convertion_config.csv_data_path, index=False)
            
            logging.info("Dataset of indeces and lat/long values correctly saved")
            
            logging.info("Data convertion process finished")
            
            return self.convertion_config.csv_data_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #