# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from pandas import read_csv
from numpy import argmax, unique
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataIngestionConfig:
    def __init__(self, file_name):
        # Path to output dataset
        self.file_path = os.path.join('../../data/initial_datasets/', file_name)
        
class DataIngestion:
    def __init__(self, input_data_path, file_name):
        # This variable will consist in the input I need to initialize
        self.ingestion_config = DataIngestionConfig(file_name)
        
        # Path to the input dataset
        self.input_data_path = input_data_path
        
    def initiate_data_ingestion(self):
        # Here needs to be the code to read the data (from local, database, etc)
        logging.info("Entered the data ingestion method or component")
        
        try:
            # Read the dataset
            df = read_csv(self.input_data_path)
            
            # Drop the NaN values of the latitude and longitude columns
            df.dropna(subset = ['latitude', 'longitude'], inplace = True)
            
            # Get index of the mode of the unitstotal column
            idx_mode = argmax(unique(df['unitstotal'].values, return_counts = True)[1])
            
            # Compute the mode of the unitstotal column
            mode = unique(df['unitstotal'].values)[idx_mode]
            
            logging.info("Mode computed")
            
            # Fill NaNs with predetermined values
            df.fillna(value = {'numfloors': df['numfloors'].dropna().values.mean(), 'unitstotal': mode}, inplace = True)
            
            logging.info("NaNs replaced")
            
            # Rename columns for consistency
            df.rename(columns={"longitude": "Longitude", "latitude": "Latitude"}, inplace = True)
            
            # Save the new dataset
            df.to_csv(self.ingestion_config.file_path, index = False)
            
            logging.info("Dataset saved")
            
            return self.ingestion_config.file_path
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    input = '../../data/initial_datasets/pluto_24v4_1_clean.csv'
    name = 'pluto_data.csv'
    
    obj = DataIngestion(input, name)
    data = obj.initiate_data_ingestion()