# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:16:53 2024

@author: Santiago Collazo
@original_author: Krish Naik
"""
# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

# Data Science
from pandas import read_csv, DataFrame

# Geospatial raster data handling
import rioxarray as rxr

# Coordinate transformations
from pyproj import Proj, Transformer

# Others
from tqdm import tqdm
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataConvertionConfig:
    def __init__(self, dataset):    
        # Path to output dataset
        self.csv_data_path: str = os.path.join('../../data/initial_datasets/', dataset+'_landsat_data.csv')
    
class DataConvertion:
    def __init__(self, tiff_path, csv_path, type_of_dataset):
        # This variable will consist in the input I need to initialize
        self.convertion_config = DataConvertionConfig(type_of_dataset)
        
        # Path to raw .tiff data
        self.tiff_path = tiff_path
        
        # Path to base training .csv or test .csv data
        self.csv_path = csv_path
        
    def initiate_data_convertion(self):
        # Extracts satellite band values from a GeoTIFF based on coordinates from a csv file and 
        # returns them in a DataFrame object
        logging.info("Entered the data convertion method or component")
        
        try:
            # Load the GeoTIFF data as a xarray.Dataset
            data = rxr.open_rasterio(self.tiff_path)
            tiff_crs = data.rio.crs
            
            logging.info("Data loaded as xarray.Dataset")

            # Read the csv file using pandas
            df = read_csv(self.csv_path)
            latitudes = df['Latitude'].values
            longitudes = df['Longitude'].values
            
            logging.info("Dataset loaded")

            # Convert latitudes/longitudes to the GeoTIFF's CRS
            # Create a Proj object for EPSG:4326 (WGS84 - lat/long) and the GeoTIFF's CRS
            proj_wgs84 = Proj(init = 'epsg:4326')            # EPSG:4326 is the common lat/long CRS
            proj_tiff = Proj(tiff_crs)
            
            logging.info("Proj objects instantiated")
            
            # Create a transformer object
            transformer = Transformer.from_proj(proj_wgs84, proj_tiff)
            
            logging.info("Transformer object created")

            # Empty lists to store the values of the indeces
            lst_median = []

            # Iterate over the latitudes and longitudes, and extract the corresponding indeces values
            for lat, lon in tqdm(zip(latitudes, longitudes), total = len(latitudes), desc = "Mapping values"):
            # Assuming the correct dimensions are 'y' and 'x' (replace these with actual names 
            # from data.coords)
            
                lst = data.sel(x = lon, y = lat, band = 1, method = "nearest").values
                lst_median.append(lst)
                
            logging.info("Indeces correctly loaded")
            
            # Create a DataFrame to store the band values
            df_out = DataFrame()
            df_out['Latitude'] = latitudes
            df_out['Longitude'] = longitudes
            df_out['lst_median_res10'] = lst_median
            
            logging.info("Indeces converted to DataFrame columns")
            
            # Save the DataFrame object as csv
            df_out.to_csv(self.convertion_config.csv_data_path, index=False)
            
            logging.info("Dataset of indeces and lat/long values correctly saved")
            
            logging.info("Data convertion process finished")
            
            return self.convertion_config.csv_data_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    path_to_tiff = '../../data/initial_datasets/raw_landsat_data.tiff'
    path_to_csv = '../../data/initial_datasets/Training_data_uhi_index_2025-02-18.csv'
    #path_to_csv = '../../data/Test_data_uhi_index_UHI2025-v2.csv'
    #dataset_type = 'test'
    dataset_type = 'training'
    
    obj = DataConvertion(path_to_tiff, path_to_csv, dataset_type)
    csv_data = obj.initiate_data_convertion()