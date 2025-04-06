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
import pandas as pd
from dataclasses import dataclass

# Import common GIS tools
import numpy as np
import xarray as xr
import rioxarray as rio
import rasterio

# Import Planetary Computer tools
import stackstac
import pystac_client
import planetary_computer 
from odc.stac import stac_load

#from data_transformation import DataTransformation
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataIngestionConfig:
    # Path to output datasets
    raw_data_path: str = os.path.join('../../data', 'raw_landsat_data.tiff')
    
class DataIngestion:
    def __init__(self, lower_left, upper_right, time_window, collection, cloud_cover_rate,
                 resolution, bands):
        # This variable will consist in the input I need to initialize
        self.ingestion_config = DataIngestionConfig()
        
        # Lower-left (Latitude, Longitude) point
        self.lower_left = lower_left                                                      # Degrees
        
        # Upper-right (Latitude, Longitude) point
        self.upper_right = upper_right                                                    # Degrees
        
        # Time window where to perform the search
        self.time_window = time_window
        
        # Collection to search for
        self.collection = collection
        
        # Upper limit of the percentage of cloud cover in the given scenes
        self.cloud_cover_rate = cloud_cover_rate
        
        # Pixel resolution for the final product
        self.resolution = resolution                                             # Meters per pixel
        
        # Bands to do the searching
        self.bands = bands
        
    def initiate_data_ingestion(self):
        # Here needs to be the code to read the data (from local, database, etc)
        logging.info("Entered the data ingestion method or component")
        
        try:
            # Calculate the bounds for doing an archive data search
            # bounds = (min_lon, min_lat, max_lon, max_lat)
            bounds = (self.lower_left[1], self.lower_left[0], self.upper_right[1], self.upper_right[0])
            
            # Opens a STAC Catalog or API. This function will read the root catalog of a STAC Catalog
            # or API. It returns a Client instance for this Catalog/API.
            stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            
            logging.info("Client opened")

            # Search the Planetary Computer's STAC endpoint for items matching the query parameters.
            # It returns an ItemSearch instance that can be used to iterate through Items.
            search = stac.search(
                bbox = bounds, 
                datetime = self.time_window,
                collections = [self.collection],
                query = {"eo:cloud_cover": {"lt": self.cloud_cover_rate},"platform": {"in": ["landsat-8"]}})
            
            logging.info("Items satisfying criteria searched")
            
            # Return all items of this catalog
            items = list(search.get_items())
            
            # Sign items with the Planetary Computer API
            signed_items = [planetary_computer.sign(item).to_dict() for item in items]
            
            logging.info("Items signed with the Planetary Computer API")
            
            # Define the scale according to the selected CRS, so we will use degrees
            scale = self.resolution/111320.0                     # Degrees per pixel for CRS = 4326
            
            # Load several STAC Item objects as an xarray.Dataset
            # We will use only the Surface Temperature data
            data = stac_load(
                items,
                bands = self.bands,
                crs = "EPSG:4326",                                             # Latitude-Longitude
                resolution = scale,                                                       # Degrees
                chunks = {"x": 2048, "y": 2048},
                dtype = "uint16",
                patch_url = planetary_computer.sign,
                bbox = bounds)
            
            # Scale factors for the Surface Temperature band
            scale_lst = 0.00341802
            offset = 149.0 
            kelvin_celsius = 273.15                                # Convert from Kelvin to Celsius
            data = data.astype(float)*scale_lst + offset - kelvin_celsius
            
            # Compute the median over the time dimension for the different bands
            median = data.median(dim = "time").compute()
            
            logging.info("Median over the time dimension computed for every band in the dataset")
            
            # -------------------------- Computation of different bands ---------------------------
            # Get the Surface Temperature band for the median mosaic
            lst_median = median.lwir11
            
            logging.info("LST band loaded")
            # -------------------------------------------------------------------------------------            
            
            # Calculate the dimensions of the output file
            height = median.dims["latitude"]
            width = median.dims["longitude"]
            
            # Define the Coordinate Reference System (CRS) to be common Lat-Lon coordinates
            # Define the tranformation using the bounding box so the Lat-Lon information is written
            # to the GeoTIFF
            gt = rasterio.transform.from_bounds(self.lower_left[1], self.lower_left[0], self.upper_right[1], 
                                                self.upper_right[0], width, height)
            
            # Write the CRS to the dataset in a CF compliant manner. It returns a modified dataset 
            # with CF compliant CRS information.
            lst_median.rio.write_crs("epsg:4326", inplace = True)
            
            # Write the GeoTransform to the dataset where GDAL can read it in. It returns a modified 
            # dataset with GeoTransform written.
            lst_median.rio.write_transform(transform = gt, inplace = True)
            
            logging.info("Transformation to the EPSG:4326 CRS finished")            
            
            # Create the GeoTIFF output file using the defined parameters 
            with rasterio.open(self.ingestion_config.raw_data_path, 'w', driver = 'GTiff', width = width, 
                               height = height, crs = 'epsg:4326', transform = gt, count = 1, 
                               compress = 'lzw', dtype = 'float64') as dst:
                # Save the raw data in its path
                dst.write(lst_median, 1)
                dst.close()
            
            logging.info("Raw data saved as .tiff files")
            
            logging.info("Ingestion of the data completed")
            
            return self.ingestion_config.raw_data_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    l_l = (40.75, -74.01)
    u_r = (40.88, -73.86)
    t_w = "2021-06-01/2021-09-01"
    coll = "landsat-c2-l2"
    clouds = 50.0                                                                      # Percentage
    res = 30.0                                                                   # Meters per pixel
    bands = ["lwir11"]
    
    obj = DataIngestion(l_l, u_r, t_w, coll, clouds, res, bands)
    raw_data = obj.initiate_data_ingestion()