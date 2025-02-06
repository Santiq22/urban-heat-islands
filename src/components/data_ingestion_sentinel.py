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
import numpy as np
from dataclasses import dataclass

# Import common GIS tools
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
    raw_data_path: str = os.path.join('../../data', 'raw_data.tiff')
    
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
                query = {"eo:cloud_cover": {"lt": self.cloud_cover_rate}})
            
            logging.info("Items satisfying criteria searched")
            
            # Return all items of this catalog
            items = list(search.get_items())
            
            # Sign items with the Planetary Computer API
            signed_items = [planetary_computer.sign(item).to_dict() for item in items]
            
            logging.info("Items signed with the Planetary Computer API")
            
            # Define the scale according to the selected CRS, so we will use degrees
            scale = self.resolution/111320.0                     # Degrees per pixel for CRS = 4326
            
            # Load several STAC Item objects as an xarray.Dataset
            data = stac_load(
                items,
                bands = self.bands,
                crs = "EPSG:4326",                                             # Latitude-Longitude
                resolution = scale,                                                       # Degrees
                chunks = {"x": 2048, "y": 2048},
                dtype = "uint16",
                patch_url = planetary_computer.sign,
                bbox = bounds)
            
            # Compute the median over the time dimension for the different bands
            median = data.median(dim="time").compute()
            
            logging.info("Median over the time dimension computed for every band in the dataset")
            
            # ------------------------- Computation of different filters --------------------------
            # Calculate NDVI for the median mosaic
            ndvi_median = (median.B08 - median.B04)/(median.B08 + median.B04)
            logging.info("NDVI index computed")
            
            # Calculate gNDVI for the median mosaic
            gndvi_median = (median.B08 - median.B03)/(median.B08 + median.B03)
            logging.info("gNDVI index computed")
            
            # Calculate NDBI for the median mosaic
            ndbi_median = (median.B11 - median.B08)/(median.B11 + median.B08)
            logging.info("NDBI index computed")
            
            # Calculate NDWI for the median mosaic
            ndwi_median = (median.B03 - median.B08)/(median.B03 + median.B08)
            logging.info("NDWI index computed")
            
            # Calculate BWDRVI for the median mosaic
            bwdrvi_median = (0.1*median.B08 - median.B02)/(0.1*median.B08 + median.B02)
            logging.info("BWDRVI index computed")
            
            # Calculate CCCI for the median mosaic
            ccci_median = (median.B08 - median.B05)/(median.B08 + median.B05)*(median.B08 + median.B04)/(median.B08 - median.B04)
            # Replace infs and -infs by NaNs
            ccci_median = ccci_median.where(ccci_median.values != -np.inf)
            ccci_median = ccci_median.where(ccci_median.values != np.inf)
            # Replace NaNs by 0.0
            ccci_median.fillna(0.0)
            logging.info("CCCI index computed")
            
            # Calculate CTVI for the median mosaic
            ctvi_median = (ndvi_median + 0.5)/abs(ndvi_median + 0.5)*np.sqrt(abs(ndvi_median + 0.5))
            logging.info("CTVI index computed")
            
            # Calculate Datt1 for the median mosaic
            datt1_median = (median.B08 - median.B05)/(median.B08 - median.B04)
            # Replace infs and -infs by NaNs
            datt1_median = datt1_median.where(datt1_median.values != -np.inf)
            datt1_median = datt1_median.where(datt1_median.values != np.inf)
            # Replace NaNs by 0.0
            datt1_median.fillna(0.0)
            logging.info("Datt1 index computed")
            
            # Calculate Fe2+ for the median mosaic
            fe2_median = median.B12/median.B08 + median.B03/median.B04
            logging.info("Fe2+ index computed")
            
            # Calculate Ferric Oxides for the median mosaic
            fo_median = median.B11/median.B08
            logging.info("Ferric Oxides index computed")
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
            ndvi_median.rio.write_crs("epsg:4326", inplace = True)
            gndvi_median.rio.write_crs("epsg:4326", inplace = True)
            ndbi_median.rio.write_crs("epsg:4326", inplace = True)
            ndwi_median.rio.write_crs("epsg:4326", inplace = True)
            bwdrvi_median.rio.write_crs("epsg:4326", inplace = True)
            ccci_median.rio.write_crs("epsg:4326", inplace = True)
            ctvi_median.rio.write_crs("epsg:4326", inplace = True)
            datt1_median.rio.write_crs("epsg:4326", inplace = True)
            fe2_median.rio.write_crs("epsg:4326", inplace = True)
            fo_median.rio.write_crs("epsg:4326", inplace = True)
            
            # Write the GeoTransform to the dataset where GDAL can read it in. It returns a modified 
            # dataset with GeoTransform written.
            ndvi_median.rio.write_transform(transform = gt, inplace = True)
            gndvi_median.rio.write_transform(transform = gt, inplace = True)
            ndbi_median.rio.write_transform(transform = gt, inplace = True)
            ndwi_median.rio.write_transform(transform = gt, inplace = True)
            bwdrvi_median.rio.write_transform(transform = gt, inplace = True)
            ccci_median.rio.write_transform(transform = gt, inplace = True)
            ctvi_median.rio.write_transform(transform = gt, inplace = True)
            datt1_median.rio.write_transform(transform = gt, inplace = True)
            fe2_median.rio.write_transform(transform = gt, inplace = True)
            fo_median.rio.write_transform(transform = gt, inplace = True)
            
            logging.info("Transformation to the EPSG:4326 CRS finished")
            
            # Create the GeoTIFF output file using the defined parameters 
            with rasterio.open(self.ingestion_config.raw_data_path, 'w', driver = 'GTiff', width = width, 
                               height = height, crs = 'epsg:4326', transform = gt, count = &&&&&&, 
                               compress = 'lzw', dtype = 'float64') as dst:
                # Save the raw data in its path
                dst.write(ndvi_median, 1)
                dst.write(gndvi_median, 2)
                dst.write(ndbi_median, 3)
                dst.write(ndwi_median, 4)
                dst.write(bwdrvi_median, 5)
                dst.write(ccci_median, 6)
                dst.write(ctvi_median, 7)
                dst.write(datt1_median, 8)
                dst.write(fe2_median, 9)
                dst.write(fo_median, 10)
                dst.write(ndwi_median, 4)
                dst.write(ndwi_median, 4)
                dst.write(ndwi_median, 4)
                dst.write(ndwi_median, 4)
                dst.write(ndwi_median, 4)
                dst.write(ndwi_median, 4)
                dst.write(ndwi_median, 4)
                dst.write(ndwi_median, 4)
                dst.write(ndwi_median, 4)
                dst.write(ndwi_median, 4)
                dst.write(ndwi_median, 4)
                dst.write(ndwi_median, 4)
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
    coll = "sentinel-2-l2a"
    clouds = 30.0
    res = 10.0
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    
    obj = DataIngestion(l_l, u_r, t_w, coll, clouds, res, bands)
    raw_data = obj.initiate_data_ingestion()
    
    #data_transformation = DataTransformation()
    #_ = data_transformation.initiate_data_transformation(raw_data)