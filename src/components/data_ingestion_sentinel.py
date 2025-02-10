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
    raw_data_path: str = os.path.join('../../data', 'raw_sentinel_data.tiff')
    
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
            
            # Calculate Ferrous Silicates for the median mosaic
            fs_median = median.B12/median.B11
            logging.info("Ferric Silicates index computed")
            
            # Calculate mSR for the median mosaic
            msr_median = (median.B08 - median.B01)/(median.B04 - median.B01)
            # Replace infs and -infs by NaNs
            msr_median = msr_median.where(msr_median.values != -np.inf)
            msr_median = msr_median.where(msr_median.values != np.inf)
            # Replace NaNs by 0.0
            msr_median.fillna(0.0)
            logging.info("mSR index computed")
            
            # Calculate MSAVI for the median mosaic
            msavi_median = 0.5*(2.0*median.B08 + 1.0 - np.sqrt((2.0*median.B08 + 1.0)**2.0 - 8.0*(median.B08 - median.B04)))
            logging.info("MSAVI index computed")
            
            # Calculate PVR for the median mosaic
            pvr_median = (median.B03 - median.B04)/(median.B03 + median.B04)
            logging.info("PVR index computed")
            
            # Calculate PSNDc2 for the median mosaic
            psndc2_median = (median.B08 - median.B02)/(median.B08 + median.B02)
            logging.info("PSNDc2 index computed")
            
            # Calculate SIWSI for the median mosaic
            siwsi_median = (median.B8A - median.B11)/(median.B8A + median.B11)
            logging.info("SIWSI index computed")
            
            # Calculate NDMI for the median mosaic
            ndmi_median = (median.B08 - median.B11)/(median.B08 + median.B11)
            logging.info("NDMI index computed")
            
            # Calculate BNDVI for the median mosaic
            bndvi_median = (median.B08 - median.B02)/(median.B08 + median.B02)
            logging.info("BNDVI index computed")
            
            # Calculate NBR for the median mosaic
            nbr_median = (median.B08 - median.B12)/(median.B08 + median.B12)
            logging.info("NBR index computed")
            
            # Calculate PNDVI for the median mosaic
            pndvi_median = (median.B08 - (median.B03 + median.B04 + median.B02))/(median.B08 + (median.B03 + median.B04 + median.B02))
            logging.info("PNDVI index computed")
            
            # Calculate SI for the median mosaic
            si_median = (2.0*median.B04 - median.B03 - median.B02)/(median.B03 - median.B02)
            # Replace infs and -infs by NaNs
            si_median = si_median.where(si_median.values != -np.inf)
            si_median = si_median.where(si_median.values != np.inf)
            # Replace NaNs by 0.0
            si_median.fillna(0.0)
            logging.info("SI index computed")
            
            # Calculate RBNDVI for the median mosaic
            rbndvi_median = (median.B08 - (median.B04 + median.B02))/(median.B08 + (median.B04 + median.B02))
            logging.info("RBNDVI index computed")
            
            # Calculate SRSWIRI/NIR for the median mosaic
            srswirnir_median = median.B11/median.B08
            logging.info("SRSWIRI/NIR index computed")
            
            # Calculate SBL for the median mosaic
            sbl_median = median.B8A - 2.4*median.B04
            logging.info("SBL index computed")
            
            # Calculate wetness for the median mosaic
            w_median = 0.1509*median.B02 + 0.1973*median.B03 + 0.3279*median.B04 + 0.3406*median.B08 - 0.7112*median.B11 - 0.4572*median.B12
            logging.info("wetness index computed")
            
            # Calculate SIPI1 for the median mosaic
            sipi1_median = (median.B08 - median.B01)/(median.B08 - median.B04)
            # Replace infs and -infs by NaNs
            sipi1_median = sipi1_median.where(sipi1_median.values != -np.inf)
            sipi1_median = sipi1_median.where(sipi1_median.values != np.inf)
            # Replace NaNs by 0.0
            sipi1_median.fillna(0.0)
            logging.info("SIPI1 index computed")
            
            # Calculate VARI for the median mosaic
            vari_median = (median.B03 - median.B04)/(median.B03 + median.B04 - median.B02)
            logging.info("VARI index computed")
            
            # Calculate TDVI for the median mosaic
            tdvi_median = 1.5*(median.B08 - median.B04)/np.sqrt(median.B08*median.B08 + median.B04 + 0.5)
            logging.info("TDVI index computed")
            
            # Calculate EVI for the median mosaic
            evi_median = 2.5*(median.B08 - median.B04)/(median.B08 + 6.0*median.B04 -7.5*median.B02 + 1.0)
            # Replace infs and -infs by NaNs
            evi_median = evi_median.where(evi_median.values != -np.inf)
            evi_median = evi_median.where(evi_median.values != np.inf)
            # Replace NaNs by 0.0
            evi_median.fillna(0.0)
            logging.info("EVI index computed")
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
            fs_median.rio.write_crs("epsg:4326", inplace = True)
            msr_median.rio.write_crs("epsg:4326", inplace = True)
            msavi_median.rio.write_crs("epsg:4326", inplace = True)
            pvr_median.rio.write_crs("epsg:4326", inplace = True)
            psndc2_median.rio.write_crs("epsg:4326", inplace = True)
            siwsi_median.rio.write_crs("epsg:4326", inplace = True)
            ndmi_median.rio.write_crs("epsg:4326", inplace = True)
            bndvi_median.rio.write_crs("epsg:4326", inplace = True)
            nbr_median.rio.write_crs("epsg:4326", inplace = True)
            pndvi_median.rio.write_crs("epsg:4326", inplace = True)
            si_median.rio.write_crs("epsg:4326", inplace = True)
            rbndvi_median.rio.write_crs("epsg:4326", inplace = True)
            srswirnir_median.rio.write_crs("epsg:4326", inplace = True)
            sbl_median.rio.write_crs("epsg:4326", inplace = True)
            w_median.rio.write_crs("epsg:4326", inplace = True)
            sipi1_median.rio.write_crs("epsg:4326", inplace = True)
            vari_median.rio.write_crs("epsg:4326", inplace = True)
            tdvi_median.rio.write_crs("epsg:4326", inplace = True)
            evi_median.rio.write_crs("epsg:4326", inplace = True)
            
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
            fs_median.rio.write_transform(transform = gt, inplace = True)
            msr_median.rio.write_transform(transform = gt, inplace = True)
            msavi_median.rio.write_transform(transform = gt, inplace = True)
            pvr_median.rio.write_transform(transform = gt, inplace = True)
            psndc2_median.rio.write_transform(transform = gt, inplace = True)
            siwsi_median.rio.write_transform(transform = gt, inplace = True)
            ndmi_median.rio.write_transform(transform = gt, inplace = True)
            bndvi_median.rio.write_transform(transform = gt, inplace = True)
            nbr_median.rio.write_transform(transform = gt, inplace = True)
            pndvi_median.rio.write_transform(transform = gt, inplace = True)
            si_median.rio.write_transform(transform = gt, inplace = True)
            rbndvi_median.rio.write_transform(transform = gt, inplace = True)
            srswirnir_median.rio.write_transform(transform = gt, inplace = True)
            sbl_median.rio.write_transform(transform = gt, inplace = True)
            w_median.rio.write_transform(transform = gt, inplace = True)
            sipi1_median.rio.write_transform(transform = gt, inplace = True)
            vari_median.rio.write_transform(transform = gt, inplace = True)
            tdvi_median.rio.write_transform(transform = gt, inplace = True)
            evi_median.rio.write_transform(transform = gt, inplace = True)
            
            logging.info("Transformation to the EPSG:4326 CRS finished")
            
            # Create the GeoTIFF output file using the defined parameters 
            with rasterio.open(self.ingestion_config.raw_data_path, 'w', driver = 'GTiff', width = width, 
                               height = height, crs = 'epsg:4326', transform = gt, count = 29, 
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
                dst.write(fs_median, 11)
                dst.write(msr_median, 12)
                dst.write(msavi_median, 13)
                dst.write(pvr_median, 14)
                dst.write(psndc2_median, 15)
                dst.write(siwsi_median, 16)
                dst.write(ndmi_median, 17)
                dst.write(bndvi_median, 18)
                dst.write(nbr_median, 19)
                dst.write(pndvi_median, 20)
                dst.write(si_median, 21)
                dst.write(rbndvi_median, 22)
                dst.write(srswirnir_median, 23)
                dst.write(sbl_median, 24)
                dst.write(w_median, 25)
                dst.write(sipi1_median, 26)
                dst.write(vari_median, 27)
                dst.write(tdvi_median, 28)
                dst.write(evi_median, 29)
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
    clouds = 30.0                                                                      # Percentage
    res = 10.0                                                                   # Meters per pixel
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    
    obj = DataIngestion(l_l, u_r, t_w, coll, clouds, res, bands)
    raw_data = obj.initiate_data_ingestion()
    
    #data_transformation = DataTransformation()
    #_ = data_transformation.initiate_data_transformation(raw_data)