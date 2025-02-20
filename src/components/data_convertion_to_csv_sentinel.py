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
        self.csv_data_path: str = os.path.join('../../data/initial_datasets/', dataset+'_sentinel_data.csv')
    
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
            ndvi_median = []
            gndvi_median = []
            ndbi_median = []
            ndwi_median = []
            bwdrvi_median = []
            ccci_median = []
            ctvi_median = []
            datt1_median = []
            fe2_median = []
            fo_median = []
            fs_median = []
            msr_median = []
            msavi_median = []
            pvr_median = []
            psndc2_median = []
            siwsi_median = []
            ndmi_median = []
            bndvi_median = []
            nbr_median = []
            pndvi_median = []
            si_median = []
            rbndvi_median = []
            srswirnir_median = []
            sbl_median = []
            w_median = []
            sipi1_median = []
            vari_median = []
            tdvi_median = []
            evi_median = []

            # Iterate over the latitudes and longitudes, and extract the corresponding indeces values
            for lat, lon in tqdm(zip(latitudes, longitudes), total = len(latitudes), desc = "Mapping values"):
            # Assuming the correct dimensions are 'y' and 'x' (replace these with actual names 
            # from data.coords)
            
                ndvi = data.sel(x = lon, y = lat, band = 1, method = "nearest").values
                ndvi_median.append(ndvi)
                
                gndvi = data.sel(x = lon, y = lat, band = 2, method = "nearest").values
                gndvi_median.append(gndvi)
                
                ndbi = data.sel(x = lon, y = lat, band = 3, method = "nearest").values
                ndbi_median.append(ndbi)
                
                ndwi = data.sel(x = lon, y = lat, band = 4, method = "nearest").values
                ndwi_median.append(ndwi)
                
                bwdrvi = data.sel(x = lon, y = lat, band = 5, method = "nearest").values
                bwdrvi_median.append(bwdrvi)
                
                ccci = data.sel(x = lon, y = lat, band = 6, method = "nearest").values
                ccci_median.append(ccci)
            
                ctvi = data.sel(x = lon, y = lat, band = 7, method = "nearest").values
                ctvi_median.append(ctvi)
                
                datt1 = data.sel(x = lon, y = lat, band = 8, method = "nearest").values
                datt1_median.append(datt1)
                
                fe2 = data.sel(x = lon, y = lat, band = 9, method = "nearest").values
                fe2_median.append(fe2)
                
                fo = data.sel(x = lon, y = lat, band = 10, method = "nearest").values
                fo_median.append(fo)
                
                fs = data.sel(x = lon, y = lat, band = 11, method = "nearest").values
                fs_median.append(fs)
                
                msr = data.sel(x = lon, y = lat, band = 12, method = "nearest").values
                msr_median.append(msr)
                
                msavi = data.sel(x = lon, y = lat, band = 13, method = "nearest").values
                msavi_median.append(msavi)
                
                pvr = data.sel(x = lon, y = lat, band = 14, method = "nearest").values
                pvr_median.append(pvr)
                
                psndc2 = data.sel(x = lon, y = lat, band = 15, method = "nearest").values
                psndc2_median.append(psndc2)
                
                siwsi = data.sel(x = lon, y = lat, band = 16, method = "nearest").values
                siwsi_median.append(siwsi)
                
                ndmi = data.sel(x = lon, y = lat, band = 17, method = "nearest").values
                ndmi_median.append(ndmi)
                
                bndvi = data.sel(x = lon, y = lat, band = 18, method = "nearest").values
                bndvi_median.append(bndvi)
                
                nbr = data.sel(x = lon, y = lat, band = 19, method = "nearest").values
                nbr_median.append(nbr)
                
                pndvi = data.sel(x = lon, y = lat, band = 20, method = "nearest").values
                pndvi_median.append(pndvi)
                
                si = data.sel(x = lon, y = lat, band = 21, method = "nearest").values
                si_median.append(si)
                
                rbndvi = data.sel(x = lon, y = lat, band = 22, method = "nearest").values
                rbndvi_median.append(rbndvi)
                
                srswirnir = data.sel(x = lon, y = lat, band = 23, method = "nearest").values
                srswirnir_median.append(srswirnir)
                
                sbl = data.sel(x = lon, y = lat, band = 24, method = "nearest").values
                sbl_median.append(sbl)
                
                w = data.sel(x = lon, y = lat, band = 25, method = "nearest").values
                w_median.append(w)
                
                sipi1 = data.sel(x = lon, y = lat, band = 26, method = "nearest").values
                sipi1_median.append(sipi1)
                
                vari = data.sel(x = lon, y = lat, band = 27, method = "nearest").values
                vari_median.append(vari)
                
                tdvi = data.sel(x = lon, y = lat, band = 28, method = "nearest").values
                tdvi_median.append(tdvi)
                
                evi = data.sel(x = lon, y = lat, band = 29, method = "nearest").values
                evi_median.append(evi)
                
            logging.info("Indeces correctly loaded")
            
            # Create a DataFrame to store the band values
            df_out = DataFrame()
            df_out['Latitude'] = latitudes
            df_out['Longitude'] = longitudes
            df_out['ndvi_median_res10'] = ndvi_median
            df_out['gndvi_median_res10'] = gndvi_median
            df_out['ndbi_median_res10'] = ndbi_median
            df_out['ndwi_median_res10'] = ndwi_median
            df_out['bwdrvi_median_res10'] = bwdrvi_median
            df_out['ccci_median_res10'] = ccci_median
            df_out['ctvi_median_res10'] = ctvi_median
            df_out['datt1_median_res10'] = datt1_median
            df_out['fe2_median_res10'] = fe2_median
            df_out['fo_median_res10'] = fo_median
            df_out['fs_median_res10'] = fs_median
            df_out['msr_median_res10'] = msr_median
            df_out['msavi_median_res10'] = msavi_median
            df_out['pvr_median_res10'] = pvr_median
            df_out['psndc2_median_res10'] = psndc2_median
            df_out['siwsi_median_res10'] = siwsi_median
            df_out['ndmi_median_res10'] = ndmi_median
            df_out['bndvi_median_res10'] = bndvi_median
            df_out['nbr_median_res10'] = nbr_median
            df_out['pndvi_median_res10'] = pndvi_median
            df_out['si_median_res10'] = si_median
            df_out['rbndvi_median_res10'] = rbndvi_median
            df_out['srswirnir_median_res10'] = srswirnir_median
            df_out['sbl_median_res10'] = sbl_median
            df_out['w_median_res10'] = w_median
            df_out['sipi1_median_res10'] = sipi1_median
            df_out['vari_median_res10'] = vari_median
            df_out['tdvi_median_res10'] = tdvi_median
            df_out['evi_median_res10'] = evi_median
            
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
    path_to_tiff = '../../data/initial_datasets/raw_sentinel_data.tiff'
    path_to_csv = '../../data/initial_datasets/Training_data_uhi_index_2025-02-18.csv'
    #path_to_csv = '../../data/Test_data_uhi_index_UHI2025-v2.csv'
    #dataset_type = 'test'
    dataset_type = 'training'
    
    obj = DataConvertion(path_to_tiff, path_to_csv, dataset_type)
    csv_data = obj.initiate_data_convertion()