# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.coordinates import get_sun, AltAz, EarthLocation
import astropy.units as u
#from datetime import datetime
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataIngestionConfig:
    # Path to output datasets
    data_path: str = os.path.join('../../data', 'weather_data.csv')
    
class DataIngestion:
    def __init__(self, utcoffset, mean_latitude, mean_longitude, mean_altitude):
        # This variable will consist in the input I need to initialize
        self.ingestion_config = DataIngestionConfig()
        
        # UTC difference
        self.utcoffset = utcoffset
        
        # Mean latitude of the location where it is the observer
        self.mean_latitude = mean_latitude
        
        # Mean longitude of the location where it is the observer
        self.mean_longitude = mean_longitude

        # Mean altitude of the location where it is the observer
        self.mean_altitude = mean_altitude
        
    def initiate_data_ingestion(self):
        # Here needs to be the code to read the data (from local, database, etc)
        logging.info("Entered the data ingestion method or component from weather data")
        try:
        
            # Load Bronx dataset
            df_b = pd.read_excel('../../data/bronx_mesonet_weather_data.xlsx')
            
            logging.info("Bronx data correctly loaded")
            
            # Load Manhattan dataset
            df_m = pd.read_excel('../../data/manhattan_mesonet_weather_data.xlsx')
            
            logging.info("Manhattan data correctly loaded")
            
            # Convert Date and Time columns in datetime objects
            df_b[df_b.columns[0]] = pd.to_datetime(df_b[df_b.columns[0]])
            df_m[df_m.columns[0]] = pd.to_datetime(df_m[df_m.columns[0]])
            
            logging.info("Data correctly transformed to datetime objects")

            # Set time
            times = Time(df_m[df_m.columns[0]]) - self.utcoffset
            
            logging.info("Array of astropy Time objects defined")
            
            # Get the Sun in the GCRS system for the given times
            sun = get_sun(times)

            # Set the location
            new_york = EarthLocation(lat = self.mean_latitude*u.deg, 
                                     lon = self.mean_longitude*u.deg, 
                                     height = self.mean_altitude*u.m)
            
            logging.info("Location correcly set")
            
            # Transform the Sun to the horizontal coordinate system
            local_sun = sun.transform_to(AltAz(obstime = times, location = new_york))
            
            logging.info("Sun transformed to the horizontal coordinate system of New York")
            
            # Define azimuth and altitude arrays
            local_sun_azimuth = np.array(local_sun.az)
            local_sun_altitude = np.array(local_sun.alt)
            
            # Let us aggregate the azimuth and altitude arrays to the dataframe objects
            df_m['sun_altitude [deg]'] = local_sun_altitude
            df_m['sun_azimuth [deg]'] = local_sun_azimuth
            
            logging.info("Altitude and azimuth added to the dataset")
            
            # Rename columns for clarity
            df_m.rename(columns = {df_m.columns[1] : "air_temperature_at_surface_manhattan [degC]",
                                  df_m.columns[2] : "relative_humidity_manhattan [percent]",
                                  df_m.columns[3] : "avg_wind_speed_manhattan [m/s]",
                                  df_m.columns[4] : "wind_direction_manhattan [degrees]",
                                  df_m.columns[5] : "solar_flux_manhattan [W/m^2]"}, inplace = True)
            
            df_b.rename(columns = {df_b.columns[1] : "air_temperature_at_surface_bronx [degC]",
                                  df_b.columns[2] : "relative_humidity_bronx [percent]",
                                  df_b.columns[3] : "avg_wind_speed_bronx [m/s]",
                                  df_b.columns[4] : "wind_direction_bronx [degrees]",
                                  df_b.columns[5] : "solar_flux_bronx [W/m^2]"}, inplace = True)
            
            # Concatenate both datasets so we have a unique set of data
            df_conc = pd.concat((df_m, df_b.drop(columns = [df_b.columns[0]])), axis = 1)
            
            # Save the final dataset
            df_conc.to_csv(self.ingestion_config.data_path, index=False)
            
            logging.info("Data saved as a .csv file")
            
            logging.info("Ingestion of the data completed")
            
            return self.ingestion_config.data_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    # UTC difference
    utcoffset = -4.0*u.hour                                 # EDT (Eastern Daylight Time)
    
    # Mean latitude
    mean_lat = (40.76754 + 40.87248)/2.0                    # degrees

    # Mean longitude
    mean_lon = (-73.96449 - 73.89352)/2.0                   # degrees

    # Mean altitude
    mean_alt = (94.8 + 57.5)/2.0                            # meters
    
    # Instantiate DataIngestion object
    obj = DataIngestion(utcoffset, mean_lat, mean_lon, mean_alt)
    
    data = obj.initiate_data_ingestion()
    
    #data_transformation = DataTransformation()
    #_ = data_transformation.initiate_data_transformation(raw_data)