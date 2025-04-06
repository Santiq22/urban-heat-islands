# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

# Data Science
from pandas import read_csv, concat, DataFrame
from numpy import array, arange, argmin, argwhere, append, hstack, pi, float64, int32

# Interpolation
from scipy.interpolate import InterpolatedUnivariateSpline

# Others
from datetime import datetime
from shapely import Point
from tqdm import tqdm
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataGeneratorConfig:
    def __init__(self, file_name):
        # Path to output dataset
        self.file_path: str = os.path.join('../../data/final_datasets/raw_datasets/', file_name)
    
class DataGenerator:
    def __init__(self, sentinel_data, landsat_data, building_data, weather_data, stations_data, 
                 base_data, output_name, type_of_dataset, drop_columns = []):
                
        # This variable will consist in the input I need to initialize
        self.generator_config = DataGeneratorConfig(output_name)
        
        # Path to sentinel .csv data
        self.sentinel_data = sentinel_data
        
        # Path to landsat .csv data
        self.landsat_data = landsat_data
        
        # Path to building footprint .csv data
        self.building_data = building_data
        
        # Path to weather .csv data
        self.weather_data = weather_data
        
        # Path to stations .csv data
        self.stations_data = stations_data
        
        # Path to training or test data
        self.base_data = base_data
        
        # Type of dataset to concatenate
        self.type_of_dataset = type_of_dataset
        
        # Columns to drop before saving the concatened dataset
        self.drop_columns = drop_columns
        
    def initiate_data_generation(self):
        # Load all the .csv datasets and concatenate them
        logging.info("Entered the data generation method or component")
        
        try:
            # Load Sentinel data
            df_sentinel = read_csv(self.sentinel_data)
            logging.info("Sentinel .csv data loaded")
            
            # Load Landsat data
            df_landsat = read_csv(self.landsat_data)
            logging.info("Landsat .csv data loaded")
            
            # Load building footprint data
            df_building = read_csv(self.building_data)
            logging.info("Building footprint .csv data loaded")
            
            # Load weather data
            df_weather = read_csv(self.weather_data)
            logging.info("Weather .csv data loaded")
            
            # Load stations data
            df_stations = read_csv(self.stations_data)
            logging.info("Stations .csv data loaded")
            
            # Load training or test data
            df_base = read_csv(self.base_data)
            logging.info("Training or test .csv data loaded")
            
            # Array of datetime objects from the weather_dataset
            times = array([datetime.strptime(time, '%Y-%m-%d %H:%M:%S').hour for time in df_weather[df_weather.columns[0]].values],
                          dtype = int32)
            
            # Indeces where the hour is equal to 15
            idx = argwhere(times == 15)
            
            # Append one index more to include 16 o'clock. It reshapes the array and returns a Numpy array of shape (n,)
            idx = append(idx, [[idx[-1, 0] + 1]])
            
            # Array of minutes corresponding from '2021-07-24 15:00:00' to '2021-07-24 16:00:00'
            times_weather = arange(0, 65, 5, dtype = float64)
            
            # Array of minutes from 'YYYY-MM-DD HH:MM:SS' objects from training or test dataset
            times_base = array([datetime.strptime(time, '%d-%m-%Y %H:%M').minute for time in df_base[df_base.columns[2]].values], 
                               dtype = float64)
            
            # Array of meteorological quantities to interpolate
            air_temp_manhattan = df_weather['air_temperature_at_surface_manhattan [degC]'].values[idx]
            humidity_manhattan = df_weather['relative_humidity_manhattan [percent]'].values[idx]
            wind_speed_manhattan = df_weather['avg_wind_speed_manhattan [m/s]'].values[idx]
            wind_direction_manhattan = df_weather['wind_direction_manhattan [degrees]'].values[idx]
            solar_flux_manhattan = df_weather['solar_flux_manhattan [W/m^2]'].values[idx]
            air_temp_bronx = df_weather['air_temperature_at_surface_bronx [degC]'].values[idx]
            humidity_bronx = df_weather['relative_humidity_bronx [percent]'].values[idx]
            wind_speed_bronx = df_weather['avg_wind_speed_bronx [m/s]'].values[idx]
            wind_direction_bronx = df_weather['wind_direction_bronx [degrees]'].values[idx]
            solar_flux_bronx = df_weather['solar_flux_bronx [W/m^2]'].values[idx]
            sun_altitude = df_weather['sun_altitude [deg]'].values[idx]
            sun_azimuth = df_weather['sun_azimuth [deg]'].values[idx]
            
            logging.info("Array of meteorological variables loaded")
            
            # Interpolate meteorological variables
            air_temp_manhattan = InterpolatedUnivariateSpline(times_weather, air_temp_manhattan, k = 3)
            humidity_manhattan = InterpolatedUnivariateSpline(times_weather, humidity_manhattan, k = 3)
            wind_speed_manhattan = InterpolatedUnivariateSpline(times_weather, wind_speed_manhattan, k = 3)
            wind_direction_manhattan = InterpolatedUnivariateSpline(times_weather, wind_direction_manhattan, k = 3)
            solar_flux_manhattan = InterpolatedUnivariateSpline(times_weather, solar_flux_manhattan, k = 3)
            air_temp_bronx = InterpolatedUnivariateSpline(times_weather, air_temp_bronx, k = 3)
            humidity_bronx = InterpolatedUnivariateSpline(times_weather, humidity_bronx, k = 3)
            wind_speed_bronx = InterpolatedUnivariateSpline(times_weather, wind_speed_bronx, k = 3)
            wind_direction_bronx = InterpolatedUnivariateSpline(times_weather, wind_direction_bronx, k = 3)
            solar_flux_bronx = InterpolatedUnivariateSpline(times_weather, solar_flux_bronx, k = 3)
            sun_altitude = InterpolatedUnivariateSpline(times_weather, sun_altitude, k = 3)
            sun_azimuth = InterpolatedUnivariateSpline(times_weather, sun_azimuth, k = 3)
            
            logging.info("InterpolatedUnivariateSpline objects created on meteorological variables")
            
            # Arrays of meteorological variables corresponding to each time of the training or test set
            air_temp_manhattan = air_temp_manhattan(times_base)
            humidity_manhattan = humidity_manhattan(times_base)
            wind_speed_manhattan = wind_speed_manhattan(times_base)
            wind_direction_manhattan = wind_direction_manhattan(times_base)
            solar_flux_manhattan = solar_flux_manhattan(times_base)
            air_temp_bronx = air_temp_bronx(times_base)
            humidity_bronx = humidity_bronx(times_base)
            wind_speed_bronx = wind_speed_bronx(times_base)
            wind_direction_bronx = wind_direction_bronx(times_base)
            solar_flux_bronx = solar_flux_bronx(times_base)
            sun_altitude = sun_altitude(times_base)
            sun_azimuth = sun_azimuth(times_base)
            
            logging.info("Meteorological variables evaluated in the times of the training or test set")
            
            # Get training or test latitudes
            latitudes = df_base['Latitude'].values
            longitudes = df_base['Longitude'].values
            
            # Mean latitude
            mean_lat = (40.76754 + 40.87248)/2.0                    # degrees
            
            # Set empty list to store values
            air_temp = []
            humidity = []
            wind_speed = []
            wind_direction = []
            solar_flux = []
            
            # Check latitudes to assign Manhattan or Bronx to the meteorological variables
            for j, lat in enumerate(latitudes):
                if lat <= mean_lat:
                    air_temp.append(air_temp_manhattan[j])
                    humidity.append(humidity_manhattan[j])
                    wind_speed.append(wind_speed_manhattan[j])
                    wind_direction.append(wind_direction_manhattan[j])
                    solar_flux.append(solar_flux_manhattan[j])
                else:
                    air_temp.append(air_temp_bronx[j])
                    humidity.append(humidity_bronx[j])
                    wind_speed.append(wind_speed_bronx[j])
                    wind_direction.append(wind_direction_bronx[j])
                    solar_flux.append(solar_flux_bronx[j])
                    
            logging.info("Meteorological variables assigned based on latitudes")
                    
            # Create weather data DataFrame
            df_weather_new = DataFrame()
            df_weather_new['air_temperature_at_surface [degC]'] = air_temp
            df_weather_new['relative_humidity [percent]'] = humidity
            df_weather_new['avg_wind_speed [m/s]'] = wind_speed
            df_weather_new['wind_direction [degrees]'] = wind_direction
            df_weather_new['solar_flux [W/m^2]'] = solar_flux
            df_weather_new['sun_altitude [deg]'] = sun_altitude
            df_weather_new['sun_azimuth [deg]'] = sun_azimuth
            
            logging.info("Meteorological variables saved on DataFrame object")
            
            # Define shapely Points objects on the training or test data
            points_base = [Point(coord) for coord in hstack((longitudes.reshape(-1, 1), 
                                                             latitudes.reshape(-1, 1)))*pi/180.0]
            
            # Define shapely Points objects on the stations data
            points_stations = [Point(coord) for coord in df_stations[['Longitude', 'Latitude']].values*pi/180.0]
            
            # New DataFrame object to store the results in each step
            df_stations_new = DataFrame(columns = ['temperature', 'pressure_3hr_change', 'temperature_3h_moving_avg', 
                                                   'temperature_6h_moving_avg', 'temperature_12h_moving_avg'])
            
            # Map each point in the training or test data with each station
            for point in tqdm(points_base, total = len(points_base), desc = "Mapping values"):
                # Compute the distance between the point and the weather stations
                d = array([p.centroid.distance(point) for p in points_stations])
                
                # Check which distance is the smallest
                idx_min = argmin(d)
                
                # Concatenate with the new DataFrame
                df_stations_new = concat((df_stations_new, df_stations[['temperature', 'pressure_3hr_change', 
                                                                        'temperature_3h_moving_avg', 
                                                                        'temperature_6h_moving_avg', 
                                                                        'temperature_12h_moving_avg']].iloc[[idx_min]]), axis = 0)
            
            # Check if the base dataset is the training or test one
            if self.type_of_dataset == 'training':
                # Concatenate all the datasets and targets in the case of the training dataset
                df_conc = concat((df_sentinel.drop(columns = ['Longitude', 'Latitude']),
                                df_landsat.drop(columns = ['Longitude', 'Latitude']),
                                df_building.drop(columns = ['Longitude', 'Latitude']),
                                df_weather_new,
                                df_stations_new.reset_index(drop = True),
                                df_base[df_base.columns[-1]]), axis = 1)
                
                # Check if there are some columns to drop
                if not (len(self.drop_columns) == 0):
                    df_conc.drop(columns = self.drop_columns, inplace = True)
                
            else:
                # Concatenate all the datasets without the targets in the case of the test dataset
                df_conc = concat((df_sentinel.drop(columns = ['Longitude', 'Latitude']),
                                df_landsat.drop(columns = ['Longitude', 'Latitude']),
                                df_building.drop(columns = ['Longitude', 'Latitude']),
                                df_weather_new,
                                df_stations_new.reset_index(drop = True)), axis = 1)
                
                # Check if there are some columns to drop
                if not (len(self.drop_columns) == 0):
                    df_conc.drop(columns = self.drop_columns, inplace = True)
            
            # Save the DataFrame object as csv
            df_conc.to_csv(self.generator_config.file_path, index=False)
            
            logging.info("Final dataset correctly saved")
            
            logging.info("Data generation process finished")
            
            return self.generator_config.file_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    dataset_type = 'training'
    #dataset_type = 'test'
    sentinel = '../../data/initial_datasets/' + dataset_type + '_sentinel_data.csv'
    landsat = '../../data/initial_datasets/' + dataset_type + '_landsat_data.csv'
    building = '../../data/initial_datasets/' + dataset_type + '_building_footprint_data.csv'
    weather = '../../data/initial_datasets/weather_data.csv'
    station = '../../data/initial_datasets/weather_stations.csv'
    base = '../../data/initial_datasets/Training_data_uhi_index_2025-02-18.csv'
    #base = '../../data/initial_datasets/Test_data_uhi_index_UHI2025-v2_30min.csv'
    output = 'raw_' + dataset_type + '_data.csv'
    columns_to_drop = ['ndvi_median_res10',
                       'bwdrvi_median_res10',
                       'ctvi_median_res10',
                       'ccci_median_res10',
                       'fe2_median_res10',
                       'fo_median_res10',
                       'msr_median_res10',
                       'msavi_median_res10',
                       'psndc2_median_res10',
                       'bndvi_median_res10',
                       'nbr_median_res10',
                       'pndvi_median_res10',
                       'pvr_median_res10',
                       'rbndvi_median_res10',
                       'srswirnir_median_res10',
                       'sbl_median_res10',
                       'sipi1_median_res10',
                       'vari_median_res10',
                       'tdvi_median_res10',
                       'ndbi_median_res10',
                       'ndwi_median_res10',
                       'B6_median_res10',
                       'datt1_median_res100',
                       'B7_median_res100',
                       'B8_median_res100',
                       'B8A_median_res100',
                       'datt1_median_res250',
                       'datt1_median_res500',
                       'si_median_res500',
                       'polygon_perimeter',
                       'polygon_area',
                       'relative_humidity [percent]',
                       'avg_wind_speed [m/s]',
                       'wind_direction [degrees]',
                       'solar_flux [W/m^2]',
                       'sun_altitude [deg]',
                       'sun_azimuth [deg]']
    #columns_to_drop = []
    
    obj = DataGenerator(sentinel, landsat, building, weather, station, base, output, dataset_type, drop_columns = columns_to_drop)
    data = obj.initiate_data_generation()