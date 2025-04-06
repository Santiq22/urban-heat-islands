# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from numpy import argwhere, array, pi
from pandas import DataFrame, read_csv, concat
from shapely.geometry import Point
from tqdm import tqdm
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataMovingConfig:
    def __init__(self, input_dataset):
        # Path to output datasets
        #self.data_path: str = os.path.join('../../data/initial_datasets/', dataset+'_landsat_moving_data.csv')
        self.data_path = input_dataset
        
class DataMoving:
    def __init__(self, input_dataset, grid_path, indeces, radius):
        # Set the output path to save the new dataset
        self.moving_data_config = DataMovingConfig(input_dataset)
        
        # Path to the input dataset
        self.input_dataset = input_dataset
        
        # Path to the grid data
        self.grid_path = grid_path
        
        # List of indeces to compute the moving mean
        self.indeces = indeces
        
        # Radius of the circle where to compute the mean
        self.r = radius
        
    def initiate_moving_mean_computation(self):
        logging.info("Entered the moving mean computation")
        
        # Earth radius
        earth_radius = 6378000.0                        # Meters
            
        try:
            # Load the input dataset
            df = read_csv(self.input_dataset)
            
            logging.info("Input dataset loaded")
            
            # Separate longitudes and latitudes and convert them to radians
            lon_lat = df[['Longitude', 'Latitude']].to_numpy()*pi/180.0
            
            # Create Shapely Point objects
            points = [Point(coord) for coord in lon_lat]
            
            logging.info("Point objects created")
            
            # Load the grid
            df_grid = read_csv(self.grid_path)
            logging.info("Grid dataset loaded")
            
            # Separate longitudes and latitudes of the grid and convert them to radians
            lon_lat_grid = df_grid[['Longitude', 'Latitude']].to_numpy()*pi/180.0
            
            # Create Shapely Point objects of points in the grid
            points_grid = [Point(coord) for coord in lon_lat_grid]
            
            logging.info("Point grid objects created")
            
            # Set the list to store the moving means
            moving_means = []
            
            # Compute the mean over the different indeces
            for point in tqdm(points, total = len(points), desc = "Mapping values"):
                # Compute the distance in physical units between the point and the points on the grid
                d = earth_radius*array([p.distance(point) for p in points_grid])
                
                # Get the indeces where d <= r
                idx = argwhere(d <= self.r)
                
                # Set the list to store the moving means temporarily
                moving_means_tmp = []
                
                # Compute the moving mean for each index
                for index in self.indeces:
                    # Compute the mean
                    mov_mean = df_grid[index].values[idx].sum()/len(idx)
                    
                    # Append it to the list
                    moving_means_tmp.append(mov_mean)
                
                # Append the list of moving means
                moving_means.append(moving_means_tmp)
                
            # Convert the list in an array
            moving_means = array(moving_means)
            
            logging.info("Moving means computed over each band")
                
            # Define a list with new names
            new_names = [name[:-6]+'_res'+'{}'.format(self.r) for name in self.indeces]
                
            # Create a DataFrame object containing the new moving mean values for each index
            df_new = DataFrame()
            for i, new_variable in enumerate(new_names):
                # Define a new column
                df_new[new_variable] = moving_means[:,i]
            
            logging.info("New dataset computed")
                
            # Concatenate the datasets
            df_concat = concat((df, df_new), axis = 1)
            
            # Save the new dataset
            df_concat.to_csv(self.moving_data_config.data_path, index = False)
            
            logging.info("New dataset saved")
            
            return self.moving_data_config.data_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    dataset = 'training'
    #dataset = 'test'
    input = '../../data/initial_datasets/'+dataset+'_landsat_data.csv'
    grid = '../../data/initial_datasets/longitude_latitude_grid_data.csv'
    """indeces = ["gndvi_median_res10",
               "datt1_median_res10",
               "fs_median_res10",
               "siwsi_median_res10",
               "ndmi_median_res10",
               "si_median_res10",
               "w_median_res10",
               "evi_median_res10",
               "B2_median_res10",
               "B3_median_res10",
               "B4_median_res10",
               "B5_median_res10",
               "B6_median_res10",
               "B7_median_res10",
               "B8_median_res10",
               "B8A_median_res10",
               "B11_median_res10",
               "B12_median_res10"]"""
    indeces = ["lst_median_res10"]
    radius = 500                          # Meters
    
    obj = DataMoving(input, grid, indeces, radius)
    path = obj.initiate_moving_mean_computation()