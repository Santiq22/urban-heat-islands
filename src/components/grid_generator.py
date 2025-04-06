# ========================================= Packages ============================================ #
from numpy import linspace, meshgrid, vstack
from pandas import DataFrame
# =============================================================================================== #

# ======================================= Main program ========================================== #
# Create the grid of points
latitudes = linspace(40.75, 40.88, 290)
longitudes = linspace(-74.01, -73.86, 334)

# Create a meshgrid from latitudes and longitudes
Lon, Lat = meshgrid(longitudes, latitudes)

# Flatten the grid to create a list of points
points = vstack([Lon.flatten(), Lat.flatten()]).T

df_grid = DataFrame()
df_grid['Longitude'] = points[:,0]
df_grid['Latitude'] = points[:,1]
df_grid.to_csv('../../data/initial_datasets/longitude_latitude_grid_data.csv', index = False)
# =============================================================================================== #