# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from numpy import ones
from numpy.random import default_rng
from pandas import read_csv, DataFrame
from datetime import datetime
# =============================================================================================== #

# ======================================= Main program ========================================== #
# Read dataset
test_df = read_csv('../../data/Submission_template_UHI2025-v2.csv')

# Get longitudes and latitudes
longitudes = test_df["Longitude"]
latitudes = test_df["Latitude"]

# Get lenght of arrays
lenght = len(longitudes)

# Generate array of random minutes, corresponding to the minutes between 15 o'clock and 16 o'clock
rng = default_rng()
minutes = rng.integers(low = 0, high = 59, size = lenght, endpoint = True)

# Create an array of datetime objects
datetimes = []
for i in range(lenght):
    date = datetime(2021, 7, 24, hour = 15, minute = minutes[i])
    date = date.strftime('%d-%m-%Y %H:%M')
    datetimes.append(date)
    
# Create new test DataFrame object
new_test_df = DataFrame()
new_test_df["Longitude"] = longitudes
new_test_df["Latitude"] = latitudes
new_test_df["datetime"] = datetimes
new_test_df["UHI Index"] = ones(lenght)

# Save the dataset
new_test_df.to_csv('../../data/Test_data_uhi_index_UHI2025-v2.csv', index=False)
# =============================================================================================== #