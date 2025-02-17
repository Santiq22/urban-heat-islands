# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from pandas import read_csv

# Import Planetary Computer tools
import pystac_client
import planetary_computer 
from odc.stac import stac_load

# Make plots
import matplotlib.pyplot as plt
# =============================================================================================== #

try:
    # ====================================== Load the data ========================================== #
    # Define the bounding box for the entire data region using (Latitude, Longitude)
    # This is the region of New York City that contains our temperature dataset
    lower_left = (40.75, -74.01)
    upper_right = (40.88, -73.86)

    # Calculate the bounds for doing an archive data search
    # bounds = (min_lon, min_lat, max_lon, max_lat)
    bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])

    # Define the time window
    time_window = "2021-06-01/2021-09-01"

    # Open the client
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # Search based on the selected criterion 
    search = stac.search(
        bbox = bounds, 
        datetime = time_window,
        collections = ["sentinel-2-l2a"],
        query = {"eo:cloud_cover": {"lt": 30}})

    # Get items and store them into a list
    items = list(search.get_items())

    # Sign items
    signed_items = [planetary_computer.sign(item).to_dict() for item in items]

    # Define the pixel resolution for the final product
    # Define the scale according to our selected crs, so we will use degrees
    resolution = 10.0                                                                # meters per pixel 
    scale = resolution / 111320.0                                      # degrees per pixel for crs=4326

    # Convert loaded data into a xarray.Dataset object
    data = stac_load(
        items,
        bands = ["B02", "B03", "B04"],
        crs = "EPSG:4326",                                                         # Latitude-Longitude
        resolution = scale,                                                                   # Degrees
        chunks = {"x": 2048, "y": 2048},
        dtype = "uint16",
        patch_url = planetary_computer.sign,
        bbox = bounds)

    # Make a median composition over time for the different bands
    median = data.median(dim = "time").compute()

    # Load training data
    training_df = read_csv('../../data/Training_data_uhi_index_UHI2025-v2.csv')

    # Load test data
    test_df = read_csv('../../data/Test_data_uhi_index_UHI2025-v2.csv')
    # =============================================================================================== #

    # ====================================== Plot the data ========================================== #
    # Plot an RGB image for the median composite or mosaic
    fig, ax = plt.subplots(figsize = (9, 9), dpi = 400)
    median[["B04", "B03", "B02"]].to_array().plot.imshow(robust = True, ax = ax, vmin = 0, vmax = 2500)
    ax.scatter(training_df['Longitude'].values, training_df['Latitude'].values,
                s = 0.5, color = 'dodgerblue')
    ax.scatter(test_df['Longitude'].values, test_df['Latitude'].values, s = 0.5, color = 'red')
    ax.set_title("RGB Median Composite")
    ax.axis('off')
    fig.savefig('../../plots/longitudes_and_latitudes.png', bbox_inches = 'tight')
    # =============================================================================================== #
except Exception as e:
    raise CustomException(e, sys)