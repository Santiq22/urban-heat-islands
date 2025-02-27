# urban-heat-islands
Repository containing the material needed in the 2025 EY Open Science AI and Data Challenge: Cooling Urban Heat Islands

### Repo structure and files
- `.gitignore`: Ignores of the repo.
- `env-ey25.yml`: File to set up the conda environment needed to the challenge.
- `notes.txt`: Text file to write some stuff and important things.
- bibliography: All the bibliography and references used in the competition.
- data: Easy and light data. The main/raw data has to be saved locally or downloaded at running time.

    - initial_datasets:
        - `Training_data_uhi_index_UHI2025-v2.csv`: Training dataset containing longitudes, latitudes, datetime variables and the UHI indeces to predict.
        - `Test_data_uhi_index_UHI2025-v2.csv`: Test dataset containing longitudes, latitudes, and datetime variables to generate predictions to upload as submission.
        - `Building_Footprint.kml`: Building footprints of the Bronx and Manhattan regions.
        - `building_footprint_data.csv`: Area, perimeter, and density of buildings, and HAG for the training dataset and test dataset locations.
        - `NY_Mesonet_Weather.csv`: Detailed local weather dataset of the Bronx and Manhattan regions on 24 July 2021 taken from NYS Mesonet.
        - `bronx_mesonet_weather_data.xlsx`: Detailed local weather dataset of the Bronx region on 24 July 2021 taken from NYS Mesonet.
        - `manhattan_mesonet_weather_data.xlsx`: Detailed local weather dataset of the Manhattan region on 24 July 2021 taken from NYS Mesonet.
        - `weather_data.csv`: Detailed local weather dataset of the Bronx and Manhattan regions on 24 July 2021 taken from NYS Mesonet including the azimuth and altitude of the Sun for a mean position and during the same period of time.
        - `Submission_template_UHI2025-v2.csv`: Validation dataset to predict the UHI index values on the identified locations.
        - `landast_data.csv`: Landast bands or indeces corresponding to the locations of the training or test data.
        - `sentinel_data.csv`: Sentinel bands or indeces corresponding to the locations of the training or test data.
        - `hag_data.csv`: HAG index based on longitudes and latitudes.
        - `longitude_latitude_grid_data.csv`: Grid of points over the working region spaced by 50 meters.
    - final_dataset:
        - `raw_data.csv`: Joint datasets combining satellite, building footprint, and weather data without longitudes, latitudes, and datetime variables.
        - `transformed_data.csv`: Datasets result of the transformation of the raw data.
    - submissions: 
        - `submissions-v1.csv`: First submission made.

- notebooks: All the notebooks we need for the competition.

    - `Sentinel2_GeoTIFF.ipynb`: Sample notebook to download a GeoTIFF image from the Sentinel-2 satellite dataset.
    - `Landsat_LST.ipynb`: Sample notebook to download a GeoTIFF image from the Landsat satellite dataset.
    - `UHI Experiment Sample Benchmark Notebook V5.ipynb`: Jupyter notebook where a sample model has been built by using challenge training data.
    - `eda_surface_weather_data.ipynb`: Notebook to perform EDA over weather data and test the transformation of the Sun to the horizontal coordinate system.
    - `eda_final_dataset.ipynb`: Notebook to perform EDA over the final dataset with all the variables considered for training, before transforming the data.
    - `model_training.ipynb`: Notebook to test different ML models and make predictions to submit.

- src:
    - `exception.py`: Script defining the CustomException class.
    - `logger.py`: Script setting up logger's format.
    - components:
        - `data_ingestion`: Scripts needed to perform the ingestion of the satellite, building footprint, or weather data.
        - `data_transformation`: Script to transform the joint dataset into the one used in the data mining process.
        - `data_convertion_to_csv`: Script to convert the .tiff satellite data into a .csv dataset based on given longitudes and latitudes.
        - `final_dataset_generator.py`: Joins the Sentinel, Landsat, building footprint, and weather data stored as .csv files to form a joint dataset.
        - `random_datetime_variable_generator.py`: Generates randomly datetime variables betweem the period '24-7-2021 15:00' to '24-7-2021 16:00' to the set of longitudes and latitudes given in `Submission_template_UHI2025-v2.csv`. This way it returns a viable test set to make predictions.
    - plots_makers:
        - `latitudes_and_longitudes.py`: Makes a plot of an RGB image of the zone of Manhattan and Bronx superimposing the latitudes and longitudes of the training and test set.
        
- plots
- logs: Just in case we need to log some runnings.