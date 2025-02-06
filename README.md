# urban-heat-islands
Repository containing the material needed in the 2025 EY Open Science AI and Data Challenge: Cooling Urban Heat Islands

### Repo structure and files
- `.gitignore`: Ignores of the repo.
- `env-ey25.yml`: File to set up the conda environment needed to the challenge.
- `notes.txt`: Text file to write some stuff and important things.
- bibliography: All the bibliography and references used in the competition
- data: Easy and light data. The main/raw data has to be saved locally or downloaded at running time

    - `Training_data_uhi_index_UHI2025-v2.csv`: These UHI Index values are the target parameters for the model.
    - `Building_Footprint.kml`: Building footprints of the Bronx and Manhattan regions.
    - `building_footprint_data.csv`: Area and perimeter of buildings of the Bronx and Manhattan regions.
    - `NY_Mesonet_Weather.csv`: Detailed local weather dataset of the Bronx and Manhattan regions on 24 July 2021 taken from NYS Mesonet.
    - `bronx_mesonet_weather_data.xlsx`: Detailed local weather dataset of the Bronx region on 24 July 2021 taken from NYS Mesonet.
    - `manhattan_mesonet_weather_data.xlsx`: Detailed local weather dataset of the Manhattan region on 24 July 2021 taken from NYS Mesonet.
    - `weather_data.csv`: Detailed local weather dataset of the Bronx and Manhattan regions on 24 July 2021 taken from NYS Mesonet including the azimuth and altitude of the Sun for a mean position and during the same period of time.
    - `Submission_template_UHI2025-v2.csv`: Validation dataset to predict the UHI index values on the identified locations.

- notebooks: All the notebooks we need for the competition

    - `Sentinel2_GeoTIFF.ipynb`: Sample notebook to download a GeoTIFF image from the Sentinel-2 satellite dataset.
    - `Landsat_LST.ipynb`: Sample notebook to download a GeoTIFF image from the Landsat satellite dataset.
    - `UHI Experiment Sample Benchmark Notebook V5.ipynb`: Jupyter notebook where a sample model has been built by using challenge training data.
    - `eda_surface_weather_data.ipynb`: Notebook to perform EDA over weather data and test the transformation of the Sun to the horizontal coordinate system.

- src:
    - `exception.py`: Script defining the CustomException class.
    - `logger.py`: Script setting up logger's format.
    - components:
        - `data_ingestion`: notebooks or scripts needed to perform data ingestion.
        - `data_transformation`: notebooks or scripts needed to transform data and to do EDA.
        
- logs: Just in case we need to log some runnings