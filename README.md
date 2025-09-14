# Urban Heat Islands – EY Open Science AI & Data Challenge 2025

This repository contains all materials developed for the **EY Open Science AI and Data Challenge 2025: Cooling Urban Heat Islands**.

The main aim of the project is to **predict the Urban Heat Island (UHI) index in New York City (Bronx and Manhattan) using satellite, weather, building, and demographic data**, and to design reproducible workflows for data integration, transformation, and modeling, contributing to data-driven strategies for urban cooling and climate resilience.

---

## 🔎 Project Overview

Urban Heat Islands are areas where urban structures (buildings, roads, infrastructure) cause higher temperatures compared to surrounding rural areas.  

This project focuses on:

- Collecting, cleaning, and integrating multi-source geospatial and meteorological datasets.  
- Engineering features related to land cover, building density, and population.  
- Testing machine learning models for predicting UHI index values.  
- Creating visualizations to better understand patterns and drivers of UHIs.  

---

## 📊 Data Sources

The project leverages multiple datasets from satellites, weather stations, and census information:

- **Challenge datasets**: Training and test sets with UHI index, lat/long, and datetime values.  
- **Satellite Data**:
  - *Sentinel-2* and *Landsat* indices extracted for target locations.
- **Weather data**:
  - High-resolution meteorological data (NY Mesonet) including solar position, temperature, humidity, and wind for Bronx and Manhattan.
- **Urban and demographic data**:
  - Building footprints (`.kml` + CSV).
  - PLUTO dataset (building features) with floors/units per building.
  - Population counts from US Census Blocks.  

These datasets are organized into **raw**, **transformed**, and **final** versions, with additional grids for spatial smoothing.

---

## ⚙️ Data Processing Workflow

The `src/components` folder contains scripts to automate ingestion, transformation, and feature engineering:

1. **Ingestion & conversion**  
   - Satellite bands extraction (Sentinel, Landsat) and convertion of GeoTIFFs to CSV.  
   - Merge satellite, building, demographic, and weather data.  

2. **Feature Engineering**  
   - Random datetime generator for test set alignment.  
   - Grid generator + moving means to smooth spatial variables.  
   - Polynomial features.  
   - PCA transformations.  
   - SMOTE-based oversampling for class balancing.  

3. **Dataset Preparation**  
   - Creation of training and submission-ready test datasets.  

---

## 🤖 Modeling Approach

Several models were benchmarked and trained, using the scripts in `src/data_mining` and notebooks:
  
- **Correlations analysis** for dimensionality reduction.  
- **K-Means clustering** on transformed data to identify UHI patterns.  
- **Model trainer** with hyperparameter tuning.  

---

## 📈 Visualizations

Plots and notebooks provide exploratory analysis and visual summaries:

- **EDA** of weather data and final datasets.  
- **Spatial visualizations**: RGB maps of Manhattan/Bronx with UHI indeces and building heights. Heatmap of Sentinel and Landsat indeces spatial distribution.
- **Model evaluation**: Learning curves and feature importance plots.  

All generated figures are collected in the `plots/` folder.

---

## 📚 References

Relevant bibliography and sources are in the `bibliography/` folder.

---

## 📂 Repo structure and files

- `.gitignore`: Ignores of the repo.
- `env-ey25.yml`: File to set up the conda environment needed to the challenge.
- bibliography: All the bibliography and references used in the competition.
- data: Easy and light data. The main/raw data has to be saved locally or downloaded at running time.

  - initial_datasets:
    - `Training_data_uhi_index_2025-02-18.csv`: Training dataset containing longitudes, latitudes, datetime variables and the UHI indeces to predict.
    - `Test_data_uhi_index_UHI2025-v2.csv`: Test dataset containing longitudes, latitudes, and datetime variables to generate predictions to upload as submission.
    - `Building_Footprint.kml`: Building footprints of the Bronx and Manhattan regions.
    - `building_footprint_data.csv`: Building footprint data corresponding to the training dataset and test dataset locations.
    - `NY_Mesonet_Weather.csv`: Detailed local weather dataset of the Bronx and Manhattan regions on 24 July 2021 taken from NYS Mesonet.
    - `bronx_mesonet_weather_data.xlsx`: Detailed local weather dataset of the Bronx region on 24 July 2021 taken from NYS Mesonet.
    - `manhattan_mesonet_weather_data.xlsx`: Detailed local weather dataset of the Manhattan region on 24 July 2021 taken from NYS Mesonet.
    - `weather_data.csv`: Detailed local weather dataset of the Bronx and Manhattan regions on 24 July 2021 taken from NYS Mesonet including the azimuth and altitude of the Sun for a mean position and during the same period of time.
    - `weather_stations.csv`: Data of weather stations and some metheorological variables for the region of interest.
    - `Submission_template_UHI2025-v2.csv`: Validation dataset to predict the UHI index values on the identified locations.
    - `landast_data.csv`: Landast bands or indeces corresponding to the locations of the training or test data.
    - `sentinel_data.csv`: Sentinel bands or indeces corresponding to the locations of the training or test data.
    - `longitude_latitude_grid_data.csv`: Grid of points over the working region spaced by 50 meters inteded to compute moving means over the different training and test locations.
    - `pluto_data.csv`: Data reduced from the Pluto dataset for the working region. It contains the location of a given point and the number of building floors and the number of units asociated to such a location.
    - `population_count_data.csv`: Population count for the working area. Taken from US Census Blocks: https://hub.arcgis.com/datasets/fedmaps::u-s-census-blocks-1/explore?location=40.793367%2C-73.967479%2C18.00
  - final_dataset:
    - raw_datasets:
      - `raw_data.csv`: Joint datasets combining satellite, building footprint, weather, and demographic data without longitudes, latitudes, and datetime variables.
    - transformed_datasets:
      - `transformed_data.csv`: Different datasets result of the transformation of the raw data.
      - interactions_datasets:
        - `interactions_data.csv`: Transformed data after adding new variables resulting from the interaction between older ones.
      - pca_datasets:
        - `pca_data.csv`: Transformed data after a PCA transformation
  - submissions:
    - `submissions.csv`: All the submissions made.

- notebooks: All the notebooks we need for the competition.

  - `Sentinel2_GeoTIFF.ipynb`: Sample notebook to download a GeoTIFF image from the Sentinel-2 satellite dataset.
  - `Landsat_LST.ipynb`: Sample notebook to download a GeoTIFF image from the Landsat satellite dataset.
  - `UHI Experiment Sample Benchmark Notebook.ipynb`: Jupyter notebook where a sample model has been built by using challenge training data.
  - `eda_surface_weather_data.ipynb`: Notebook to perform EDA over weather data and test the transformation of the Sun to the horizontal coordinate system.
  - `eda_final_dataset.ipynb`: Notebook to perform EDA over the final dataset with all the variables considered for training, before transforming the data.
  - `model_training.ipynb`: Notebook make predictions to submit.
  - `correlations_analysis.ipynb`: Dimensionality reduction analysis based on the correlations between different variables of the final transformed dataset.
  - `benchmark_model_training.ipynb`: Benchmark of different machine-learning models.
  - `k_means_analisys.ipynb`: K-Means analisys over the final transformed dataset.

- src:
  - `exception.py`: Script defining the CustomException class.
  - `logger.py`: Script setting up logger's format.
  - `utils.py`: Script defining a function to save objects and a function used to look for the best hyperparameters of a given model.
  - components:
    - `data_ingestion`: Scripts needed to perform the ingestion of the satellite, building footprint, demographic, or weather data.
    - `data_transformation.py`: Script to transform the joint dataset into the one used in the data mining process.
    - `data_convertion_to_csv`: Script to convert the .tiff satellite data into a .csv dataset based on given longitudes and latitudes.
    - `final_dataset_generator.py`: Joins the Sentinel, Landsat, building footprint, demographic, and weather data stored as .csv files to form a joint dataset.
    - `random_datetime_variable_generator.py`: Generates randomly datetime variables betweem the period '24-7-2021 15:00' to '24-7-2021 16:00' to the set of longitudes and latitudes given in `Submission_template_UHI2025-v2.csv`. This way it returns a viable test set to make predictions.
    - `grid_generator.py`: Generates a grid of points of the working region.
    - `moving_mean.py`: Computes the moving mean in points of the training and test data using the locations of `longitude_latitude_grid_data.csv`.
    - `pca_transformation.py`: Carries out a PCA transformation over the transformed datasets.
    - `smote_resampling.py`: Script to resample training data using the SMOTE technique. It discretizes the response variable to the largest number of possible classes.
  - data_mining:
    - trained_models: Contains .pkl files representing best fit machine learning models.
    - `model_trainer.py`: Looks for the best fit hyperparameters for a given machine learning model and dataset.
    - `oversampler.py`: Oversamples training data using synthetic data generated with a trained machine learning model based on randomly sampled predictors.
    - `polynomial_model.py`: This script engineers new variables through interactions between different old predictors.
  - plots_makers:
    - `latitudes_and_longitudes.py`: Makes a plot of an RGB image of the zone of Manhattan and Bronx superimposing the latitudes and longitudes of the training and test set.
    - `colormap_of_heights.py`: Plots an RGB imagen of the working region superimposing the number of floors of the buildings on it.
- plots: Different useful plots.
