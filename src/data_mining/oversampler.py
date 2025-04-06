# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from pandas import read_csv, DataFrame
from numpy.random import multivariate_normal
from numpy import array, concatenate

# Machine learning
from sklearn.neighbors import KNeighborsRegressor
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class OversamplerConfig:
    def __init__(self, file_name):
        # Path to the output oversampled dataset
        self.file_path = os.path.join('../../data/final_datasets/transformed_datasets/interactions_datasets/', file_name)
    
class Oversampler:
    # This class oversamples the dataset and saves it in OversamplerConfig.file_path
    def __init__(self, data_path, data_path_oversampled, model, parameters, n_points, sigma_factor, file_name):
        # Set the output path to save the oversampled dataset
        self.oversampler_config = OversamplerConfig(file_name)
        
        # Path to the input data
        self.data_path = data_path
        
        # Path to the input oversampled data
        self.data_path_oversampled = data_path_oversampled
        
        # Model to use for oversampling
        self.model = model
        
        # Dictionary of hyperparameter names and values to set the model to use for oversampling
        self.parameters = parameters
        
        # Number of datapoints to generate
        self.n_points = n_points
        
        # Factor defining the sigma region to exclude predictions
        self.sigma_factor = sigma_factor
        
    def initiate_oversampling(self):
        logging.info("Entered the process of oversampling the data")
        
        try:
            # Read the input dataset
            df = read_csv(self.data_path)
            
            logging.info("Dataset correctly read")
            
            # Split between predictors and responses
            X = df.drop(columns = ['UHI Index'])
            y = df['UHI Index']
            
            # Compute the covariance matrix between variables
            cov_matrix = X.cov().values
            
            # Compute the mean of each variable
            mean_matrix = X.values.mean(axis = 0)
            
            logging.info("Covariance matrix and mean computed")
            
            # Compute the standard deviation of response values and scale it
            sigma = self.sigma_factor*y.values.std()
            
            # Set the model to train
            trained_model = self.model.set_params(**self.parameters)
            
            # Train the model
            trained_model.fit(X, y)
            
            logging.info("Model trained an ready to predict")
            
            # List of new synthetic datapoints
            X_new = []
            y_new = []
            
            # Iterate until the number of oversampled points is equal to self.n_points
            while(True):
                # Sample from multivariate normal a synthetic predictor
                X_tmp = multivariate_normal(mean_matrix, cov_matrix).reshape(1, -1)
                
                # Predict the response value of the corresponding synthetic predictor
                y_tmp = trained_model.predict(X_tmp)
                
                # Check if the predicted response falls out the ±simga/2 region
                if (y_tmp <= (1.0 - sigma) or y_tmp >= (1.0 + sigma)):
                    # Save the synthetic predictor and predicted response
                    X_new.append(X_tmp.reshape(-1,).tolist())
                    y_new.append(y_tmp)
                    print("Step: {}".format(len(y_new)))        
                
                # Check if the number of new synthetic datapoints is equal to self.n_points
                if(len(y_new) < self.n_points):
                    continue
                else:
                    break
                
            logging.info("Oversampling finished")
            
            # Convert the lists to arrays
            X_new = array(X_new)
            y_new = array(y_new)
            
            # Load the oversampled dataset
            df_over = read_csv(self.data_path_oversampled)
            
            logging.info("Oversampled data correctly read")
            
            # Split the oversampled data between predictors and responses
            X_over = df_over.drop(columns = ['UHI Index'])
            y_over = df_over['UHI Index']
            
            # Concatenate the new datapoints with the ones from the oversampled dataset
            X_conc = concatenate((X_over.values, X_new), axis = 0)
            y_conc = concatenate((y_over.values.reshape(-1, 1), y_new), axis = 0)
            
            # Concatenate the new arrays
            output_dataset = concatenate((X_conc, y_conc), axis = 1)
            
            # Create a new DataFrame object to save the oversampled dataset
            output_dataset = DataFrame(output_dataset, columns = df.columns)
            
            logging.info("New dataset created")
            
            # Save the new dataset
            output_dataset.to_csv(self.oversampler_config.file_path, index = False)
            
            logging.info("New oversampled dataset saved")
            
            return self.oversampler_config.file_path
        except Exception as e:
            raise CustomException(e, sys)
            
# =============================================================================================== #

if __name__ == '__main__':
    data_path = '../../data/final_datasets/transformed_datasets/interactions_datasets/interactions_reduced_training_data.csv'
    data_path_oversampled = '../../data/final_datasets/transformed_datasets/interactions_datasets/interactions_reduced_training_data.csv'
    model = KNeighborsRegressor(n_jobs = -1)
    hyperparameters = {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'manhattan', 'metric_params': None,
                       'n_neighbors': 2, 'p': 2, 'weights': 'distance'}
    n = 2000
    sigma_factor = 0.5
    output_name = 'interactions_reduced_oversampled_training_data.csv'
    
    obj = Oversampler(data_path = data_path, data_path_oversampled = data_path_oversampled,
                      model = model, parameters = hyperparameters, n_points = n, 
                      sigma_factor = sigma_factor, file_name = output_name)
    path = obj.initiate_oversampling()