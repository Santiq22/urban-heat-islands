# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:45:02 2024

@author: Santiago Collazo
@original_author: Krish Naik
"""
# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object
from dataclasses import dataclass
from pandas import read_csv
from scipy.stats import loguniform

# Machine learning
#from catboost import CatBoostRegressor
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
#from xgboost import XGBRegressor
#from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class ModelTrainerConfig:
    def __init__(self, model_name):
        # Path to the output trained model saved as .pkl files
        self.trained_model_file_path = os.path.join('trained_models/', model_name)
    
class ModelTrainer:
    # This class trains the model and saves it in ModelTrainerConfig.trained_model_file_path
    def __init__(self, model_name, training_data_path, test_size, random_state, response_name,
                 model, parameters, n_iterations = None):
        # Set the output path to save the trained model
        self.model_trainer_config = ModelTrainerConfig(model_name)
        
        # Name to save the trained model as a .pkl file
        self.model_name = model_name
        
        # Path to the training data
        self.training_data_path = training_data_path
        
        # Proportion of training data used as test data
        self.test_size = test_size
        
        # Random state to set the train_test_split
        self.random_state = random_state
        
        # Name of response variable
        self.response_name = response_name
        
        # Object defining the model to train
        self.model = model
        
        # Dictionary of hyperparameter names and values to optimize the model using grid search CV
        self.parameters = parameters
        
        # Number of iterations the RandomizedSearchCV has to perform
        self.n_iterations = n_iterations
        
    def initiate_model_trainer(self):
        try:
            # Load dataset to train the model
            df = read_csv(self.training_data_path)
            
            logging.info("Dataset loaded")
            
            # Separate in predictors (X) and response (y)
            X = df.drop(columns = [self.response_name]).values
            y = df[self.response_name].values
            
            # Generate training and test predictors and responses
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size,
                                                                random_state = self.random_state)
            
            logging.info("Splitted training and test input data")
            
            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train, 
                                                X_test = X_test, y_test = y_test, model = self.model,
                                                parameters = self.parameters, n_iterations = self.n_iterations)
            
            logging.info("Grid search CV optimization and model trained finished")
            
            # Get the trained best model from dict
            best_model = model_report['trained_model']
            
            # Get best model score from dict
            best_training_model_score = model_report['training_score']
            best_test_model_score = model_report['test_score']
            
            # Get the best fit hyperparameters from dict
            best_fit_hyperparameters = model_report['best_fit_hyperparameters']

            if best_training_model_score < 0.6 or best_test_model_score < 0.6:
                print("Best model training R^2:", best_training_model_score)
                print("Best model test R^2:", best_test_model_score)
                print("Best fit hyperparameters:", best_fit_hyperparameters)
                raise CustomException("No best model found", sys)
            
            logging.info("Best model found on both training and test dataset")

            # Save the model using the utils function save_object
            save_object(file_path = self.model_trainer_config.trained_model_file_path, obj = best_model)
            
            logging.info("Best model trained object saved")
            
            print("Best model training R^2:", best_training_model_score)
            print("Best model test R^2:", best_test_model_score)
            print("Best fit hyperparameters:", best_fit_hyperparameters)
            
            return self.model_trainer_config.trained_model_file_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    model_name = 'best_fit_LGBMR_smote.pkl'
    training_data = '../../data/final_datasets/transformed_datasets/transformed_reduced_smote_training_data.csv'
    test_size = 0.2
    random_state = 10
    target_name = 'UHI Index binned'
    model = LGBMRegressor(random_state = 10)
    hyperparameters = dict(boosting_type = ['gbdt', 'dart', 'rf'],
                           num_leaves = [i for i in range(1, 2001, 1)],
                           max_depth = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                           learning_rate = loguniform(a = 1.0e-6, b = 10.0),
                           n_estimators = [i for i in range(1, 301, 1)],
                           colsample_bytree = loguniform(a = 1.0/27000, b = 1.0),
                           reg_alpha = loguniform(a = 1.0e-7, b = 1000.0),
                           reg_lambda = loguniform(a = 1.0e-7, b = 1000.0))
    obj = ModelTrainer(model_name, training_data, test_size, random_state, target_name,
                       model, hyperparameters, n_iterations = 500)
    path = obj.initiate_model_trainer()
  
    """model = DecisionTreeRegressor(random_state = 10)
    hyperparameters = dict(criterion = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                           splitter = ['best', 'random'],
                           max_depth = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                           min_samples_split = loguniform(a = 1.0/11229, b = 1.0),
                           min_samples_leaf = loguniform(a = 1.0/11229, b = 1.0),
                           max_features = ['sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           max_leaf_nodes = [i for i in range(1, 201)],
                           ccp_alpha = loguniform(a = 1.0e-8, b = 1.0e-4))"""
                           
    """model = RandomForestRegressor(n_jobs = 8, bootstrap = False, random_state = 10)
    hyperparameters = dict(n_estimators = [8, 16, 32, 64, 128, 256],
                            criterion = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                            min_samples_split = [0.05, 0.07, 0.1],
                            min_samples_leaf = [0.01, 0.025, 0.05],
                            max_features = ['sqrt', 'log2', None],
                            ccp_alpha = [0.0, 1.0e-6, 2.0e-6])"""
                            
    """model = XGBRegressor(random_state = 10)
    hyperparameters = dict(eta = loguniform(a = 1.0e-7, b = 1.0),
                           gamma = loguniform(a = 1.0e-7, b = 1000.0),
                           max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           min_child_weight = loguniform(a = 1.0e-7, b = 1000.0),
                           subsample = loguniform(a = 1.0/11229, b = 1.0),
                           colsample_bytree = loguniform(a = 1.0/11229, b = 1.0),
                           colsample_bylevel = loguniform(a = 1.0/11229, b = 1.0),
                           colsample_bynode = loguniform(a = 1.0/11229, b = 1.0),
                           reg_lambda = loguniform(a = 1.0e-7, b = 1000.0),
                           reg_alpha = loguniform(a = 1.0e-7, b = 1000.0),
                           tree_method = ['auto', 'exact', 'approx', 'hist'])"""
                           
    """model = KNeighborsRegressor(n_jobs = 8)
    hyperparameters = dict(n_neighbors = [i for i in range(1, 7100)],
                           weights = ['uniform', 'distance'],
                           algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute'],
                           leaf_size = [i for i in range(2000)])"""
                           
    """model = AdaBoostRegressor(estimator = KNeighborsRegressor(algorithm = 'auto', leaf_size = 30, metric = 'manhattan', metric_params = None, n_jobs = 1, n_neighbors = 3, p = 2, weights = 'distance'),
                              random_state = 10)
    hyperparameters = dict(n_estimators = [i for i in range(1, 501)],
                           learning_rate = loguniform(a = 1.0e-7, b = 1000.0),
                           loss = ['linear', 'square', 'exponential'])"""