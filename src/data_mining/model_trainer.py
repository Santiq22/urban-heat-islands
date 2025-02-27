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

# Machine learning
#from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,
                              GradientBoostingRegressor,
                              RandomForestRegressor)
#from sklearn.linear_model import LinearRegression
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor, XGBRFRegressor
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
                 model, parameters):
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
        
    def initiate_model_trainer(self):
        try:
            # Load dataset to train the model
            df = read_csv(self.training_data_path)
            
            logging.info("Dataset loaded")
            
            # Separate in predictors (X) and response (y)
            X = df.drop(columns = [self.response_name])
            y = df[self.response_name]
            
            # Generate training and test predictors and responses
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size,
                                                                random_state = self.random_state)
            
            logging.info("Splitted training and test input data")
            
            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train, 
                                                X_test = X_test, y_test = y_test, model = self.model,
                                                parameters = self.parameters)
            
            logging.info("Grid search CV optimization and model trained finished")
            
            # Get the trained best model from dict
            best_model = model_report['trained_model']
            
            # Get best model score from dict
            best_training_model_score = model_report['training_score']
            best_test_model_score = model_report['test_score']

            if best_training_model_score < 0.6 or best_test_model_score < 0.6:
                print("Best model training R^2:", best_training_model_score)
                print("Best model test R^2:", best_test_model_score)
                raise CustomException("No best model found", sys)
            
            logging.info("Best model found on both training and test dataset")

            # Save the model using the utils function save_object
            save_object(file_path = self.model_trainer_config.trained_model_file_path, obj = best_model)
            
            logging.info("Best model trained object saved")
            
            print("Best model training R^2:", best_training_model_score)
            print("Best model test R^2:", best_test_model_score)
            
            return self.model_trainer_config.trained_model_file_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    model_name = 'best_fit_RFR.pkl'
    training_data = '../../data/final_datasets/transformed_datasets/transformed_reduced_training_data.csv'
    test_size = 0.2
    random_state = 10
    target_name = 'UHI Index'
    model = RandomForestRegressor(n_jobs = 8, bootstrap = False, random_state = 10)
    hyperparameters = dict(n_estimators = [8, 16, 32, 64, 128, 256],
                            criterion = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                            min_samples_split = [0.05, 0.07, 0.1],
                            min_samples_leaf = [0.01, 0.025, 0.05],
                            max_features = ['sqrt', 'log2', None],
                            ccp_alpha = [0.0, 1.0e-6, 2.0e-6])
    
    obj = ModelTrainer(model_name, training_data, test_size, random_state, target_name,
                       model, hyperparameters)
    path = obj.initiate_model_trainer()

"""{"Random Forest": RandomForestRegressor(),
                      "Decision Tree": DecisionTreeRegressor(),
                      "Gradient Boosting": GradientBoostingRegressor(),
                      "Linear Regression": LinearRegression(),
                      "K-Neighbors Regressor": KNeighborsRegressor(),
                      "XGBRegressor": XGBRegressor(),
                      "CatBoost Regressor": CatBoostRegressor(verbose = False),
                      "AdaBoost Regressor": AdaBoostRegressor()}
                      
                      
{"Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2']
                    },
                    "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                    },
                    "Linear Regression":{},
                    "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                    },
                    "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                    },
                    "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                    }}"""