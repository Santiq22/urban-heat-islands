# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:36:44 2024

@@author: Santiago Collazo
@original_author: Krish Naik
"""
# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../'))
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
r2 = make_scorer(r2_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearnex import patch_sklearn
patch_sklearn()
# =============================================================================================== #

# ======================================= Main functions ======================================== #
# Save an object in a given file path
def save_object(file_path, obj):
    try:
        # Directory path where is located file_path
        dir_path = os.path.dirname(file_path)
        
        # Create recursively all directories needed to contain dir_path
        os.makedirs(dir_path, exist_ok = True)
        
        # Open file_path in write and byte modes and pickle obj to file_obj
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, model, parameters, n_iterations = None):
    try:
        # Instantiate model
        model_obj = model

        # Instantiate a GridSearchCV object using 5-fold CV or a RandomizedSearchCV
        if n_iterations == None:
            gs = GridSearchCV(model_obj, parameters, scoring = r2, cv = 5, n_jobs = 8, verbose = 2)
        else:
            gs = RandomizedSearchCV(model_obj, parameters, n_iter = n_iterations, 
                                    scoring = r2, cv = 5, n_jobs = 8, verbose = 2)
            
        # Perform a grid or randomized search cv
        gs.fit(X_train, y_train)

        # Set best fit parameters to the corresponding model
        model_obj.set_params(**gs.best_params_)
            
        # Train the model
        model_obj.fit(X_train, y_train)
            
        # Predictions over training data
        y_train_pred = model_obj.predict(X_train)

        # Predictions over test data
        y_test_pred = model_obj.predict(X_test)

        # R^2 over training data
        training_model_score = r2_score(y_train, y_train_pred)

        # R^2 over test data
        test_model_score = r2_score(y_test, y_test_pred)

        # Save the trained model, training and test R^2 into the report dictionary
        report = {'trained_model': model_obj, 'training_score': training_model_score, 
                  'test_score': test_model_score, 'best_fit_hyperparameters': gs.best_params_}

        return report

    except Exception as e:
        raise CustomException(e, sys)
# =============================================================================================== #