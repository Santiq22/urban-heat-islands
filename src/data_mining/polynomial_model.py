# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from pandas import read_csv, concat, DataFrame
from sklearn.preprocessing import PolynomialFeatures
# =============================================================================================== #

# ======================================== Main classes ========================================= #
# This class will provide the paths for the outputs to the data transformation process
@dataclass
class DataTransformationConfig:
    def __init__(self, training_file_name, test_file_name):
        # Training transformed dataset filepath
        self.training_dataset_file_path = os.path.join('../../data/final_datasets/transformed_datasets/interactions_datasets', training_file_name)
        
        # Test transformed dataset filepath
        self.test_dataset_file_path = os.path.join('../../data/final_datasets/transformed_datasets/interactions_datasets', test_file_name)

# Class to set the inputs and perform the transformation
class DataTransformation:
    def __init__(self, training_path, test_path, training_output, test_output, columns_to_interact, degree):
        # Attribute representing the transformed filepaths
        self.data_transformation_config = DataTransformationConfig(training_output, test_output)
        
        # Path to the input training dataset
        self.training_path = training_path
        
        # Path to the input test dataset
        self.test_path = test_path
        
        # List of columns to form the interactions
        self.columns_to_interact = columns_to_interact
        
        # Degree of the polynomial
        self.degree = degree
        
    # This function initiates the data transformation process
    def initiate_data_transformation(self):
        logging.info("Initiate the process of creating interactions between columns")
        
        try:
            # Load the input datasets
            df_training = read_csv(self.training_path)
            df_test = read_csv(self.test_path)
            
            logging.info("Training and test datasets loaded")
            
            # Choose the predictors to form interactions
            df_training_tmp = df_training[self.columns_to_interact]
            df_test_tmp = df_test[self.columns_to_interact]
            
            # Instantiate a PolynomialFeatures object
            poly = PolynomialFeatures(degree = self.degree, interaction_only = True, include_bias = False)
            
            # Fit and transform the PolynomialFeatures object
            X_training_poly = poly.fit_transform(df_training_tmp.values)
            X_test_poly = poly.transform(df_test_tmp.values)
            
            logging.info("Transformation completed")
            
            # Convert array to DataFrame objects
            X_training_poly = DataFrame(X_training_poly)
            X_test_poly = DataFrame(X_test_poly)
            
            # Add polynomial features for key predictors
            X_training_new = concat((df_training.drop(columns = self.columns_to_interact + ['UHI Index']),
                                     X_training_poly,
                                     df_training['UHI Index']), axis = 1)
            X_test_new = concat((df_test.drop(columns = self.columns_to_interact),
                                 X_test_poly), axis = 1)
            
            # Save the new datasets
            X_training_new.to_csv(self.data_transformation_config.training_dataset_file_path, index = False)
            X_test_new.to_csv(self.data_transformation_config.test_dataset_file_path, index = False)
            
            logging.info("New datasets with interaction terms correctly saved")            
            
            return (self.data_transformation_config.training_dataset_file_path,
                    self.data_transformation_config.test_dataset_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    training = '../../data/final_datasets/transformed_datasets/transformed_reduced_training_data.csv'
    test = '../../data/final_datasets/transformed_datasets/transformed_reduced_test_data.csv'
    training_output = 'interactions_reduced_training_data.csv'
    test_output = 'interactions_reduced_test_data.csv'
    interactions = ['lst_median_res250', 
                    'lst_median_res500',
                    'polygon_density_250', 
                    'polygon_density_500',
                    'mean_number_of_floors_250', 
                    'mean_number_of_floors_500',
                    'units_density_250', 
                    'units_density_500', 
                    'population_density_250',
                    'population_density_500']
    degree = 2
    
    obj = DataTransformation(training, test, training_output, test_output, interactions, degree)
    data = obj.initiate_data_transformation()
    
"""features_to_bin = ['mean_number_of_floors_250', 
                   'mean_number_of_floors_500',
                   'std_number_of_floors_250', 
                   'std_number_of_floors_500']

kbins = KBinsDiscretizer(n_bins = 3, encode = 'onehot-dense', strategy = 'kmeans')
X_binned = kbins.fit_transform(X[features_to_bin])

X_test_binned = kbins.transform(test_df[features_to_bin])

X_enhanced_bins = hstack((X.values, X_binned))
X_test_enhanced_bins = hstack((test_df.values, X_test_binned))

X_combined = hstack((X_enhanced, X_binned))
X_test_combined = hstack((X_test_enhanced, X_test_binned))"""