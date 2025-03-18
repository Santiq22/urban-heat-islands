# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from numpy import c_, array
from pandas import read_csv, DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# =============================================================================================== #

# ======================================== Main classes ========================================= #
# This class will provide the paths for the outputs to the data transformation process
@dataclass
class DataTransformationConfig:
    def __init__(self, training_file_name, test_file_name):
        # Training transformed dataset filepath
        self.training_dataset_file_path = os.path.join('../../data/final_datasets/transformed_datasets', training_file_name)
        
        # Test transformed dataset filepath
        self.test_dataset_file_path = os.path.join('../../data/final_datasets/transformed_datasets', test_file_name)
    
# Class to set the inputs
class DataTransformation:
    def __init__(self, training_path, test_path, training_output, test_output, target_column_name,
                 columns_not_to_transform):
        # Attribute representing the transformed filepaths
        self.data_transformation_config = DataTransformationConfig(training_output, test_output)
        
        # Path to the raw training dataset
        self.training_path = training_path
        
        # Path to the raw test dataset
        self.test_path = test_path
        
        # Response variable name
        self.target_column_name = target_column_name
        
        # Columns no to apply the transformation pipeline. It is of the kind [(str, str), ...]
        self.columns_not_to_transform = columns_not_to_transform
        
    # This function initiates the data transformation process
    def initiate_data_transformation(self):
        try:
            # Load raw training dataset
            training_df = read_csv(self.training_path)
            
            # Load raw test dataset
            test_df = read_csv(self.test_path)
            
            logging.info("Reading of training and test data completed")
            
            # Pipeline to apply to training and test numerical predictors
            numerical_pipeline = Pipeline(steps = [("imputer", SimpleImputer(strategy = "mean")),
                                                   ("scaler", StandardScaler())])
            
            logging.info("Pipeline object instantiated")
            
            # Remove columns not to transform
            training_columns = training_df.columns[:-1].values.tolist()
            training_columns_to_drop = []
            test_columns = test_df.columns.values.tolist()
            test_columns_to_drop = []
            for item in self.columns_not_to_transform:
                if item[0] == 'training':
                    training_columns.remove(item[1])
                    training_columns_to_drop.append(item[1])
                else:
                    test_columns.remove(item[1])
                    test_columns_to_drop.append(item[1])
            
            # Process of combining both pipelines to transform the training dataset without the responses
            preprocessor_training = ColumnTransformer([("numerical_pipeline", numerical_pipeline, training_columns)])
            
            # Process of combining both pipelines to transform the test dataset without the responses
            preprocessor_test = ColumnTransformer([("numerical_pipeline", numerical_pipeline, test_columns)])
            
            logging.info("Preprocessor objects created")
            
            # Training predictors
            input_feature_training_df = training_df.drop(columns = [self.target_column_name] + training_columns_to_drop,
                                                         axis = 1)
            
            # Training response
            target_feature_training_df = training_df[self.target_column_name]
            
            # Test predictors
            input_feature_test_df = test_df.drop(columns = test_columns_to_drop, axis = 1)
            
            logging.info("Applying preprocessing object on training data and test data")
            
            # Fit and transform the training predictors
            input_feature_training_arr = preprocessor_training.fit_transform(input_feature_training_df)
            
            # Transform the test predictors
            input_feature_test_arr = preprocessor_test.fit_transform(input_feature_test_df)
            
            logging.info("Transformations correctly done")
            
            # Concatenate training arrays along their second axis ---------------------------------
            if len(training_columns_to_drop) == 0:
                training_arr = c_[input_feature_training_arr, 
                                array(target_feature_training_df)]
            else:
                training_arr = c_[input_feature_training_arr, 
                                training_df[training_columns_to_drop].values,
                                array(target_feature_training_df)]
                
            # Convert transformed training dataset into DataFrame object
            training_df_new = DataFrame(training_arr)
            
            # Set a dict of new names for training data
            new_names_training = {col_old: col_new  for col_old, col_new in 
                                  zip(training_df_new.columns, training_columns + training_columns_to_drop + [self.target_column_name])}
            
            # Rename the columns of the training data
            training_df_new.rename(columns = new_names_training, inplace = True)
            # -------------------------------------------------------------------------------------
            
            # Concatenate test arrays along their second axis -------------------------------------
            if len(test_columns_to_drop) != 0:
                test_arr = c_[input_feature_test_arr, test_df[test_columns_to_drop].values]
        
            # Convert transformed test dataset into DataFrame object
            test_df_new = DataFrame(test_arr)
            
            # Set a dict of new names for test data
            new_names_test = {col_old: col_new  for col_old, col_new in 
                              zip(test_df_new.columns, test_columns + test_columns_to_drop)}
            
            # Rename the columns of the test data
            test_df_new.rename(columns = new_names_test, inplace = True)
            # -------------------------------------------------------------------------------------
            
            # Save both datasets
            training_df_new.to_csv(self.data_transformation_config.training_dataset_file_path, index=False)
            test_df_new.to_csv(self.data_transformation_config.test_dataset_file_path, index=False)
            
            logging.info("Training and test datasets saved")
            
            return (self.data_transformation_config.training_dataset_file_path,
                   self.data_transformation_config.test_dataset_file_path)
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    training = '../../data/final_datasets/raw_datasets/raw_training_data.csv'
    test = '../../data/final_datasets/raw_datasets/raw_test_data.csv'
    training_output = 'transformed_training_data.csv'
    test_output = 'transformed_test_data.csv'
    response_name = 'UHI Index'
    columns = [('training', 'air_temperature_at_surface [degC]'),
               ('training', 'temperature'),
               ('training', 'pressure_3hr_change'),
               ('training', 'temperature_3h_moving_avg'), 
               ('training', 'temperature_6h_moving_avg'), 
               ('training', 'temperature_12h_moving_avg'),
               ('test', 'air_temperature_at_surface [degC]'),
               ('test', 'temperature'),
               ('test', 'pressure_3hr_change'),
               ('test', 'temperature_3h_moving_avg'), 
               ('test', 'temperature_6h_moving_avg'), 
               ('test', 'temperature_12h_moving_avg')]
    
    obj = DataTransformation(training, test, training_output, test_output, response_name, columns)
    data = obj.initiate_data_transformation()