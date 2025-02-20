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
# This class will provide the paths for the inputs to the data transformation process
@dataclass
class DataTransformationConfig:
    # Training transformed dataset filepath
    training_dataset_file_path = os.path.join('../../data/final_datasets/transformed_datasets', 'transformed_training_data.csv')
    
    # Test transformed dataset filepath
    test_dataset_file_path = os.path.join('../../data/final_datasets/transformed_datasets', 'transformed_test_data.csv')
    
# Class to set the inputs
class DataTransformation:
    def __init__(self, training_path, test_path, target_column_name):
        # Attribute representing the transformed filepaths
        self.data_transformation_config = DataTransformationConfig()
        
        # Path to the raw training dataset
        self.training_path = training_path
        
        # Path to the raw test dataset
        self.test_path = test_path
        
        # Response variable name
        self.target_column_name = target_column_name
        
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
            
            # Process of combining both pipelines to transform the training dataset without the responses
            preprocessor_training = ColumnTransformer([("numerical_pipeline", numerical_pipeline, training_df.columns[:-1])])
            
            # Process of combining both pipelines to transform the test dataset without the responses
            preprocessor_test = ColumnTransformer([("numerical_pipeline", numerical_pipeline, test_df.columns)])
            
            logging.info("Preprocessor objects created")
            
            # Training predictors
            input_feature_training_df = training_df.drop(columns = [self.target_column_name], axis = 1)
            
            # Training response
            target_feature_training_df = training_df[self.target_column_name]
            
            logging.info("Applying preprocessing object on training data and test data")
            
            # Fit and transform the training predictors
            input_feature_training_arr = preprocessor_training.fit_transform(input_feature_training_df)
            
            # Transform the test predictors
            input_feature_test_arr = preprocessor_test.fit_transform(test_df)
            
            logging.info("Transformations correctly done")
            
            # Concatenate training arrays along their second axis
            training_arr = c_[input_feature_training_arr, array(target_feature_training_df)]
            
            # Convert transformed training dataset into DataFrame object
            training_df_new = DataFrame(training_arr)
            
            # Rename the last column of the training data
            training_df_new.rename(columns = {training_df_new.columns[-1]: self.target_column_name},
                                   inplace = True)
            
            # Convert transformed test dataset into DataFrame object
            test_df_new = DataFrame(input_feature_test_arr)
            
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
    response_name = 'UHI Index'
    
    obj = DataTransformation(training, test, response_name)
    data = obj.initiate_data_transformation()