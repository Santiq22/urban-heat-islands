# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

# Data Science
from pandas import read_csv, DataFrame
from sklearn.decomposition import PCA
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class PCATransformationConfig:
    def __init__(self, file_name):
        # Path to output dataset
        self.file_path: str = os.path.join('../../data/final_datasets/transformed_datasets/pca_datasets/', file_name)
        
class PCATransformation:
    def __init__(self, input_path, n_features, random_state, output_name):
        # Attribute representing the output filepath
        self.pca_transformation_config = PCATransformationConfig(output_name)
        
        # Filepath of the input dataset
        self.input_path = input_path
        
        # Number of variables after transformation
        self.n_features = n_features
        
        # Random state to set up the PCA class
        self.random_state = random_state
        
    def initiate_pca_transformation(self):
        logging.info("PCA data transformation process started")
        
        try:
            # Read the input dataset
            df = read_csv(self.input_path)
            
            logging.info("Dataset correctly read")
            
            # Get predictors to transform
            if df.columns[-1] == 'UHI Index':
                X = df.drop(columns = ['UHI Index']).values
            else:
                X = df.values
            
            # Set up the PCA object
            pca_transformer = PCA(n_components = self.n_features, random_state = self.random_state)
            
            logging.info("PCA object instantiated")
            
            # Fit the PCA object with the input data
            pca_transformer.fit(X)
            
            logging.info("PCA object fitted")
            
            # Transform the data
            X_new = pca_transformer.transform(X)
            
            logging.info("Data transformed through PCA")
            
            # Create a DataFrame object
            df_new = DataFrame(X_new)
            if df.columns[-1] == 'UHI Index':
                df_new['UHI Index'] = df['UHI Index'].values
            
            # Save the new dataframe
            df_new.to_csv(self.pca_transformation_config.file_path, index = False)
            
            return self.pca_transformation_config.file_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == '__main__':
    input = '../../data/final_datasets/transformed_datasets/transformed_reduced_test_data.csv'
    n_features = None
    random_state = 10
    output = 'transformed_reduced_pca_test_data.csv'
    
    obj = PCATransformation(input, n_features, random_state, output)
    data = obj.initiate_pca_transformation()