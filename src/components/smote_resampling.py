# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from numpy import unique
from pandas import read_csv, cut, concat
from imblearn.over_sampling import SMOTE
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataSMOTEResamplingConfig:
    def __init__(self):
        # Path to output datasets
        self.data_path: str = os.path.join('../../data/final_datasets/transformed_datasets/', 'transformed_reduced_smote_training_data.csv')
        
class DataSMOTEResampling:
    def __init__(self, path_to_data, bins, rs):
        # This variable will consist in the input I need to initialize
        self.smote_resampling_config = DataSMOTEResamplingConfig()
        
        # Path to the dataset
        self.path_to_data = path_to_data
        
        # Number of bins
        self.bins = bins
        
        # Random state to pass to SMOTE class
        self.random_state = rs
        
    def initiate_smote_resampling(self):
        logging.info("Entered the data resampling method or component")
        try:
            # Load the dataset
            df = read_csv(self.path_to_data)
            
            logging.info("Dataset loaded")
        
            # Bin the data
            uhi_binned = cut(df['UHI Index'], bins = self.bins, include_lowest = True)
            
            logging.info("Bins created")
            
            # List of binned UHI indeces
            binned_uhi_indeces = []

            # Assign a binned value of the UHI index based on its interval
            for interval in uhi_binned:
                binned_uhi_indeces.append(str((interval.right + interval.left)/2.0))
                
            logging.info("Bins values assigned to each datapoint")
                
            # Save array of binned UHI indeces as a column
            df['UHI Index binned'] =  binned_uhi_indeces
            
            # Get the max number of samples in a bin
            max_samples_in_a_bin = unique(df['UHI Index binned'].values, return_counts = True)[1].max()
            
            # Generate the dictionary of classes about which to perform the SMOTE resampling
            dict_of_classes = {c : max_samples_in_a_bin for c in unique(df['UHI Index binned'].values)}
            
            # Split variables in predictors and responses
            X = df.drop(columns = ['UHI Index', 'UHI Index binned'], axis = 1)
            y = df['UHI Index binned']
            
            logging.info("Variables splitted in predictors and responses")

            # Resampling the minority class. The strategy can be changed as required.
            sm = SMOTE(sampling_strategy = dict_of_classes, random_state = self.random_state)
            
            logging.info("SMOTE object instantiated")

            # Fit the model to generate the data
            X_SMOTE, y_SMOTE = sm.fit_resample(X, y)
            
            logging.info("SMOTE resampling done")
            
            # SMOTE DataFrame
            df_smote = concat((X_SMOTE, y_SMOTE), axis = 1)
            
            # Convert the column from str to float
            df_smote['UHI Index binned'] = df_smote['UHI Index binned'].astype(float)
            
            # Save the new dataset
            df_smote.to_csv(self.smote_resampling_config.data_path, index = False)
            
            logging.info("SMOTE dataset correctly saved")
            
            return self.smote_resampling_config.data_path
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    path = '../../data/final_datasets/transformed_datasets/transformed_reduced_training_data.csv'
    bins = 75
    random_state = 10
    
    obj = DataSMOTEResampling(path, bins, random_state)
    path_to_data = obj.initiate_smote_resampling()