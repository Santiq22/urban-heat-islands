# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:36:44 2024

@@author: Santiago Collazo
@original_author: Krish Naik
"""
# ========================================= Packages ============================================ #
import os
import sys
from exception import CustomException
from logger import logging

from dataclasses import dataclass
import numpy as np
import pandas as pd
import dill
# =============================================================================================== #

# ======================================= Main functions ======================================== #
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
# =============================================================================================== #