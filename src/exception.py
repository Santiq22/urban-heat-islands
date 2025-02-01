# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:36:20 2024

@author: Santiago Collazo
@original_author: Krish Naik
"""
# ================================ Packages ================================== #
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), '../', '../'))
from src.logger import logging
# ============================================================================ #

# ========================= Error message function =========================== #
def error_message_detail(error, error_detail:sys):
    # Information at execution time
    _, _, exc_tb = error_detail.exc_info()
    
    # Name of the file where the error has occured
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Error message
    error_message = "Error occured in python script name [{0}], line number [{1}], with error message [{2}]".format(file_name, exc_tb.tb_lineno, str(error))

    return error_message
# ============================================================================ #

# ========================= Custom exception class =========================== #
class CustomException(Exception):
    """ Custom exception class. It inherits from the 'Exception' class. """
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        
        # Error message instance variable outcome of the error_message_detail function
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    # This method is called when the print function is called on an object of this class
    def __str__(self):
        return self.error_message
# ============================================================================ #