# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:36:03 2024

@author: Santiago Collazo
@original_author: Krish Naik
"""
# ================================ Packages ================================== #
import logging
import os
from datetime import datetime
# ============================================================================ #

# ==================== Definition of path and filenames ====================== #
# Log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Path to log files
logs_path = os.path.join('/mnt/c/Programacion/Data Science Competitions/urban-heat-islands/logs')

# Create directory to store log files
os.makedirs(logs_path, exist_ok = True)

# Location of the log files
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
# ============================================================================ #

# ============================= Set up logging =============================== #
logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
    )
# ============================================================================ #