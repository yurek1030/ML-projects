import os
import sys
import dill
import numpy  as np
import pandas as pd
from src.exception         import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)  # Get the directory path of the file
        os.makedirs(dir_path, exist_ok=True)   # Create the directory if it doesn't exist
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)   # Save the object to the specified file path
            
    except Exception as e:
        raise CustomException(e, sys)  # Raise a custom exception if an error occurs
    