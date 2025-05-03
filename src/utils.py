import os
import sys
import dill
import numpy  as np
import pandas as pd
from src.exception           import CustomException
from sklearn.metrics         import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)  # Get the directory path of the file
        os.makedirs(dir_path, exist_ok=True)   # Create the directory if it doesn't exist
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)   # Save the object to the specified file path
            
    except Exception as e:
        raise CustomException(e, sys)  # Raise a custom exception if an error occurs
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]        # Get the model from the dictionary
            param = params[list(models.keys())[i]]  # Get the parameters for the model
            
            grid_search = GridSearchCV(model, param_grid=param, cv=3)  # Perform grid search for hyperparameter tuning
            grid_search.fit(X_train, y_train)                          # Fit the grid search to the training data
            
            model.set_params(**grid_search.best_params_)  # Set the best parameters to the model
            model.fit(X_train, y_train)                   # Fit the model to the training data
            
            y_train_pred = model.predict(X_train)  # Make predictions on the training data
            y_test_pred  = model.predict(X_test)   # Make predictions on the testing data
            
            train_model_score = r2_score(y_train, y_train_pred) # Calculate R2 score for training data
            test_model_score  = r2_score(y_test, y_test_pred)   # Calculate R2 score for testing data
            
            report[list(models.keys())[i]] = test_model_score   # Store the test score in the report dictionary
            
        return report  # Return the report containing the model scores
            
    except Exception as e:
        raise CustomException(e, sys)  # Raise a custom exception if an error occurs
    