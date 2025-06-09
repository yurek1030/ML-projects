import sys
import pandas as pd
from src.exception       import CustomException
from src.utils           import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    # Load the preprocessor and model
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'                       # Path to the saved model
            preprocessor_path = 'artifacts/preprocessor.pkl'         # Path to the saved preprocessor
            
            model = load_object(file_path=model_path)                # Load the model using the utility function
            preprocessor = load_object(file_path=preprocessor_path)  # Load the preprocessor
            
            data_scaled = preprocessor.transform(features)           # Scale the input features using the preprocessor
            preds = model_prediction = model.predict(data_scaled)    # Make predictions using the model
            
            return preds  # Return the predictions
        
        except Exception as e:
            raise CustomException(e, sys)  # Raise a custom exception if an error occurs
        
    
class CustomData:
    def __init__(self,
        gender                     : str,
        race_ethnicity             : str,
        parental_level_of_education: str,
        lunch                      : str,
        test_preparation_course    : str,
        reading_score              : int,
        writing_score              : int,
    ):
        self.gender                      = gender
        self.race_ethnicity              = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch                       = lunch
        self.test_preparation_course     = test_preparation_course
        self.reading_score               = reading_score
        self.writing_score               = writing_score
    
    # Convert the custom data input to a DataFrame    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'gender'                     : [self.gender],
                'race_ethnicity'             : [self.race_ethnicity],      
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch'                      : [self.lunch],
                'test_preparation_course'    : [self.test_preparation_course],
                'reading_score'              : [self.reading_score],
                'writing_score'              : [self.writing_score],
            }
            
            return pd.DataFrame(custom_data_input_dict)  # Convert the dictionary to a DataFrame and return it
        
        except Exception as e:
            raise CustomException(e, sys)  # Raise a custom exception if an error occurs
        