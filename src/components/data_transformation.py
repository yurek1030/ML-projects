import os
import sys
import numpy  as np
import pandas as pd
from dataclasses           import dataclass
from sklearn.compose       import ColumnTransformer
from sklearn.impute        import SimpleImputer
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception         import CustomException
from src.logger            import logging
from src.utils             import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')  # Path to save the preprocessor object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This method creates a data transformation pipeline using ColumnTransformer.
        '''
        try:
           numerical_features = ['writing_score', 'reading_score']  # List of numerical feature columns
           categorical_features = [                                 # List of categorical feature columns
               'gender',
               'race_ethnicity',
               'parental_level_of_education',
               'lunch',
               'test_preparation_course'
           ]
           
           num_pipeline = Pipeline(steps=[
               ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with the median
               ('scaler',  StandardScaler())                   # Scale numerical features using StandardScaler
           ])
           
           cat_pipeline = Pipeline(steps=[
               ('imputer',         SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
               ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))    # One-hot encode categorical features
           ])
           
           logging.info(f'Categorical columns: {categorical_features}')
           logging.info(f'Numerical columns: {numerical_features}')
           
           preprocessor = ColumnTransformer(
               transformers=[
                   ('num_pipeline', num_pipeline, numerical_features),   # Apply numerical pipeline to numerical features
                   ('cat_pipeline', cat_pipeline, categorical_features)  # Apply categorical pipeline to categorical features
               ]
           )
           
           return preprocessor # Return the preprocessor object
           
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This method initiates the data transformation process.
        '''
        try:
            train_df = pd.read_csv(train_path)  # Read the training data
            test_df  = pd.read_csv(test_path)   # Read the testing data
            
            logging.info('Read train and test data completed')
            
            logging.info('Obtaining preprocessor object')
            preprocessor_obj = self.get_data_transformer_object()    # Get the preprocessor object
            
            target_column_name = 'math_score'                        # Define the target column name
            numerical_features = ['writing_score', 'reading_score']  # List of numerical feature columns
            
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)  # Drop target column from training data
            target_feature_train_df = train_df[target_column_name]                         # Extract target column from training data
            
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)    # Drop target column from testing data
            target_feature_test_df = test_df[target_column_name]                           # Extract target column from testing data
            
            logging.info('Applying preprocessing object on training and testing data')
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_df)  # Fit the preprocessor on training data
            input_feature_test_arr  = preprocessor_obj.transform(input_features_test_df)       # Transform the testing data using the fitted preprocessor
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]  # Combine input features and target for training data
            test_arr  = np.c_[input_feature_test_arr,  np.array(target_feature_test_df)]   # Combine input features and target for testing data
            
            logging.info('Saving preprocessing object')
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)