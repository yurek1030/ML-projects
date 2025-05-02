import os
import sys
import pandas as pd
from src.exception                      import CustomException
from src.logger                         import logging
from sklearn.model_selection            import train_test_split
from dataclasses                        import dataclass
from src.components.data_transformation import DataTransformationConfig, DataTransformation

@dataclass
class DataIngestionConfig:
    # Configuration class to define paths for train, test and raw data files
    train_data_path: str = os.path.join('artifacts', 'train.csv')  # Path to save the training data
    test_data_path:  str = os.path.join('artifacts', 'test.csv')   # Path to save the testing data
    raw_data_path:   str = os.path.join('artifacts', 'data.csv')   # Path to save the raw data

class DataIngestion:
    def __init__(self):
        # Initialize the DataIngestionConfig to set up file paths
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        # Method to handle the data ingestion process
        logging.info("Entering the data ingestion method or component")    
        try:
            # Read the dataset into a pandas DataFrame
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as dataframe")
            
            # Create directories for saving the data if they do not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,   index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()                                                    # Create an instance of the DataIngestion class
    train_data, test_data = obj.initiate_data_ingestion()                    # Get the paths for train and test data
    
    data_transformation = DataTransformation()                               # Create an instance of the DataTransformation class
    data_transformation.initiate_data_transformation(train_data, test_data)  # Get the data transformer object 
    
