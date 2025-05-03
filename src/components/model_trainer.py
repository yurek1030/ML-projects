import os 
import sys
from dataclasses          import dataclass
from sklearn.metrics      import r2_score
from sklearn.ensemble     import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors    import KNeighborsRegressor
from sklearn.tree         import DecisionTreeRegressor
from catboost             import CatBoostRegressor
from xgboost              import XGBRegressor
from src.exception        import CustomException
from src.logger           import logging
from src.utils            import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')  # Path to save the trained model
     
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()  # Initialize the ModelTrainerConfig to set up file paths

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")  # Log the start of the model training process
            
            X_train, y_train, X_test, y_test = (              # Split the data into features and target variable
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1],  test_array[:, -1]
            )
            
            models = {
                'LinearRegression'         : LinearRegression(),
                'DecisionTreeRegressor'    : DecisionTreeRegressor(),
                'RandomForestRegressor'    : RandomForestRegressor(),
                'KNeighborsRegressor'      : KNeighborsRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor'        : AdaBoostRegressor(),
                'CatBoostRegressor'        : CatBoostRegressor(verbose=0),
                'XGBRegressor'             : XGBRegressor()
            }
            
            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models=models)               # Evaluate the models
            best_model_score = max(sorted(model_report.values()))                                              # Get the best model score
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]   # Get the name of the best model
            best_model = models[best_model_name]                                                               # Get the best model object
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found on both training and testing dataset")  # Log the best model found
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,  # Save the best model to the specified path
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)   # Make predictions using the best model
            r2_square = r2_score(y_test, predicted)  # Calculate the R-squared score of the predictions
            
            return r2_square
                    
        except Exception as e:
            raise CustomException(e, sys)  # Raise a custom exception if an error occurs
        