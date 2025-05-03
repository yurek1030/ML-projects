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
            
            # Define the models to be evaluated
            models = {
                'LinearRegression'         : LinearRegression(),
                'DecisionTreeRegressor'    : DecisionTreeRegressor(),
                'RandomForestRegressor'    : RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor'        : AdaBoostRegressor(),
                'CatBoostRegressor'        : CatBoostRegressor(verbose=0),
                'XGBRegressor'             : XGBRegressor()
            }
            
            # Define the hyperparameters for each model
            params = {
                "LinearRegression":{},
                
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1, .01, .05, .001],
                    'subsample':[0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                
                "XGBRegressor":{
                    'learning_rate':[.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                
                "CatBoostRegressor":{
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                
                "AdaBoostRegressor":{
                    'learning_rate':[.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }    
            }
            
            model_report:dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)  # Evaluate the models
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
        