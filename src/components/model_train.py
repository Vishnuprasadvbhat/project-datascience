import os, sys
from dataclasses import dataclass
from sklearn.ensemble import (
  RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
)
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBRFRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import  save_object, eval_models


@dataclass 
class ModelTrainConfig:
  trainmodel_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
  def __init__(self) -> None:
    self.model_trainer_config= ModelTrainConfig()

  
  def initiate_model_training(self,train_data,test_data):
    try:
      logging.info("Spliting training and test input data ")
      x_train,y_train,x_test,y_test = (
        train_data[:,:-1],
        train_data[:,-1],
        test_data[:,:-1],
        test_data[:,-1]
      )

      models = {
        "Random Forest" : RandomForestRegressor(),
        "Decision Tree" : DecisionTreeRegressor(),
        "Adaboost" : AdaBoostRegressor(),
        "XGBoost" : XGBRegressor(),
        "XGBRFBoost": XGBRFRegressor(),
        "Kneighbors" : KNeighborsRegressor(),
        "CatBoost" : CatBoostRegressor(verbose=False),
        "Gradient Boosting": GradientBoostingRegressor(),
      }

      model_report:dict = eval_models(x_train=x_train,y_train=y_train,x_test= x_test, y_test = y_test,models=models)
      
      # get the best model score
      best_model_score = max(sorted(model_report.values()))
      
      
      #get nest model name
      best_model_name= list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]

      best_model = models[best_model_name]

      if best_model_score < 0.6:
        raise CustomException("No best model found")
      logging.info(f"Best found mode: {best_model} ")

      save_object(
        file_path=self.model_trainer_config.trainmodel_file_path, object=best_model
      )

      predicted = best_model.predict(x_test)

      r2_square = r2_score(y_test,predicted)

      return r2_square
    
    except Exception as e:
      raise CustomException(e,sys)