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

  
  def initiate_model_training(self,train_data, test_data,preprocess_path):
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
    except:
      pass