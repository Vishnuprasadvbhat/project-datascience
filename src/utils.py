import os
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def save_object(file_path,object):
  try:
    dir_path = os.path.dirname(file_path)


    os.makedirs(dir_path,exist_ok=True)

    with open(file_path, "wb") as file_obj:
      dill.dump(object,file_obj)

  except Exception as e:
    raise CustomException(e,sys)
  


def eval_models(x_train,y_train,x_test,y_test,models):
  try:
    report ={}

    for i in range(len(list(models))):
      model = list(models.values())[i]
      # param = list(params.values())[i]

      # Due to Hyperparametering issue Could add parameters

      # gridSearch = GridSearchCV(model,param_grid=params,cv=3,)
      # gridSearch.fit(x_train,y_train) #training the grid search instance

      # model.set_params(**gridSearch.best_params_)

      model.fit(x_train,y_train) # training the model with best params

      
      y_train_pred = model.predict(x_train)

      y_test_pred = model.predict(x_test)  

      train_model_score = r2_score(y_train,y_train_pred)

      test_model_score = r2_score(y_test,y_test_pred)

      report[list(models.keys())[i]]= test_model_score

      return report

  except Exception as e:
    raise CustomException(e,sys)
