import sys
import os
import logging
from dataclasses import dataclass
from data_ingestion import DataIngestion
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
data_ingestion = DataIngestion()

data_ingestion.ingestion_config.train_data_path

@dataclass
class DatatransformationConfig: 
  preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class Datatranformation:
  def __init__(self):
    self.data_transformation_config = DatatransformationConfig()


  def get_transformer_object(self):
    """ 
    This is function is responsible for transformation of Data into a numerical format 
    """
    try:
      numerical_cols = ['writing_score', 'reading_score']
      categorical_loop = [
        'gender',
        'race_ethnicity',
        'parental_level_of_education',
        'lunch',
        'test_preparation_course'
      ]
      num_pipeline = Pipeline(
        steps=[
          "imputer", SimpleImputer(),
          "scaler", StandardScaler(),   
          # "one_hot", OneHotEncoder(),
          ])

      cat_pipeline = Pipeline(
        steps = [
          ("imputer", SimpleImputer(strategy = 'most_frequent')),
          ("one_hot_encoder", OneHotEncoder()),
          ("scaler", StandardScaler())
          ])    
      logging.info(f'Numerical columns {numerical_cols}')

      logging.info(f'Categorical Columns {categorical_loop}')

      preprocess =ColumnTransformer(
        [
          ('num_pipeline', num_pipeline,numerical_cols),
          ('cat_pipeline', cat_pipeline, categorical_loop)
        ]
      )

      return preprocess
    except Exception as e:
      raise CustomException(e,sys)
  

  def initiate_data_transformation(self, train_path, test_path):
    try:
      train_df = pd.read_csv('artifacts\train.csv')
      test_df = pd.read_csv('artifacts\test.csv')

      logging.info('read training and testing data')

      logging.info('Obtaining preprocessing oject')

      preprocessing_obj = self.get_transformer_object()

      target_columns_name ='math_score'

      numerical_col = ['writing_score', 'reading_score']

      input_feature_train_df = train_df.drop(columns=[target_columns_name],axis=1)
      target_feature_train_df = train_df[target_columns_name]


      input_feature_test_df = test_df.drop(columns=[target_columns_name],axis=1)
      target_feature_test_df = test_df[target_columns_name]

      logging.info(
        f"Applying preprocessing object on training dataframe and testing dataframe"
      )

      input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
      input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

      train_arr = np.c_[
        input_feature_train_arr, np.array(target_feature_train_df)
      ]

      test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

      logging.info(f'Saved preprocessing object')

      save_object(

        file_path = self.data_transformation_config.preprocessor_obj_file_path,
        obj = preprocessing_obj
      )
      return (
        train_arr,
        test_arr,
        self.data_transformation_config.preprocessor_obj_file_path,
      )
    except:
      pass


