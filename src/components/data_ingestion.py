import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from dataclasses import dataclass


@dataclass
class DataIngestionCongif:
  train_data_path : str=
