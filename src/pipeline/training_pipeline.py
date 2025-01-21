import os
import sys
from logger import logging
from exception import CustomException
import pandas as pd

from data_ingestions import DataIngestion
from data_transformations import DataTransformation
from model_trainers import ModelTrainer


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)
