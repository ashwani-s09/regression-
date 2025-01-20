import os # to create a path, to join path
import sys
from loggy import logging
from excep import CustomException
import pandas as pd # to read the dataset
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # we didn't required to contruct a constructor line __init__ , we can directly create the variables using dataclass

#from src.components.data_transformation import DataTransformation
from data_transformation import DataTransformation


## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig: # we didnt required to use init as we are not creating any function inside my class , whenever i need to write fucntions inside my class then i need to use init 
    train_data_path:str=os.path.join('artifacts','train.csv') # os.path is the directory path 
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self): # i need to know the path that's why i am initialising it
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self): # this is the function
        logging.info('Data Ingestion methods Starts')
        try:
            df=pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)


## run data ingestion

if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)