import os
import sys
from src.exception import CustomException
from src.logger import log
from src import utils
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    project_path = utils.getPath()
    train_data_path: str=os.path.join(project_path,'artifacts','train.csv')
    test_data_path: str=os.path.join(project_path,'artifacts','test.csv')
    raw_data_path: str=os.path.join(project_path,'artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        log('Entered the data ingestion method or component')
        try:
            df = pd.read_csv(os.path.join(self.ingestion_config.project_path,'notebook','data','stud.csv'))
            log('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            log('Train test split initiated')
            train_set, test_set = tts(df, test_size=.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            log('Ingestion of the data is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    train_data, test_data = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_data, test_data)
    best_model, score = ModelTrainer().initiate_model_trainer(train_arr, test_arr)