import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

# initialize the data ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts', 'test.csv')
    raw_data_path : str = os.path.join('artifacts', 'raw.csv')


# create a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestion()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion methods started.')
        try:
            df = pd.read_csv(os.path.join('notebooks','data','finalTrain.csv'))

            logging.info('Dataset read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info('Train test split started')

            train_set, test_set = train_test_split(df, test_state=0.3, random_state=42)

            logging.info('Train test split completed, starting to export train data and test data.')

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)

            logging.info('Ingestion of data is completed.')


            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at data ingestion stage.')
            raise CustomException(e,sys)
