import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet

@dataclass
class ModeltrainerConfig:
    trainer_model_file_path = os.path.join('artifacts', 'model.pkl')


class Modeltrainer:

    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Splitting dependent and independent variables from train and test array')
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            hyp_par= {
                'max_depth': 10,
                'min_samples_leaf': 5,
                'min_samples_split': 8,
                'n_estimators': 20
                }
            

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'ElasticNet':ElasticNet(),
            'RandomForest':RandomForestRegressor(**hyp_par)
            }
        
            report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print(report)
            print('\n')
            print('-'*40)
            logging.info(f"Model Report : {report}")

            # to get best model score from dictionary
            best_model_score = max(sorted(report.values()))

            best_model_name = list(report.keys())[list(report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trainer_model_file_path,
                 obj=best_model
            )
        except Exception as e:
            logging.info('Error occured at Model training.')
            raise CustomException(e,sys)