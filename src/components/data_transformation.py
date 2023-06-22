import os 
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from math import radians, sin, cos, sqrt, atan2

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')
    numeric_features = [
                'Delivery_person_Age','Delivery_person_Ratings','multiple_deliveries',
                'Distance','Time_Orderd_Hour','Time_Orderd_Minutes','Time_Order_picked_Hour',
                'Time_Order_picked_Minutes','Order_Date_Day','Order_Date_Month','Order_Date_Year',
                'Order_Date_DayOfWeek','Order_Date_DayOfYear'
                ]
            
    categorical_features = [
                'Weather_conditions', 'Road_traffic_density', 'Type_of_order',
                'Type_of_vehicle', 'Festival', 'City'
                ]

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def handle_time_feature(self,df):
        time_features = ['Time_Orderd','Time_Order_picked']
        # creating multiple columns of hours and minutes
        for time_feature in time_features:
            df[time_feature] = df[time_feature].apply(lambda x: str(int(float(x) * 24)) + ':' + str(int((float(x) * 24 * 60) % 60)) if pd.notna(x) and ':' not in str(x) else x)
            df[time_feature+'_Hour'] = df[time_feature].str.split(':', expand=True)[0]
            df[time_feature+'_Minutes'] = df[time_feature].str.split(':', expand=True)[1]
            df[time_feature+'_Hour'] = pd.to_numeric(df[time_feature+'_Hour'], errors='coerce')
            df[time_feature+'_Minutes'] = pd.to_numeric(df[time_feature+'_Minutes'], errors='coerce')

        df.drop(time_features,axis=1,inplace=True)
        return df

    def handle_date_feature(self,df):
        # creating multiple columns of date
        date_features=['Order_Date']
        for date_feature in date_features:
            df[date_feature+'_Day']= df[date_feature].dt.day
            df[date_feature+'_Month'] = df[date_feature].dt.month
            df[date_feature+'_Year'] = df[date_feature].dt.year
            df[date_feature+'_DayOfWeek'] = df[date_feature].dt.dayofweek
            df[date_feature+'_DayOfYear'] = df[date_feature].dt.dayofyear

        df.drop(date_features,axis=1,inplace=True)
        return df


    def calculate_distance(self,lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)

        # Earth's radius in kilometers
        radius = 6371

        # Calculate the differences in latitude and longitude
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Apply the Haversine formula
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # Calculate the distance
        distance = radius * c

        return distance

    def get_distance(self,df):
        df['Distance'] = df.apply(lambda row: self.calculate_distance(row['Restaurant_latitude'], row['Restaurant_longitude'], row['Delivery_location_latitude'], row['Delivery_location_longitude']), axis=1)
        df.drop(['Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude'],inplace=True,axis=1)
        return df
    
    def get_transformation_obj(self):
        try:
            Weather_conditions=['Sunny','Cloudy','Fog','Windy','Stormy','Sandstorms']
            Road_traffic_density=['Low','Medium','High','Jam']
            Type_of_order=['Snack','Drinks','Meal','Buffet']
            Type_of_vehicle=['bicycle','motorcycle','electric_scooter','scooter']
            Festival=['No','Yes']
            City=['Semi-Urban','Urban','Metropolitian']

            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            ## Categorical pipeline
            categorical_pipeline= Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[Weather_conditions,
                                                                Road_traffic_density,
                                                                Type_of_order,
                                                                Type_of_vehicle,
                                                                Festival,
                                                                City])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,self.data_transformation_config.numeric_features),
                ('categorical_pipeline',categorical_pipeline,self.data_transformation_config.categorical_features),
            ])

            return preprocessor
        
        except Exception as e:
            logging.info("Error occured during preprocessing.")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        target_col = 'Time_taken (min)'
        logging.info('Data transformation initiated.')
        try:
            # reading train and test data
            train_df = pd.read_csv(train_path,parse_dates=["Order_Date"],dayfirst=True)
            test_df = pd.read_csv(test_path,parse_dates=["Order_Date"],dayfirst=True)


            logging.info('Read train and test data complete')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            drop_features=['ID','Delivery_person_ID']

            train_df.dropna(subset=['Time_Orderd'], inplace=True)
            test_df.dropna(subset=['Time_Orderd'], inplace=True)


            train_df.drop(drop_features,axis=1,inplace=True)
            test_df.drop(drop_features,axis=1,inplace=True)

            train_df = self.get_distance(train_df)
            test_df = self.get_distance(test_df)

            train_df=self.handle_time_feature(train_df)
            test_df=self.handle_time_feature(test_df)

            train_df=self.handle_date_feature(train_df)
            test_df=self.handle_date_feature(test_df)

            train_df['multiple_deliveries'] = train_df['multiple_deliveries'].astype('float')
            test_df['multiple_deliveries'] = test_df['multiple_deliveries'].astype('float')

            logging.info('Obtaining preprocessing objects')

            preprocessor = self.get_transformation_obj()


            input_train_df = train_df.drop(columns=target_col,axis=1)
            target_train_df = train_df[target_col]

            input_test_df = train_df.drop(columns=target_col,axis=1)
            target_test_df = train_df[target_col]

            logging.info('Applying preprocessing object on training and testing dataset.')

            input_train_array = preprocessor.fit_transform(input_train_df)
            input_test_array = preprocessor.transform(input_test_df)

            train_array = np.c_[input_train_array,np.array(target_train_df)]
            test_array = np.c_[input_test_array,np.array(target_test_df)]


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor
                )
            
            logging.info('Data transformation completed')
            logging.info(f'Train Dataframe Head : \n{train_array[:5,:]}')
            logging.info(f'Test Dataframe Head : \n{test_array[:5,:]}')
            return (train_array,test_array,self.data_transformation_config.preprocessor_obj_path)

        except Exception as e:
            logging.info('Error occured during transformation.')
            raise CustomException(e,sys)

