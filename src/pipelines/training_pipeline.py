from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import Modeltrainer

if __name__ == '__main__':
    obj=DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer = Modeltrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)