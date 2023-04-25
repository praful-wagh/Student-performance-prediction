import os
import sys
import numpy as np
import pandas as pd
from src.logger import log
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    project_path =(os.getcwd().split('\\Proj'))[0]+'\\Proj'
    preprocessor_obj_file_path: str=os.path.join(project_path,'artifacts','preprocessor.pkl')

def get_data_transformer_object():
    """ This function is responsible for data transformation """
    try:
        num_columns = ["writing_score", "reading_score"]
        cat_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

        num_pipeline = Pipeline(
            steps=[
                ('impute', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        )
        cat_pipeline = Pipeline(
            steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ]
        )

        log(f"Categorical columns: {cat_columns}")
        log(f"Numerical columns: {num_columns}")

        preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, num_columns),
            ('cat_pipeline', cat_pipeline, cat_columns)
        ])

        return preprocessor

    except Exception as e:
        raise CustomException(e, sys)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
           train_df = pd.read_csv(train_path)
           test_df = pd.read_csv(test_path)

           log("Read train and test data completed")
           log("Obtaining preprocessing object")

           preprocessing_obj = get_data_transformer_object()

           y_column = 'math_score'
           # num_cols = ["writing_score", "reading_score"]
           
           xtr = train_df.drop(y_column,axis=1)
           ytr = train_df[y_column]

           xte = test_df.drop(y_column,axis=1)
           yte = test_df[y_column]

           log("Applying preprocessing object on training dataframe and testing dataframe.")

           xtr_arr = preprocessing_obj.fit_transform(xtr)
           xte_arr = preprocessing_obj.fit_transform(xte)

           train_arr = np.c_[xtr_arr, np.array(ytr)]
           test_arr = np.c_[xte_arr, np.array(yte)]

           save_object(self.data_transformation_config.preprocessor_obj_file_path,preprocessing_obj)
           log("Saved preprocessing object.")

           return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
