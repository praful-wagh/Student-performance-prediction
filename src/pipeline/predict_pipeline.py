import os.path
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import log

class PredictPipeline:
   def predict(self, features):
       try:
           project_path =(os.getcwd().split('\\Proj'))[0]+'\\Proj'
           model_path = os.path.join(project_path,'artifacts','model.pkl')
           preprocessor_path = os.path.join(project_path,'artifacts','preprocessor.pkl')
           log("Loading model and preprocessor..")
           model = load_object(model_path)
           preprocessor = load_object(preprocessor_path)
           log("Loading completed.")
           data_scaled = preprocessor.transform(features)
           pred = model.predict(data_scaled)
           log('Prediction: ' + pred)
           return pred

       except Exception as e:
           raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                gender: str,
                race_ethnicity: str,
                parental_level_of_education: str,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def getData_as_DataFrame(self):
        try:
            dict_ = {
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
            }
            return pd.DataFrame(dict_)

        except Exception as e:
            CustomException(e, sys)

if __name__=='__main__':
    df = CustomData("female","group B","bachelor's degree","standard","none",72,72).getData_as_DataFrame()
    print(PredictPipeline().predict(df))