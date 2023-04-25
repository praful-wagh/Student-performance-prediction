import os, sys
import pprint
from dataclasses import dataclass
from src.logger import log
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

@dataclass
class ModelTrainerConfig:
    project_path =(os.getcwd().split('\\Proj'))[0]+'\\Proj'
    trained_model_file_path = os.path.join(project_path,'artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            log("Split training and test input data")
            xtr, ytr, xte, yte = (train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1])
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params = {
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[.6,.7,.75,.8,.85,.9],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report: dict=evaluate_models(xtr, ytr, xte, yte, models, params)
            pprint.pprint(model_report)
            model = sorted(model_report.items(),reverse=True, key=lambda x:x[1])[0]
            best_model_name = model[0]
            best_model_score = model[1]
            print('High scoring model:',best_model_name, '\nModel score:',best_model_score)
            # best_model_score = max(model_report.values())
            # best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < .6:
                log('No best model found!')
                sys.exit(1)

            log('Best found model on both training and testing dataset : '+ str(best_model_name) + ' with score: ' + str(best_model_score))
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            pred = best_model.predict(xte)
            return best_model_name,r2_score(yte, pred)

        except Exception as e:
            raise CustomException(e, sys)