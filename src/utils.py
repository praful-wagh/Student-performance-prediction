import os, sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(xtr, ytr, xte, yte, models, params):
    try:
        report = {}
        for i in range(2):#len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(xtr, ytr)

            model.set_params(**gs.best_params_)
            model.fit(xtr, ytr)

            # yp_tr = model.predict(xtr)
            yp_te = model.predict(xte)

            # tr_model_score = r2_score(ytr, yp_tr)
            te_model_score = r2_score(yte, yp_te)

            report[list(models.keys())[i]] = te_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)