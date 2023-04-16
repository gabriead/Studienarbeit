
import pandas as pd
from create_data import create_dataset
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn import metrics
import torch

np.random.seed(42)
tqdm.pandas()

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def mean_absolute_percentage_error_func(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def timeseries_evaluation_metrics_func(y_true, y_pred):

    # print('Evaluation metric results: ')

    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error_func(y_true, y_pred)
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error_func(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}', end='\n\n')

    return {"MSE":mse, "MAE":mae, "RMSE":rmse, "MAPE":mape}

y_test = pd.read_pickle("y_test.pkl")
y_val = pd.read_pickle("y_val.pkl")
y_pred = pd.read_pickle("y_pred.pkl")


timeseries_evaluation_metrics_func(y_test, y_pred)