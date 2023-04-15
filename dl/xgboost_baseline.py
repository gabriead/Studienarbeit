import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn import metrics

from pmdarima.model_selection import train_test_split as time_train_test_split

from xgboost import XGBRegressor
from xgboost import plot_importance
import joblib
from itertools import cycle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils import plotting_utils

import warnings
warnings.filterwarnings("ignore")


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


def create_features(df, target_variable):

    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    if target_variable:
        y = df[target_variable]
        return X, y
    return X



with open("test.pkl", "rb") as file:
    import pickle
    test_df = pickle.load(file)

with open("train.pkl", "rb") as file:
    import pickle
    train_df = pickle.load(file)

train_pjme_copy = train_df.copy()
test_pjme_copy = test_df.copy()

#trainX, trainY = create_features(train_pjme_copy, target_variable='readiness')
#testX, testY = create_features(test_pjme_copy, target_variable='readiness')

xgb = XGBRegressor(objective= 'reg:linear', n_estimators=1000)
xgb

trainX, trainY = train_df.iloc[:, :-3], train_df["target"]
testX, testY = test_df.iloc[:, :-3], test_df["target"]

xgb.fit(trainX, trainY,
        eval_set=[(trainX, trainY), (testX, testY)],
        early_stopping_rounds=50,
        verbose=False)


feature_importance = plot_importance(xgb, height=0.9)
feature_importance

predicted_results = xgb.predict(testX)
predicted_results

evaluation = timeseries_evaluation_metrics_func(testY, predicted_results)
dictlist = []

labels = []

for key, value in evaluation.items():
    labels.append(str(key)+":"+str(value))

plt.figure(figsize=(13,8))
plt.plot(list(testY))
plt.plot(list(predicted_results))
plt.title("Actual Readiness vs Predicted Readiness")
plt.ylabel("Readiness")
#plt.legend(dictlist, numpoints=4)
handles = [
    Line2D([0], [0], color='red',lw=3),
    Line2D([0], [0], color='red',lw=3),
    Line2D([0], [0], color='red',lw=3)
]

plt.legend(handles, labels, loc='upper right', fontsize='x-large',fancybox=True, framealpha=0.7)
#plt.show()
#print()
plt.savefig("univariate_window_30_1.png")
