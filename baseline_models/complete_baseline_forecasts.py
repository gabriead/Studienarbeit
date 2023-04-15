
import numpy as np
import pandas as pd
import plotly.io as pio
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.base import ForecastingHorizon
pio.templates.default = "plotly_white"
from sklearn.preprocessing import StandardScaler
from create_data import create_dataset
from tqdm.autonotebook import tqdm
np.random.seed(42)
tqdm.pandas()
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb
from lazypredict.Supervised import LazyRegressor


def fitXGBoost(X_train, y_train, X_test, y_test):
    """
    fits data to XGBOOST model

    Arguments:
        X_train: feature training data
        y_train: target training data
        X_test: feature validation data
        y_test: target validation data

    Returns:
        fitted XGBOOST model
    """
    reg = xgb.XGBRegressor(booster='gbtree',
                           n_estimators=200,
                           objective='reg:squarederror',
                           learning_rate=0.07,
                           colsample_bytree=0.9704161741146843,
                           gamma=3.472716930386355,
                           max_depth=9,
                           min_child_weight=9,
                           reg_alpha=44,
                           reg_lambda=0.454959775303947,
                           seed=0)

    reg.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100, early_stopping_rounds=20)

    return reg

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('ft%d_t-%d' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        #if i == 0:
        #    names += [('ft%d_t' % (j + 1)) for j in range(n_vars)]
        #else:
        names += [('ft%d_t+%d' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def plot_forecast_xgb(train_df, test_df, forecast_df,):

    mae = mean_absolute_error(test_df, forecast_df)
    mape = mean_absolute_percentage_error(test_df, forecast_df)
    rmse = mean_squared_error(test_df, forecast_df)

    plt.figure(figsize=(12, 6))
    plt.title(f"MAE: {mae:.2f}, MAPE: {mape:.3f}", size=18)
    test_df.plot(label="test", color="g")
    forecast_df = pd.DataFrame(forecast_df, columns=['forecast'])
    forecast_df.index = test_df.index
    forecast_df.plot(label="forecast", color="r")

    plt.plot(test_df)
    plt.plot(forecast_df)

    plt.show()

    return mae, mape, rmse

def plot_forecast(test_df, forecast_df, forecast_int=None):

    mae = mean_absolute_error(test_df, forecast_df)
    mape = mean_absolute_percentage_error(test_df, forecast_df)
    rmse = mean_squared_error(test_df, forecast_df)

    plt.figure(figsize=(12, 6))
    plt.title(f"MAE: {mae:.2f}, MAPE: {mape:.3f}", size=18)
    forecast_df.index = test_df.index

    plt.plot(test_df)
    plt.plot(forecast_df)

    my_array= forecast_int["Coverage"].values
    df = pd.DataFrame(my_array, columns=['lower', 'upper'])

    if df is not None:
        plt.fill_between(
            test_df.index,
            df["lower"],
            df["upper"],
            alpha=0.2,
            color="dimgray",
        )
    plt.legend(prop={"size": 16})
    plt.show()

    return mae, mape, rmse

def runBenchmarksML(df, columnNames, n_in, n_out, multistep, players, configNr=1):

    results_xgb = []
    results_LIN = []
    results_TREE = []
    players = list(df['player_name_x'].unique())
    print("amount of players to train each config: ",len(players))
    models_ = []


    #activate all players
    for i in range(1):

        #print(i+1,"/",len(players))
        all_but_one = players[:i] + players[i+1:]
        df_train = df[df['player_name_x'].isin(all_but_one)]
        df_test = df.loc[df['player_name_x'] == players[i]]

        train = df_train[columnNames]
        test = df_test[columnNames]

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        num_features = len(train.columns.tolist())

        train_scalar = StandardScaler()
        train_transformed = train_scalar.fit_transform(train)
        test_transformed = train_scalar.transform(test)

        models_ = []
        if not multistep:
            train_direct = series_to_supervised(train_transformed.copy(), n_in, n_out)
            test_direct = series_to_supervised(test_transformed.copy(), n_in, n_out)
            features = train_direct.columns.tolist()[:num_features*n_out]
            targets = test_direct.columns.tolist()[num_features:]

            n_vars = 1 if type(train_transformed) is list else train_transformed.shape[1]
            features_ = list()
            targets_ = list()
            for i in range(0, n_in):
                features_ += [(columnNames[j]+'_t-%d' % i) for j in range(n_vars)]

            for i in range(0, n_out):
                targets_ += [(columnNames[j] + '_t+%d' % i) for j in range(n_vars)]


            X_train = train_direct[features]
            X_train = X_train.set_axis(features_, axis=1, inplace=False)

            y_train = train_direct[targets]
            y_train = y_train.set_axis(targets_, axis=1, inplace=False)

            X_test = test_direct[features]
            X_test = X_test.set_axis(features_, axis=1, inplace=False)

            y_test = test_direct[targets]
            y_test = y_test.set_axis(targets_, axis=1, inplace=False)

            # select targets to predict
            target_columns = y_train.columns
            target = "readiness"

            selected_columns = [target_columns[i] for i in range(len(target_columns)) if target in target_columns[i]]

            y_train_selected = y_train[selected_columns]
            y_test_selected = y_test[selected_columns]

            #auto_arima_mae, auto_arima_mape, auto_arima_rmse = auto_arima_forecast(y_test_selected, y_train_selected)
            naive_forecast_mean_mae, naive_forecast_mean_mape, naive_forecast_mean_rmse = naive_forecast_mean(y_test_selected, y_train_selected, n_out)

            #y_test = renameColumns(y_test, columnNames)
            xgboost_mae, xgboost_mape, xgboost_rmse = xgboost_model(X_test, X_train, y_test_selected, y_train_selected)

            print("XGBOOST:", xgboost_mae, xgboost_mape, xgboost_rmse)

            #lazypredict_regressors(X_test, X_train, y_test, y_train)

            #write table!!
            naive_forecast_mean_list = [naive_forecast_mean_mae, naive_forecast_mean_mape, naive_forecast_mean_rmse, n_in, n_out]
            xgboost_results = [xgboost_mae, xgboost_mape, xgboost_rmse, n_in, n_out]


            results_df = pd.DataFrame()

            if n_out > 1:
                naive_forecast_drift_mae, naive_forecast_drift_mape, naive_forecast_drift_rmse = naive_forecast_drift(
                    y_test_selected, y_train_selected, n_out)
                naive_forecast_drift_list = [naive_forecast_drift_mae, naive_forecast_drift_mape, naive_forecast_drift_rmse, n_in,n_out]

                d = {'naive_forecast_mean': naive_forecast_mean_list, 'naive_forecast_drift': naive_forecast_drift_list, 'xgboost': xgboost_results}
                results_df = pd.DataFrame(data=d, index=["mae", "mape", "rmse", "n_in", "n_out"])
            else:
                d = {'naive_forecast_mean': naive_forecast_mean_list,  'xgboost': xgboost_results}
                results_df = pd.DataFrame(data=d, index=["mae", "mape", "rmse", "n_in", "n_out"])
                print()


        #if (i == 0):
            #createLinePlots(pd.DataFrame(train_scalar.inverse_transform(y_test), columns=columnNames), pd.DataFrame(train_scalar.inverse_transform(y_predictXGB), columns=columnNames), configNr, "BASELINE_results/XGB_lineplot")
            #createLinePlots(pd.DataFrame(train_scalar.inverse_transform(y_test), columns=columnNames), pd.DataFrame(train_scalar.inverse_transform(y_predictLIN), columns=columnNames), configNr, "BASELINE_results/LIN_lineplot")
            #createLinePlots(pd.DataFrame(train_scalar.inverse_transform(y_test), columns=columnNames), pd.DataFrame(train_scalar.inverse_transform(y_predictTREE), columns=columnNames), configNr, "BASELINE_results/TREE_lineplot")

    # results_df = pd.DataFrame(
    # {'xgb': results_xgb,
    #  'lin': results_LIN,
    #  'tree': results_TREE
    # })

    #maxVal = 5
    #for c in results_df:
        #results_df[str(c)] = results_df[str(c)].where(results_df[str(c)] <= maxVal, maxVal)

    #print(models_)
    return results_df


def lazypredict_regressors(X_test, X_train, y_test, y_train):
    reg = LazyRegressor(verbose=1, ignore_warnings=False, custom_metric=mean_absolute_error, regressors="all")
    model_dictionary = reg.provide_models(X_train, X_test, y_train, y_test)
    print(model_dictionary)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    print(models)


def xgboost_model(X_test, X_train, y_test_selected, y_train_selected):
    modelXGB = fitXGBoost(X_train, y_train_selected, X_test, y_test_selected)
    y_pred = modelXGB.predict(X_test)
#    y_predictXGB = pd.DataFrame(data=np.array(y_pred), columns=columnNames)
    mae, mape, rmse = plot_forecast_xgb(y_train_selected, y_test_selected, y_pred)
    #results_xgb.append(calculate_RMSE(y_test, y_predictXGB, train_scalar, columnNames))
    return mae, mape, rmse

def auto_arima_forecast(y_test_selected, y_train_selected):
    forecaster = AutoARIMA()
    fh = np.arange(y_test_selected.shape[0]) + 1
    forecaster.fit(y=y_train_selected, fh=fh)
    y_pred = forecaster.predict()
    coverage = 0.9
    forecast_int = forecaster.predict_interval(fh=fh, coverage=coverage)
    mae, mape, rmse = plot_forecast(y_test_selected, y_pred, forecast_int)

    return mae, mape, rmse

from sktime.utils.plotting import plot_series
def naive_forecast_mean(y_test_selected, y_train_selected, n_out):
    forecaster = NaiveForecaster(strategy="mean", window_length=n_out)
    fh = np.arange(y_test_selected.shape[0]) + 1
    forecaster.fit(y=y_train_selected, fh=fh)
    y_pred = forecaster.predict()
    coverage = 0.9
    forecast_int = forecaster.predict_interval(fh=fh, coverage=coverage)
    #plot_series(y_train_selected, y_test_selected,y_pred, labels=["y-train", "y_test", "y_pred"])
    mae, mape, rmse = plot_forecast(y_test_selected, y_pred, forecast_int)

    return mae, mape, rmse

def naive_forecast_drift(y_test_selected, y_train_selected, n_out):
    forecaster = NaiveForecaster(strategy="drift", window_length=n_out)
    fh = np.arange(y_test_selected.shape[0]) + 1
    forecaster.fit(y=y_train_selected, fh=fh)
    y_pred = forecaster.predict()
    coverage = 0.9
    forecast_int = forecaster.predict_interval(fh=fh, coverage=coverage)
    #plot_series(y_train_selected, y_test_selected,y_pred, labels=["y-train", "y_test", "y_pred"])
    mae, mape, rmse = plot_forecast(y_test_selected, y_pred, forecast_int)

    return mae, mape, rmse



#Check for the other features!!
df = create_dataset()
columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "stress", "month", "day"]
players = list(df['player_name_x'].unique())
runBenchmarksML(df,columnNames, n_in=1, n_out=1, multistep=False, players=players, configNr=1)
