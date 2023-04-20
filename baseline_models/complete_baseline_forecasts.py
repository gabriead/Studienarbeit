import numpy as np
import pandas as pd

from sktime.forecasting.naive import NaiveForecaster
from create_data import create_dataset
from tqdm.autonotebook import tqdm

np.random.seed(42)
tqdm.pandas()
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import xgboost as xgb
from lazypredict.Supervised import LazyRegressor
from sklearn import metrics




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
        # if i == 0:
        #    names += [('ft%d_t' % (j + 1)) for j in range(n_vars)]
        # else:
        names += [('ft%d_t+%d' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def plot_forecast_xgb(y_true, y_pred, player):
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(12, 6))
    plt.title(f"MAE: {mae:.2f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}", size=18)
    y_true.plot(label="true", color="g")
    y_pred.plot(label="test", color="r")

    plt.savefig(f'''plot_{player}.png''')
    return mse, mae, rmse

def plot_forecast_statistical_tests(y_true, y_pred, player, df):
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(12, 6))
    plt.title(f"MAE: {mae:.2f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}", size=18)
    y_true.plot(label="true", color="g")
    y_pred.plot(label="test", color="b")
    #plt.plot(y_pred.index[9705:9709], y_pred.values[9705:9709])

    plt.savefig(f'''plot_{player}.png''')
    return mse, mae, rmse



def pipeline():
    n_in = 30
    n_out = 1
    metrics_df = pd.DataFrame(
        columns=["player_name", "mae", "rmse", "mse", "n_in", "n_out", "features_in", "features_out"])

    player_names = []
    maes = []
    rmses = []
    mses = []
    n_ins = []
    n_outs = []
    features_in = []
    features_out = []

    # Currently not using time features !!!
    df = create_dataset()
    # CAUTION:dates are currently not being used for debugging reasons
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness",
                   "stress"]

    features = ["daily_load", "fatigue", "mood", "sleep_duration", "sleep_quality", "soreness",
                   "stress"]

    target = ["readiness"]

    # filter teams
    df = df[df["player_name_x"].str.startswith("TeamA")]
    players = list(df['player_name_x'].unique())

    print("amount of players to train each config: ", len(players))
    import random

    for i in range(0, len(players)):
        current_player = players[i]
        test_players = players.copy()

        test_players.remove(current_player)

        val_player = random.choice(test_players)
        test_players.remove(val_player)

        df_train = df[df['player_name_x'].isin(players)]
        df_test = df[df['player_name_x'].isin(test_players)]
        df_val = df[df['player_name_x'].isin([val_player])]

        train = df_train[columnNames]
        test = df_test[columnNames]
        val = df_val[columnNames]

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        val = val.reset_index(drop=True)
        from sklearn.preprocessing import MinMaxScaler
        num_features = len(train.columns.tolist())

        train_scalar = MinMaxScaler()
        train_transformed = train_scalar.fit_transform(train)
        test_transformed = train_scalar.transform(test)
        val_transformed = train_scalar.transform(val)

        train_direct = series_to_supervised(train_transformed.copy(), n_in, n_out)
        test_direct = series_to_supervised(test_transformed.copy(), n_in, n_out)
        val_direct = series_to_supervised(val_transformed.copy(), n_in, n_out)

        features = train_direct.columns.tolist()[:(n_in * num_features)]
        targets = test_direct.columns.tolist()[(n_in * num_features):]

        X_train = train_direct[features]
        # X_train.columns = features_

        y_train = train_direct[targets]
        # y_train.columns = targets_

        X_test = test_direct[features]
        # X_test.columns = features_

        y_test = test_direct[targets]
        # y_test.columns = targets_

        X_val = val_direct[features]
        # X_val.columns = features_

        y_val = val_direct[targets]
        # y_val.columns = targets_
        #xgboost_mae, xgboost_mape, xgboost_rmse = xgboost_model(X_test, X_train, y_test, y_train,players[i] ,metrics_df)
        #naive_forecast_drift_mae, naive_forecast_drift_mape, naive_forecast_drift_rmse = naive_forecast_mean(y_test, y_train, n_out, players[i] ,metrics_df)
        #naive_forecast_drift_mae, naive_forecast_drift_mape, naive_forecast_drift_rmse = naive_forecast_drift(y_test, y_train, players[i], metrics_df)
        #naive_forecast_drift_mae, naive_forecast_drift_mape, naive_forecast_drift_rmse = auto_arima_forecast(y_test, y_train, n_out, players[i] ,metrics_df)
        #lazypredict_regressors(X_test, X_train, y_test, y_train, players[i])
        treeregressor(X_test, X_train, y_test, y_train, players[i])

        #print("XGBOOST:", naive_forecast_drift_mae, naive_forecast_drift_mape, naive_forecast_drift_rmse)


from pandas.plotting import table
def lazypredict_regressors(X_test, X_train, y_test, y_train, player):
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=mean_absolute_error, regressors="all")
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    models.to_pickle(player+".pkl")


def xgboost_model(X_test, X_train, y_test_selected, y_train_selected, player, df):
    modelXGB = fitXGBoost(X_train, y_train_selected, X_test, y_test_selected)
    y_pred = modelXGB.predict(X_test)
    #    y_predictXGB = pd.DataFrame(data=np.array(y_pred), columns=columnNames)

    y_test_selected = y_test_selected['ft4_t+0']
    df_pred = pd.DataFrame(y_pred, columns=['0', '1', '2', '3', '4', '5', '6', '7'])
    y_pred = df_pred['3']

    mse, mae, rmse = plot_forecast_xgb(y_test_selected, y_pred, player)
    # results_xgb.append(calculate_RMSE(y_test, y_predictXGB, train_scalar, columnNames))
    return mse, mae, rmse

from sklearn.ensemble import ExtraTreesRegressor
def treeregressor(X_test, X_train, y_test, y_train, player):
    extra_tree_model = ExtraTreesRegressor(n_estimators=100,
                                           criterion='mse', max_features="auto")
    # Training the model
    tree_model = extra_tree_model.fit(X_train, y_train)


    feature_importance = extra_tree_model.feature_importances_
    # Plotting a Bar Graph to compare the models
    plt.bar(X_train.columns, feature_importance)
    plt.xlabel('Feature Labels')
    plt.ylabel('Feature Importances')
    plt.title('Comparison of different Feature Importances')
    plt.show()
    y_pred = tree_model.predict(X_test)

    y_test_selected = y_test['ft4_t+0']
    df_pred = pd.DataFrame(y_pred, columns=['0', '1', '2', '3', '4', '5', '6', '7'])
    y_pred = df_pred['3']



    mse, mae, rmse = plot_forecast_xgb(y_test_selected, y_pred, player)
    # results_xgb.append(calculate_RMSE(y_test, y_predictXGB, train_scalar, columnNames))
    return mse, mae, rmse


def naive_forecast_mean(y_test, y_train, n_out, player, df):
    forecaster = NaiveForecaster(strategy="mean", window_length=n_out)
    fh = np.arange(y_test.shape[0]) + 1
    forecaster.fit(y=y_train, fh=fh)
    y_pred = forecaster.predict()

    y_pred.index = y_test.index
    y_test = y_test['ft4_t+0']
    y_pred = y_pred['ft4_t+0']

    mse, mae, rmse = plot_forecast_statistical_tests(y_test, y_pred, player, df)

    return mse, mae, rmse

def naive_forecast_drift(y_test, y_train, player, df):
    forecaster = NaiveForecaster(strategy="drift")
    fh = np.arange(y_test.shape[0])+1
    forecaster.fit(y=y_train, fh=fh)
    y_pred = forecaster.predict(fh=fh)

    y_pred.index = y_test.index
    y_test = y_test['ft4_t+0']
    y_pred = y_pred['ft4_t+0']

    mse, mae, rmse = plot_forecast_statistical_tests(y_test, y_pred, player, df)

    return mse, mae, rmse

pipeline()
