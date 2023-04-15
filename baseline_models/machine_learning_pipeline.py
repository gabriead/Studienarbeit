import pandas as pd
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

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
    """
    add feature lags to pd dataframe

    Arguments:
        data: pandas dataframe.
        n_in: Number of lag observations as input
        n_out: Number of observations as output
        dropnan: Boolean whether or not to drop rows with NaN values.

    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('ft%d_t-%d' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('ft%d_t' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('ft%d_t+%d' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def createDataset():
    path = "./complete_dataset"
    df = pd.read_csv(path)
    df = df.drop("Unnamed: 0", axis=1)
    df['date'] =  pd.to_datetime(df['date'])
    df['day'] = df.date.dt.dayofweek.astype(str).astype("category").astype(int)
    df["month"] = df.date.dt.month.astype(str).astype("category").astype(int)
    df["hour"] = df.date.dt.hour.astype(str).astype("category").astype(int)
    df["quarter"] = df.date.dt.quarter.astype(str).astype("category").astype(int)
    df["month"] = df.date.dt.month.astype(str).astype("category").astype(int)
    df["year"] = df.date.dt.year.astype(str).astype("category").astype(int)
    return df

def calculate_RMSE(actual, predicted, train_scalar, columnNames):
    actual = actual.reset_index(drop=True)
    predicted = predicted.reset_index(drop=True)
    actual = pd.DataFrame(train_scalar.inverse_transform(actual), columns=columnNames)
    predicted = pd.DataFrame(train_scalar.inverse_transform(predicted), columns=columnNames)
    results = pd.DataFrame(data={'actual': actual["readiness"], 'predicted': predicted["readiness"]})
    RMSE = mean_squared_error(results['actual'], results['predicted'], squared=False)

    return RMSE


def plotReadiness(df):
    """
    plots prediction vs actual values

    Arguments:
        df: pandas dataframe containing acutal and predicted columns

    Returns:
        plot
    """

    fig, ax = plt.subplots(figsize=(15, 5))
    df["actual"].plot(ax=ax, label='Actual', title='Predicted vs. Actual', linewidth=1)
    df["predicted"].plot(ax=ax, label='Predicted', linewidth=1)
    plt.xlabel('Timesteps in days', fontsize=18)
    plt.ylabel('Readiness value', fontsize=16)
    ax.axvline(0, color='black', ls='--')
    ax.legend(['Actual', 'Predicted'])
    plt.show()

def renameColumns(df, columnNames):

    df.columns.values[-len(columnNames):] = columnNames
    return df

def createLinePlots(df_test, ystar_col, i):
    RMSE = mean_squared_error(df_test["readiness_t+1"], df_test[ystar_col], squared=False)
    MSE = mean_squared_error(df_test["readiness_t+1"], df_test[ystar_col], squared=True)
    print(f'MSE Score on Test set: {MSE:0.4f}')
    print(f'RMSE Score on Test set: {RMSE:0.4f}')
    fig, ax = plt.subplots(figsize=(15, 5))
    df_test["readiness_t+1"].plot(ax=ax, label='Actual', title='RMSE: '+str(RMSE), linewidth=1)
    df_test[ystar_col].plot(ax=ax, label='Predicted', linewidth=1)
    plt.xlabel('Timesteps in days', fontsize=18)
    plt.ylabel('Readiness value', fontsize=16)
    ax.axvline(0, color='black', ls='--')
    ax.legend(['Actual', 'Predicted'])
    plt.savefig("BASELINE_results/LSTM_lineplot"+str(i))
    plt.close()


def recursive_multistep_prediction(model, X_test, timestepsOut, num_features, columnNames):
    """
    make recursive multistep prediction reusing output data to make prediction for next timestep

    Arguments:
        model: fitted model
        X_test: feature validation data
        timestepsOut: number of timesteps in the future to predict
        num_features: number of features in dataset
        columnNames: names of all features

    Returns:
        predictions n timesteps in the future
    """

    y_predict = []

    for day in range(len(X_test)):
        current_day = X_test.iloc[[day]]
        for steps in range(timestepsOut):
            pred = model.predict(current_day)
            current_day = current_day.T.shift(-num_features).T
            current_day.iloc[:, -num_features:] = pred
        y_predict.append(pred.flatten())

    y_predict = pd.DataFrame(data=np.array(y_predict), columns=columnNames)

    return y_predict


def runBenchmarksML(df, columnNames, n_in, n_out, multistep, players, configNr=1):

    results_xgb = []
    results_LIN = []
    results_TREE = []
    players = list(df['player_name_x'].unique())
    print("amount of players to train each config: ",len(players))
    models_ = []

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
        train = train_scalar.fit_transform(train)
        test = train_scalar.transform(test)

        if multistep:
            train_multistep = series_to_supervised(train.copy(), n_in)
            test_multistep = series_to_supervised(test.copy(), n_in)

            #CAUTION: check this again!!
            features = train_multistep.columns.tolist()[:-num_features]
            targets = test_multistep.columns.tolist()[-num_features:]

            features = train_direct.columns.tolist()[:-num_features * n_out]
            targets = test_direct.columns.tolist()[-num_features:]

            X_train = train_multistep[features]
            y_train = train_multistep[targets]
            X_test = test_multistep[features]
            y_test = test_multistep[targets]

            y_test = renameColumns(y_test, columnNames)

            modelXGB = fitXGBoost(X_train, y_train, X_test, y_test)
            # modelLin = fitLinearReg(X_train, y_train)
            # modelTree = fitTree(X_train, y_train)
            #
            y_predictXGB = recursive_multistep_prediction(modelXGB, X_test, n_out, num_features, columnNames)
            results_xgb.append(calculate_RMSE(y_test, y_predictXGB, train_scalar, columnNames))
            #
            # y_predictLIN = recursive_multistep_prediction(modelLin, X_test, n_out, num_features, columnNames)
            # results_LIN.append(calculate_RMSE(y_test, y_predictLIN, train_scalar, columnNames))
            #
            # y_predictTREE = recursive_multistep_prediction(modelTree, X_test, n_out, num_features, columnNames)
            # results_TREE.append(calculate_RMSE(y_test, y_predictTREE, train_scalar, columnNames))

            results = pd.DataFrame(data={'actual': y_test["readiness"], 'predicted': y_predictXGB["readiness"]})
            results["predicted"] = results["predicted"].shift(n_in)
            plotReadiness(results)

        models_ = []
        if not multistep:
            train_direct = series_to_supervised(train.copy(), n_in, n_out)
            test_direct = series_to_supervised(test.copy(), n_in, n_out)
            features = train_direct.columns.tolist()[:-num_features*n_out]
            targets = test_direct.columns.tolist()[-num_features:]


            X_train = train_direct[features]
            y_train = train_direct[targets]
            X_test = test_direct[features]
            y_test = test_direct[targets]

            reg = LazyRegressor(verbose=1, ignore_warnings=False, custom_metric = mean_absolute_error, regressors="all")
            model_dictionary = reg.provide_models(X_train, X_test, y_train, y_test)
            print(model_dictionary)
            models, predictions = reg.fit(X_train, X_test, y_train, y_test)
            print(models)
            models_.append(models)

            # print("Classification")
            #
            # reg = LazyClassifier(verbose=1, ignore_warnings=False, custom_metric = accuracy_score, classifiers="all")
            # model_dictionary = reg.provide_models(X_train, X_test, y_train, y_test)
            # print(model_dictionary)
            # models, predictions = reg.fit(X_train, X_test, y_train, y_test)
            # print(models)
            # models_.append(models)
            #

            import numpy as np
            y_test = renameColumns(y_test, columnNames)
            #
            modelXGB = fitXGBoost(X_train, y_train, X_test, y_test)
            #
            y_predictXGB = modelXGB.predict(X_test)
            y_predictXGB = pd.DataFrame(data = np.array(y_predictXGB), columns = columnNames)
            results_xgb.append(calculate_RMSE(y_test, y_predictXGB, train_scalar, columnNames))

        #     y_predictLIN = modelLin.predict(X_test)
        #     y_predictLIN = pd.DataFrame(data = np.array(y_predictLIN), columns = columnNames)
        #     results_LIN.append(calculate_RMSE(y_test, y_predictLIN, train_scalar, columnNames))
        #
        #     y_predictTREE = modelTree.predict(X_test)
        #     y_predictTREE = pd.DataFrame(data = np.array(y_predictTREE), columns = columnNames)
        #     results_TREE.append(calculate_RMSE(y_test, y_predictTREE, train_scalar, columnNames))
        #
        if (i == 0):
            createLinePlots(pd.DataFrame(train_scalar.inverse_transform(y_test), columns=columnNames), pd.DataFrame(train_scalar.inverse_transform(y_predictXGB), columns=columnNames), configNr, "BASELINE_results/XGB_lineplot")
            #createLinePlots(pd.DataFrame(train_scalar.inverse_transform(y_test), columns=columnNames), pd.DataFrame(train_scalar.inverse_transform(y_predictLIN), columns=columnNames), configNr, "BASELINE_results/LIN_lineplot")
            #createLinePlots(pd.DataFrame(train_scalar.inverse_transform(y_test), columns=columnNames), pd.DataFrame(train_scalar.inverse_transform(y_predictTREE), columns=columnNames), configNr, "BASELINE_results/TREE_lineplot")

    results_df = pd.DataFrame(
    {'xgb': results_xgb,
     'lin': results_LIN,
     'tree': results_TREE
    })

    #maxVal = 5
    #for c in results_df:
        #results_df[str(c)] = results_df[str(c)].where(results_df[str(c)] <= maxVal, maxVal)

    print(models_)
    return results_df


df = createDataset()
columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "stress", "month", "day"]


players = list(df['player_name_x'].unique())

runBenchmarksML(df,columnNames, n_in=10, n_out=1, multistep=False, players=players, configNr=1)