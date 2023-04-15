import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from PIL._imaging import display

pio.templates.default = "plotly_white"

import warnings
from pathlib import Path
from darts.utils.utils import ModelMode, SeasonalityMode
from sklearn.preprocessing import StandardScaler

from create_data import create_dataset


import humanize

# If importing darts is throwing an error, import torch beforehand and then import darts
# import torch
from darts import TimeSeries
from darts.metrics import mae, mase, mse, ope
from darts.models import (
    ARIMA,
    FFT,
    AutoARIMA,
    ExponentialSmoothing,
    NaiveDrift,
    NaiveMean,
    NaiveSeasonal,
    Theta,
Prophet
)
from src.utils import plotting_utils
from src.utils.general import LogTime
from src.utils.ts_utils import forecast_bias
from tqdm.autonotebook import tqdm
from darts.models.filtering.moving_average import MovingAverage

np.random.seed(42)
tqdm.pandas()


os.makedirs("imgs/chapter_4", exist_ok=True)
#output = Path("data/data/london_smart_meters/output")
import matplotlib.pyplot as plt


def format_plot(
    fig, legends=None, xlabel="Time", ylabel="Value", title="", font_size=15
):
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_layout(
        autosize=False,
        width=900,
        height=500,
        title_text=title,
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
        titlefont={"size": 20},
        legend_title=None,
        legend=dict(
            font=dict(size=font_size),
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=1,
        ),
        yaxis=dict(
            title_text=ylabel,
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
        xaxis=dict(
            title_text=xlabel,
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
    )
    return fig

from src.dl.my_train_test_slit import create_train_test_data_baselines


train_df, test_df, val_df = create_train_test_data_baselines()

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

    agg["date"] = agg.index
    return agg

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

            forecaster = ARIMA()
            forecaster.fit(y=y_train, fh=[i for i in range(0, n_out)])
            y_pred = forecaster.predict()
            print()



        #if (i == 0):
            #createLinePlots(pd.DataFrame(train_scalar.inverse_transform(y_test), columns=columnNames), pd.DataFrame(train_scalar.inverse_transform(y_predictXGB), columns=columnNames), configNr, "BASELINE_results/XGB_lineplot")
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



df = create_dataset(10,10)
columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "stress", "month", "day"]
players = list(df['player_name_x'].unique())
runBenchmarksML(df,columnNames, n_in=10, n_out=30, multistep=False, players=players, configNr=1)


train_series = TimeSeries.from_dataframe(
    train_df,
    time_col="date",
    value_cols=["readiness"],
    fill_missing_dates=True,
    freq=None,
)

test_series = TimeSeries.from_dataframe(
    test_df,
    time_col="date",
    value_cols=["readiness"],
    fill_missing_dates=True,
    freq=None,
)

# val_series = TimeSeries.from_dataframe(
#     val_df,
#     time_col="date",
#     value_cols=["readiness"],
#     fill_missing_dates=True,
#     freq=None,
# )


MSEAS = 12  # seasonality default
ALPHA = 0.05

plt.figure(100, figsize=(12, 5))
train_series.plot(label="readiness_training_data")
test_series.plot(label="readiness_test_data")
#val_series.plot(label="readiness_validation_data")

#plt.show()

pred_df = pd.concat([train_df, test_df])
metric_record = []
##################################################
plt.figure(100, figsize=(12, 5))


def eval_model(model, train_series, test_series, name=None):
    if name is None:
        name = type(model).__name__
    model.fit(train_series)
    y_pred = model.predict(len(test_series))
    return y_pred, {
        "Algorithm": name,
        "MAE": mae(actual_series=test_series, pred_series=y_pred),
        "MSE": mse(actual_series=test_series, pred_series=y_pred),
        "MASE": mase(
            actual_series=test_series, pred_series=y_pred, insample=train_series
        ),
        "Forecast Bias": forecast_bias(
            actual_series=test_series, pred_series=y_pred
        ),
    }


def format_y_pred(y_pred, name):
    y_pred = y_pred.data_array().to_series()
    y_pred.index = y_pred.index.get_level_values(0)
    y_pred.name = name
    return y_pred


# In[10]:


from itertools import cycle


def plot_forecast(pred_df, forecast_columns, forecast_display_names=None):
    if forecast_display_names is None:
        forecast_display_names = forecast_columns
    else:
        assert len(forecast_columns) == len(forecast_display_names)
    mask = ~pred_df[forecast_columns[0]].isnull()
    # colors = ["rgba("+",".join([str(c) for c in plotting_utils.hex_to_rgb(c)])+",<alpha>)" for c in px.colors.qualitative.Plotly]
    colors = [
        c.replace("rgb", "rgba").replace(")", ", <alpha>)")
        for c in px.colors.qualitative.Dark2
    ]
    # colors = [c.replace("rgb", "rgba").replace(")", ", <alpha>)") for c in px.colors.qualitative.Safe]
    act_color = colors[0]
    colors = cycle(colors[1:])
    dash_types = cycle(["dash", "dot", "dashdot"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pred_df[mask].index,
            y=pred_df[mask].readiness,
            mode="lines",
            line=dict(color=act_color.replace("<alpha>", "0.3")),
            name="Actual Readiness",
        )
    )
    for col, display_col in zip(forecast_columns, forecast_display_names):
        fig.add_trace(
            go.Scatter(
                x=pred_df[mask].index,
                y=pred_df.loc[mask, col],
                mode="lines",
                line=dict(
                    dash=next(dash_types),
                    color=next(colors).replace("<alpha>", "1"),
                ),
                name=display_col,
            )
        )
    return fig


# ## Naive Forecast
from src.forecasting.baselines import NaiveMovingAverage


# name = "Moving Average Forecast"
# moving_average = MovingAverage(window=30)
# with LogTime() as timer:
#      y_pred, metrics = eval_model(moving_average, test_series, name=name)
# metrics["Time Elapsed"] = timer.elapsed
# metric_record.append(metrics)
# y_pred = format_y_pred(y_pred, "moving_average_predictions")
# pred_df = pred_df.join(y_pred)
#
# fig = plot_forecast(
#      pred_df,
#      forecast_columns=["moving_average_predictions"],
#      forecast_display_names=["Moving Average Predictions"],
# )
# fig = format_plot(
#      fig,
#      title=f"Moving Average: MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4f} | MASE: {metrics['MASE']:.4f} | Bias: {metrics['Forecast Bias']:.4f}",
# )
# fig.update_xaxes(type="date", range=["2021-12-01", "2021-12-31"])
# fig.write_image("imgs/chapter_4/ma.png")
# fig.show()
#
#
# fig = plot_forecast(pred_df,
#                      forecast_columns=[
#                          "naive_predictions",
#                          "moving_average_predictions"
#                      ],
#                      forecast_display_names=[
#                          "Naive",
#                          'Moving Average'
#                      ])
# fig = format_plot(fig, title=f"Naive and Moving Average Forecasts")
# fig.update_xaxes(type="date", range=["2021-12-01", "2021-12-31"])
# fig = plotting_utils.make_lines_greyscale(fig)
# fig.write_image("imgs/chapter_4/naive_ma.png")
# fig.show()
#

def naive_seasonal(pred_df, forecast_window, img_name):

    name = "Seasonal Naive Forecast"
    naive_seasonal = NaiveSeasonal(K=forecast_window)
    with LogTime() as timer:
        y_pred, metrics = eval_model(naive_seasonal,train_series, test_series, name=name)
    metrics["Time Elapsed"] = timer.elapsed
    metric_record.append(metrics)
    y_pred = format_y_pred(y_pred, "naive_predictions")
    pred_df = pred_df.join(y_pred)

    fig = plot_forecast(
        pred_df,
        forecast_columns=["naive_predictions"],
        forecast_display_names=["Seasonal Naive Predictions"],
    )
    fig = format_plot(
        fig,
        title=f"Seasonal Naive: MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4f} | MASE: {metrics['MASE']:.4f} | Bias: {metrics['Forecast Bias']:.4f}",
    )
    fig.update_xaxes(type="date", range=["2021-12-01", "2021-12-31"])

    fig.write_image("imgs/chapter_4/" + img_name)
    fig.show()


def auto_arima(pred_df, test_series, img_name):
    name = "AUTO_ARIMA"
    # Not using AutoARIMA because it just takes too much time for long time series
    auto_arima_model = AutoARIMA(max_p=5, max_q=3, m=30, seasonal=False)

    with LogTime() as timer:
        y_pred, metrics = eval_model(
            auto_arima_model, train_series, test_series, name=name
        )
    metrics["Time Elapsed"] = timer.elapsed
    metric_record.append(metrics)
    y_pred = format_y_pred(y_pred, "auto_arima_predictions")
    pred_df = pred_df.join(y_pred)


    fig = plot_forecast(
        pred_df,
        forecast_columns=["auto_arima_predictions"],
        forecast_display_names=["Auto ARIMA Predictions"],
    )
    fig = format_plot(
        fig,
        title=f"ARIMA: MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4f} | MASE: {metrics['MASE']:.4f} | Bias: {metrics['Forecast Bias']:.4f}",
    )
    fig.update_xaxes(type="date", range=["2021-11-01", "2021-11-31"])

    fig.write_image("imgs/chapter_4/"+img_name)
    fig.show()

def arima(pred_df):
    name = "ARIMA"
    # Not using AutoARIMA because it just takes too much time for long time series
    # arima_model = AutoARIMA(max_p=5, max_q=3, m=48, seasonal=False)
    arima_model = ARIMA(p=2, d=1, q=1, seasonal_order=(1, 1, 1, 48))
    # Taking only latest 8000 points for training (Time constraints)
    # Reduce 8000 if it is taking too much time or consuming all the memory
    with LogTime() as timer:
        y_pred, metrics = eval_model(
            arima_model, train_series, test_series, name=name
        )
    metrics["Time Elapsed"] = timer.elapsed
    metric_record.append(metrics)
    y_pred = format_y_pred(y_pred, "arima_predictions")
    pred_df = pred_df.join(y_pred)


    fig = plot_forecast(
        pred_df,
        forecast_columns=["arima_predictions"],
        forecast_display_names=["ARIMA Predictions"],
    )
    fig = format_plot(
        fig,
        title=f"ARIMA: MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4f} | MASE: {metrics['MASE']:.4f} | Bias: {metrics['Forecast Bias']:.4f}",
    )
    fig.update_xaxes(type="date", range=["2021-11-01", "2021-11-31"])

    fig.write_image("imgs/chapter_4/arima.png")
    fig.show()


def theta(pred_df):
    name = "Theta"
    theta_model = Theta(
        theta=3, seasonality_period=7, season_mode=SeasonalityMode.ADDITIVE
    )
    with LogTime() as timer:
        y_pred, metrics = eval_model(theta_model,train_series, test_series, name=name)
    metrics["Time Elapsed"] = timer.elapsed
    metric_record.append(metrics)
    y_pred = format_y_pred(y_pred, "theta_predictions")
    pred_df = pred_df.join(y_pred)


    fig = plot_forecast(
        pred_df,
        forecast_columns=["theta_predictions"],
        forecast_display_names=["Theta Predictions"],
    )
    fig = format_plot(
        fig,
        title=f"Theta: MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4f} | MASE: {metrics['MASE']:.4f} | Bias: {metrics['Forecast Bias']:.4f}",
    )
    fig.update_xaxes(type="date", range=["2021-11-01", "2021-11-31"])
    fig.write_image("imgs/chapter_4/theta.png")
    fig.show()


def fft(pred_df):

    name = "FFT"
    fft_model = FFT(nr_freqs_to_keep=35, trend="poly", trend_poly_degree=2)
    with LogTime() as timer:
        y_pred, metrics = eval_model(fft_model, train_series, test_series,name=name)
    metrics["Time Elapsed"] = timer.elapsed
    metric_record.append(metrics)
    y_pred = format_y_pred(y_pred, "fft_predictions")
    pred_df = pred_df.join(y_pred)

    # In[34]:


    fig = plot_forecast(
        pred_df,
        forecast_columns=["fft_predictions"],
        forecast_display_names=["FFT Predictions"],
    )
    fig = format_plot(
        fig,
        title=f"FFT: MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4f} | MASE: {metrics['MASE']:.4f} | Bias: {metrics['Forecast Bias']:.4f}",
    )
    fig.update_xaxes(type="date", range=["2021-11-01", "2021-11-31"])

    fig.write_image("imgs/chapter_4/fft.png")
    fig.show()

    metric_styled = (
        pd.DataFrame(metric_record)
        .style.format(
            {
                "MAE": "{:.3f}",
                "MSE": "{:.3f}",
                "MASE": "{:.3f}",
                "Forecast Bias": "{:.2f}%",
            }
        )
        .highlight_min(
            color="lightgreen", subset=["MAE", "MSE", "MASE", "Time Elapsed"]
        )
    )



def ex_sm(pred_df):
    name = "Exponential Smoothing"
    # Suppress FutureWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        ets_model = ExponentialSmoothing(
            trend=ModelMode.ADDITIVE,
            damped=True,
            seasonal=SeasonalityMode.ADDITIVE,
            seasonal_periods=7,
            random_state=42,
        )
        with LogTime() as timer:
            y_pred, metrics = eval_model(ets_model, test_series, name=name)
        metrics["Time Elapsed"] = timer.elapsed
    metric_record.append(metrics)
    y_pred = format_y_pred(y_pred, "ets_predictions")
    pred_df = pred_df.join(y_pred)

    # In[27]

    fig = plot_forecast(
        pred_df,
        forecast_columns=["ets_predictions"],
        forecast_display_names=["Exponential Smoothing Predictions"],
    )
    fig = format_plot(
        fig,
        title=f"Exponential Smoothing: MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4f} | MASE: {metrics['MASE']:.4f} | Bias: {metrics['Forecast Bias']:.4f}",
    )
    fig.update_xaxes(type="date", range=["2021-11-01", "2021-11-31"])

    fig.write_image("imgs/chapter_4/ets.png")
    fig.show()


naive_seasonal(pred_df, 30, "univariate_window_1_naive_seasonal.png")
auto_arima(pred_df, test_series, "univariate_window_1_auto_arima.png")


#display(metric_styled)
metric_record_df = pd.DataFrame(metric_record)
metric_record_df.to_pickle("metric_record.pkl")
