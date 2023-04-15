import os
import shutil

import joblib
from itertools import cycle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from tqdm.notebook import tqdm

#pio.templates.default = "plotly_white"
import torch
from pathlib import Path


from src.utils import plotting_utils

#from src.forecasting.ml_forecasting import calculate_metrics
#from src.utils import ts_utils

np.random.seed(42)
tqdm.pandas()

os.makedirs("imgs/chapter_13", exist_ok=True)
preprocessed = Path(r"C:/Users/adria/Desktop/Modern-Time-Series-Forecasting-with-Python/data/data/london_smart_meters/preprocessed")
output = Path("data/data/london_smart_meters/output")
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def format_plot(fig, legends=None, xlabel="Time", ylabel="Value", title="", font_size=15):
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
        )
    )
    return fig

def plot_forecast(pred_df, forecast_columns, forecast_display_names=None):
    if forecast_display_names is None:
        forecast_display_names = forecast_columns
    else:
        assert len(forecast_columns) == len(forecast_display_names)
    mask = ~pred_df[forecast_columns[0]].isnull()
    colors = [
        "rgba(" + ",".join([str(c) for c in plotting_utils.hex_to_rgb(c)]) + ",<alpha>)"
        for c in px.colors.qualitative.Plotly
    ]
    act_color = colors[0]
    colors = cycle(colors[1:])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pred_df[mask].index,
            y=pred_df[mask].energy_consumption,
            mode="lines",
            line=dict(color=act_color.replace("<alpha>", "0.9")),
            name="Actual Consumption",
        )
    )
    for col, display_col in zip(forecast_columns, forecast_display_names):
        fig.add_trace(
            go.Scatter(
                x=pred_df[mask].index,
                y=pred_df.loc[mask, col],
                mode="lines",
                line=dict(dash="dot", color=next(colors).replace("<alpha>", "1")),
                name=display_col,
            )
        )
    return fig

def highlight_abs_min(s, props=''):
    return np.where(s == np.nanmin(np.abs(s.values)), props, '')


try:
    #Reading the missing value imputed and train test split data
    train_df = pd.read_parquet(preprocessed/"selected_blocks_train_missing_imputed_feature_engg.parquet")
    # Read in the Validation dataset as test_df so that we predict on it
    test_df = pd.read_parquet(preprocessed/"selected_blocks_val_missing_imputed_feature_engg.parquet")
    # test_df = pd.read_parquet(preprocessed/"block_0-7_test_missing_imputed_feature_engg.parquet")
except FileNotFoundError:
    print("ERROR reading data")


#Preparing data
target = "energy_consumption"
index_cols = ["LCLid", "timestamp"]

# Setting the indices
train_df.set_index(index_cols, inplace=True, drop=False)
test_df.set_index(index_cols, inplace=True, drop=False)

sample_train_df = train_df.xs("MAC000193")
sample_test_df = test_df.xs("MAC000193")
# Creating a pred_df with actuals
pred_df = pd.concat([sample_train_df[[target]], sample_test_df[[target]]])

sample_val_df = sample_train_df.loc["2013-12"]
sample_train_df = sample_train_df.loc[:"2013-11"]

sample_train_df['type'] = "train"
sample_val_df['type'] = "val"
sample_test_df['type'] = "test"
sample_df = pd.concat([sample_train_df[[target, "type"]], sample_val_df[[target, "type"]], sample_test_df[[target, "type"]]])
sample_df.head()

try:
    pred_df = pd.read_pickle(output/"dl_seq_2_seq_w_attn_prediction_val_df_MAC000193.pkl")
    metric_record = joblib.load(output/"dl_seq_2_seq_w_attn_metrics_val_df_MAC000193.pkl")

    sel_metrics = [
        "MultiStep LSTM_LSTM_teacher_forcing_1",
        "MultiStep_Seq2Seq_dot_Attn_teacher_forcing_1",
    ]
    metric_record = [i for i in metric_record if i["Algorithm"] in sel_metrics]
except FileNotFoundError:
    print("comparison models not found")

from src.dl.dataloaders import TimeSeriesDataModule
from src.dl.models import TransformerConfig, TransformerModel
import pytorch_lightning as pl
import torch
# For reproduceability set a random seed
pl.seed_everything(42)


#multi-step prediction
HORIZON = 13
WINDOW = 17


datamodule = TimeSeriesDataModule(data = sample_df[[target]],
        n_val = sample_val_df.shape[0],
        n_test = sample_test_df.shape[0],
        window = WINDOW,
        horizon = HORIZON,
        normalize = "non_normalizaiton", # normalizing the data
        batch_size = 32,
        num_workers = 0)
datamodule.setup()


transformer_config = TransformerConfig(
    input_size=1,
    d_model=64,
    n_heads=4,
    n_layers=2,
    ff_multiplier=4,
    dropout=0.1,
    activation="relu",
    multi_step_horizon=HORIZON,
    learning_rate=1e-3,
)

model = TransformerModel(config=transformer_config)

trainer = pl.Trainer(
    auto_select_gpus=True,
    min_epochs=5,
    max_epochs=100,
    callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
)
trainer.fit(model, datamodule)
# model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
# Removing artifacts created during training
#shutil.rmtree("lightning_logs")


tag = f"MultiStep_Transformer_Multi_Step_FF_decoder"
pred = trainer.predict(model, datamodule.test_dataloader())
# pred is a list of outputs, one for each batch
pred = torch.cat(pred).squeeze().detach().numpy()
# Selecting forward predictions of HORIZON timesteps, every HORIZON timesteps and flattening it
pred = pred[0::48].ravel()
# Apply reverse transformation because we applied global normalization
pred = pred * datamodule.train.std + datamodule.train.mean
pred_df_ = pd.DataFrame({tag: pred}, index=sample_test_df.index)
pred_df = pred_df.join(pred_df_)
metrics = calculate_metrics(sample_test_df[target], pred_df_[tag], tag, pd.concat([sample_train_df[target],sample_val_df[target]]))
# metrics
metric_record.append(metrics)


formatted = pd.DataFrame(metric_record).style.format({"MAE": "{:.4f}",
                          "MSE": "{:.4f}",
                          "MASE": "{:.4f}",
                          "Forecast Bias": "{:.2f}%"})
formatted.highlight_min(color='lightgreen', subset=["MAE","MSE","MASE"]).apply(highlight_abs_min, props='color:black;background-color:lightgreen')

fig = plot_forecast(pred_df, forecast_columns=[tag], forecast_display_names=[tag])
fig = format_plot(fig, title=f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4f} | MASE: {metrics['MASE']:.4f} | Bias: {metrics['Forecast Bias']:.4f}")
fig.update_xaxes(type="date", range=["2014-01-01", "2014-01-08"])
fig.write_image(f"imgs/chapter_13/{tag}.png")
fig.show()

shutil.rmtree("lightning_logs")

pred_df.to_pickle(output/"dl_transformers_val_df_MAC000193.pkl")
joblib.dump(metric_record, output/"dl_transformers_metrics_val_df_MAC000193.pkl")