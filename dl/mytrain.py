from create_data import create_dataset
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn import metrics
import torch
from dl.my_dataloaders import TimeSeriesDataModule
from dl.models import TransformerConfig, TransformerModel
import pytorch_lightning as pl

tqdm.pandas()

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# For reproduceability set a random seed
pl.seed_everything(42)

n_in = 1
n_out = 1

# Currently not using time features !!!
df = create_dataset()
#CAUTION:dates are currently not being used for debugging reasons
columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "stress"]
players = list(df['player_name_x'].unique())
torch.set_float32_matmul_precision('medium')

num_workers = 0
DEBUG = False
if DEBUG:
    num_workers = 0
else:
    num_workers = 20


#CAUTION: change batch size
datamodule = TimeSeriesDataModule(data=df,
                                  n_in=n_in,
                                  n_out=n_out,
                                  normalize="no_normalization",  # normalizing the data
                                  batch_size=256,
                                  num_workers=num_workers,
                                  column_names=columnNames)
datamodule.setup()

transformer_config = TransformerConfig(
    input_size=1,
    d_model=64,
    n_heads=4,
    n_layers=2,
    ff_multiplier=4,
    dropout=0.1,
    activation="relu",
    multi_step_horizon=n_out,
    learning_rate=1e-3,
)

model = TransformerModel(config=transformer_config)

trainer = pl.Trainer(
    max_epochs=20,
    callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
)
trainer.fit(model, datamodule)
model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])

from torchmetrics import MeanSquaredError, MeanAbsoluteError
#
def calculate(y: pd.Series, y_pred: pd.Series, name: str, y_train: pd.Series = None):
    mean_squared_error = MeanSquaredError()
    mean_absolute_error = MeanAbsoluteError()

    return {
        "Algorithm": name,
        "MAE": mean_absolute_error(torch.tensor(y.to_list()),torch.tensor(y_pred.to_list())),
        "MSE": mean_squared_error(torch.tensor(y.to_list()), torch.tensor(y_pred.to_list())),
        "RMSE": torch.sqrt(mean_absolute_error(torch.tensor(y.to_list()),torch.tensor(y_pred.to_list())))

    }

y_test = pd.read_pickle("y_test.pkl")
y_val = pd.read_pickle("y_val.pkl")

y_pred = trainer.predict(model, datamodule.test_dataloader())
y_pred = torch.cat(y_pred).squeeze().detach().numpy()

df_pred = pd.DataFrame(y_pred, columns = ['0','1', '2', '3', '4', '5', '6', '7'])

df_pred.to_pickle("y_pred.pkl")
#extract readiness from the data
y_pred_readiness = df_pred['3']
y_test_readiness = y_test['ft4_t+0']

import matplotlib.pyplot as plt
def plot_forecast_xgb(y_true, y_pred):

    plt.figure(figsize=(12, 6))
    y_true.plot(label="true", color="g")
    y_pred.plot(label="test", color="r")

    plt.savefig("plot.png")
    plt.show()

def mean_absolute_percentage_error_func(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def timeseries_evaluation_metrics_func(y_true, y_pred):

    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    #mape = mean_absolute_percentage_error_func(y_true, y_pred)
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')

    return {"MSE":mse, "MAE":mae, "RMSE":rmse}


timeseries_evaluation_metrics_func(y_test_readiness, y_pred_readiness)
plot_forecast_xgb(y_test_readiness, y_pred_readiness)
