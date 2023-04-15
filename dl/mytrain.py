from create_data import create_dataset
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn import metrics
import torch

np.random.seed(42)
tqdm.pandas()

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))



from src.dl.my_dataloaders import TimeSeriesDataModule
from src.dl.models import TransformerConfig, TransformerModel
import pytorch_lightning as pl

# For reproduceability set a random seed
pl.seed_everything(42)

n_in = 30
n_out = 1

# Currently not using time features
df = create_dataset()
#CAUTION:dates are currently not being used for debugging reasons
columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "stress"]
players = list(df['player_name_x'].unique())

#CAUTION: change batch size
datamodule = TimeSeriesDataModule(data=df,
                                  n_in=n_in,
                                  n_out=n_out,
                                  normalize="no_normalization",  # normalizing the data
                                  batch_size=32,
                                  num_workers=0,
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
    auto_select_gpus=True,
    min_epochs=30,
    max_epochs=100,
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

tag = f"SingleStep_Transformer_"
y_pred = trainer.predict(model, datamodule.test_dataloader())
# # pred is a list of outputs, one for each batch
y_pred = torch.cat(y_pred).squeeze().detach().numpy()
# # Selecting forward predictions of HORIZON timesteps, every HORIZON timesteps and flattening it
HORIZON = n_out
y_pred = y_pred[0::HORIZON].ravel()
# # Apply reverse transformation because we applied global normalization
# pred = pred * datamodule.train.std + datamodule.train.mean

y_train = pd.read_pickle("y_train")
y_test = pd.read_pickle("y_test")
y_val = pd.read_pickle("y_val")



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


timeseries_evaluation_metrics_func(y_test, y_train)
