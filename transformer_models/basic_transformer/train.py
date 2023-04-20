from create_data import create_dataset
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn import metrics
import torch
from transformer_models.basic_transformer.dataloaders import TimeSeriesDataModule
from transformer_models.basic_transformer.models import TransformerConfig, TransformerModel
import pytorch_lightning as pl

tqdm.pandas()
import matplotlib.pyplot as plt

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# For reproduceability set a random seed
pl.seed_everything(42)

#CAUTION: take care of normalization
#CAUTION: change batch size


def plot_forecast_xgb(y_true, y_pred, player, df):

    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(12, 6))
    plt.title(f"MAE: {mae:.2f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}", size=18)
    y_true.plot(label="true", color="g")
    y_pred.plot(label="test", color="r")

    plt.savefig(f'''plot_{player}.png''')
    return mse, mae, rmse


    #plt.show()

def mean_absolute_percentage_error_func(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Adapt to save metrics
def timeseries_evaluation_metrics_func(y_true, y_pred, player):

    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    return {"MSE":mse, "MAE":mae, "RMSE":rmse}

def pipeline():
    n_in = 7
    n_out = 1
    metrics_df = pd.DataFrame(columns = ["player_name", "mae", "rmse", "mse", "n_in", "n_out", "features_in", "features_out"])

    player_names = []
    maes = []
    rmses = []
    mses= []
    n_ins= []
    n_outs = []
    features_in = []
    features_out = []

    # Currently not using time features !!!
    df = create_dataset()
    # CAUTION:dates are currently not being used for debugging reasons
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "stress"]

    #filter teams

    df = df[df["player_name_x"].str.startswith("TeamA")]
    players = list(df['player_name_x'].unique())
    torch.set_float32_matmul_precision('medium')

    num_workers = 0
    DEBUG = False
    print("DEBUGGING STAGE", DEBUG)
    if DEBUG:
        num_workers = 0
    else:
        num_workers = 20

    for i in range(0,len(players)):
        print("Current player", players[i])

        datamodule = TimeSeriesDataModule(data=df,
                                          n_in=n_in,
                                          n_out=n_out,
                                          normalize="no_normalization",  # normalizing the data
                                          batch_size=128,
                                          num_workers=num_workers,
                                          column_names=columnNames,
                                          player_index=i)
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
            min_epochs=12,
            max_epochs=20,
            callbacks=[pl.callbacks.EarlyStopping(monitor="train_loss", patience=4)],
        )


        trainer.fit(model, datamodule)
        model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])

        y_pred = trainer.predict(model, datamodule.test_dataloader())
        y_pred = torch.cat(y_pred).squeeze().detach().numpy()



        # rework mapping of correct columns
        df_pred = pd.DataFrame(y_pred, columns = ['0','1', '2', '3', '4', '5', '6', '7'])

        y_pred_readiness = df_pred['3']
        y_test_readiness = datamodule.test_dataloader().dataset.y_test_selected['ft4_t+0']

        n_in_columns = list(datamodule.test_dataloader().dataset.X_train.columns.values)

        if y_pred_readiness.shape[0] == y_test_readiness.shape[0]:
            timeseries_evaluation_metrics_func(y_test_readiness, y_pred_readiness, players[i])
            mse, mae, rmse = plot_forecast_xgb(y_test_readiness, y_pred_readiness, players[i], df)

            maes.append(mae)
            rmses.append(rmse)
            mses.append(mse)
            n_ins.append(n_in)
            n_outs.append(n_out)
            features_in.append(n_in_columns)
            features_out.append(list(datamodule.test_dataloader().dataset.y_test_selected.columns.values))
            player_names.append(players[i])
        else:
            print("Couldn't calculate Player", players[i])
            continue

    metrics_df["player_name"] = player_names
    metrics_df["mae"] = maes
    metrics_df["rmse"] = rmse
    metrics_df["mse"] = mse
    metrics_df["n_in"] = n_ins
    metrics_df["n_out"] = n_outs
    metrics_df["features_in"] = features_in
    metrics_df["features_out"] = features_out

    metrics_df.to_pickle("metrics_df.pkl")

pipeline()
