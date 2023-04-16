from typing import List, Tuple, Union
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class TimeSeriesDataset:

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):

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


    def create_data(self, df, columnNames, n_in, n_out, multistep=False, target="readines"):
        players = list(df['player_name_x'].unique())
        print("amount of players to train each config: ", len(players))

        # activate all players
        for i in range(1):

            # print(i+1,"/",len(players))

            all_but_one = players[:i] + players[i + 1:-1]
            last_player = players[len(players) - 1]

            df_train = df[df['player_name_x'].isin(all_but_one)]
            df_test = df.loc[df['player_name_x'] == players[i]]
            df_val = df.loc[df['player_name_x'] == last_player]

            train = df_train[columnNames]
            test = df_test[columnNames]
            val = df_val[columnNames]

            train = train.reset_index(drop=True)
            test = test.reset_index(drop=True)
            val = val.reset_index(drop=True)

            num_features = len(train.columns.tolist())

            train_scalar = MinMaxScaler()
            train_transformed = train_scalar.fit_transform(train)
            test_transformed = train_scalar.transform(test)
            val_transformed = train_scalar.transform(val)

            if not multistep:
                train_direct = self.series_to_supervised(train_transformed.copy(), n_in, n_out)
                test_direct = self.series_to_supervised(test_transformed.copy(), n_in, n_out)
                val_direct = self.series_to_supervised(val_transformed.copy(), n_in, n_out)

                features_m = train_direct.columns.tolist()[:num_features * n_out]
                targets_m = test_direct.columns.tolist()[num_features:]

                features = train_direct.columns.tolist()[:(n_in*num_features)]
                targets = test_direct.columns.tolist()[(n_in*num_features):]

                n_vars = 1 if type(train_transformed) is list else train_transformed.shape[1]
                features_ = list()
                targets_ = list()

                # for i in range(n_in, 0, -1):
                #     for k in range(n_vars):
                #         features_.append(columnNames[k] + '_t-%d' % i)
                #
                # for i in range(n_out):
                #     targets_.append(columnNames[i] + '_t+%d' % n_out)

                X_train = train_direct[features]
                #X_train.columns = features_

                y_train = train_direct[targets]
                #y_train.columns = targets_

                X_test = test_direct[features]
                #X_test.columns = features_

                y_test = test_direct[targets]
               # y_test.columns = targets_

                X_val = val_direct[features]
                #X_val.columns = features_

                y_val = val_direct[targets]
               #y_val.columns = targets_

                # select targets to predict
                # target_columns = y_train.columns
                #
                # selected_columns = [target_columns[i] for i in range(len(target_columns)) if
                #                     target in target_columns[i]]

                self.X_test = X_test
                self.X_train = X_train
                self.X_val = X_val
                #
                # self.y_train_selected = y_train[selected_columns]
                # self.y_test_selected = y_test[selected_columns]
                # self.y_val_selected = y_val[selected_columns]

                self.y_train_selected = y_train
                self.y_test_selected = y_test
                self.y_val_selected = y_val

                y_train.to_pickle("y_train.pkl")
                y_test.to_pickle("y_test.pkl")
                y_val.to_pickle("y_val.pkl")

                X_test.to_pickle("X_test.pkl")
                X_val.to_pickle("X_val.pkl")


    def __init__(
        self,
        data: pd.DataFrame,
        n_in: int,
        n_out: int,
        column_names: str,
        #n_val: Union[float, int] = 0.2,
        #n_test: Union[float, int] = 0.2,
        normalize: str = "None",  # options are "none", "local", "global"
        normalize_params: Tuple[
            float, float
        ] = None,  # tuple of mean and std for pre-calculated standardization
        mode="train",  # options are "train", "val", "test"
    ):

        self.column_names = column_names

        if data.ndim==1:
            data = data.reshape(-1,1)
        if normalize == "global" and mode != "train":
            assert (
                isinstance(normalize_params, tuple)
                and len(normalize_params) == 2
            ), "If using Global Normalization, in valid and test mode normalize_params argument should be a tuple of precalculated mean and std"
        self.data = data.copy()

        self.n_in = n_in
        self.n_out = n_out
        self.normalize = normalize
        self.mode = mode

        self.create_data(self.data, self.column_names, n_in, n_out, multistep=False)

        if mode == "train":
            self.data = self.X_train
            self.y = self.y_train_selected
        elif mode == "val":
            self.data = self.X_val
            self.y = self.y_val_selected
        elif mode == "test":
            self.data = self.X_test
            self.y = self.y_test_selected

        # This is the actual input on which to iterate
        #self.data = data[start_index:end_index, :]
        #CAUTION: fix normalization once the pipeline is running

        # if normalize == "global":
        #     if mode == "train":
        #         self.mean = data.mean()
        #         self.std = data.std()
        #     else:
        #         self.mean, self.std = normalize_params
        #     self.data = (self.data - self.mean) / self.std

    def __len__(self):
        #return len(self.data) - self.n_out*len(self.column_names) - self.n_in*len(self.column_names) + 1 #to account for zero indexing
        return len(self.data)

    def __getitem__(self, idx):

        x = self.data.iloc[idx, :]
        y = self.y.iloc[idx,:]

        x_transformed = x.reset_index().drop(["index"], axis=1).to_numpy()
        y_transformed = y.reset_index().drop(["index"], axis=1).to_numpy()

        x_transformed = x_transformed.astype(np.float32)
        y_transformed = y_transformed.astype(np.float32)

        return x_transformed, y_transformed

class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        n_in: int = 10,
        n_out: int = 1,
        normalize: str = "none",
        batch_size: int = 32,
        num_workers: int = 0,
        column_names: str = ""
    ):
        super().__init__()
        self.data = data
        self.n_in = n_in
        self.n_out = n_out
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self._is_global = normalize=="global"
        self.column_names = column_names

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = TimeSeriesDataset(
                data=self.data,
                n_in=self.n_in,
                n_out=self.n_out,
                normalize=self.normalize,
                normalize_params= None,
                mode="train",
                column_names= self.column_names
            )
            self.val = TimeSeriesDataset(
                data=self.data,
                n_in=self.n_in,
                n_out=self.n_out,
                normalize=self.normalize,
                normalize_params= (self.train.mean, self.train.std) if self._is_global else None,
                mode="val",
                column_names=self.column_names
            )
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = TimeSeriesDataset(
                data=self.data,
                n_in=self.n_in,
                n_out=self.n_out,
                normalize=self.normalize,
                normalize_params= (self.train.mean, self.train.std) if self._is_global else None,
                mode="test",
                column_names=self.column_names
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
