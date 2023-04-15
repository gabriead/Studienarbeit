import pandas as pd

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
        if i == 0:
            names += [('ft%d_t' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('ft%d_t+%d' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def create_features():
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

def create_dataset():
    df = create_features()
    #agg = series_to_supervised(df, n_in, n_out, dropnan=True)
    #return agg
    return df