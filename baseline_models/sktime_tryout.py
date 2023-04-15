from sktime.datasets import load_longley
from sktime.forecasting.arima import ARIMA

_, y = load_longley()


X_=y[["UNEMP"]]
y_ = y.drop(columns=["UNEMP", "ARMED", "POP"])

forecaster = ARIMA()
forecaster.fit(y=y_, fh=[i for i in range(0,30)])
y_pred = forecaster.predict()

forecaster.forecasters_