import pandas as pd
from pmdarima import auto_arima

class ARIMAForecast:
    def __init__(self, seasonal=False):
        self.model = None
        self.seasonal = seasonal

    def fit(self, series):
        self.model = auto_arima(series.dropna(), seasonal=self.seasonal, stepwise=True, suppress_warnings=True)
        return self.model

    def predict(self, n_periods=30):
        return self.model.predict(n_periods=n_periods)
