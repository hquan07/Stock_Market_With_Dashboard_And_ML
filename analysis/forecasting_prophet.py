import pandas as pd
try:
    from prophet import Prophet
except Exception:
    Prophet = None

def prepare_prophet_df(df, date_col="date", price_col="adj_close"):
    tmp = df[[date_col, price_col]].rename(columns={date_col: "ds", price_col: "y"}).copy()
    tmp["ds"] = pd.to_datetime(tmp["ds"])
    return tmp

class ProphetForecast:
    def __init__(self, daily_seasonality=True):
        if Prophet is None:
            raise ImportError("prophet package not installed")
        self.model = Prophet(daily_seasonality=daily_seasonality)

    def fit(self, df):
        dfp = prepare_prophet_df(df)
        self.model.fit(dfp)

    def predict(self, periods=30, freq="D"):
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        fcst = self.model.predict(future)
        return fcst
