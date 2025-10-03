import numpy as np
from sklearn.preprocessing import MinMaxScaler
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except Exception:
    Sequential = None

class LSTMForecast:
    def __init__(self, seq_len=60):
        self.seq_len = seq_len
        self.scaler = MinMaxScaler()
        self.model = None

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(self.seq_len, len(data)):
            X.append(data[i-self.seq_len:i, 0])
            y.append(data[i, 0])
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y

    def fit(self, series, epochs=10, batch_size=32):
        if Sequential is None:
            raise ImportError("TensorFlow not installed or not available")
        data = series.values.reshape(-1,1)
        data_scaled = self.scaler.fit_transform(data)
        X, y = self._create_sequences(data_scaled)
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, last_window, n_steps=30):
        cur = np.array(last_window).reshape(-1,1)
        cur = self.scaler.transform(cur)
        preds = []
        for _ in range(n_steps):
            x = cur[-self.seq_len:].reshape(1, self.seq_len, 1)
            p = self.model.predict(x)[0,0]
            preds.append(p)
            cur = np.vstack([cur, [[p]]])
        preds = self.scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        return preds
