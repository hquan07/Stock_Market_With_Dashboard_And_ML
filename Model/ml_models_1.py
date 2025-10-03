import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import os
import warnings
warnings.filterwarnings('ignore')

# ================= Config =================
DATA_DIR = "../data"
INPUT_FILE = os.path.join(DATA_DIR, "/home/hquan07/Stock_Market_Dashboard+ML _1/data/stocks_transformed.csv")
OUTPUT_GB_LOF = os.path.join(DATA_DIR, "results_with_gb_lof.csv")
OUTPUT_CLUSTER = os.path.join(DATA_DIR, "ticker_clusters_gmm.csv")
OUTPUT_ARIMA = os.path.join(DATA_DIR, "arima_forecasts.csv")

def run_models():
    # ================= Load Data =================
    df = pd.read_csv(INPUT_FILE)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values(["ticker", "trade_date"])

    # ================= Feature Engineering =================
    df['return'] = df.groupby('ticker')['close'].pct_change()
    df['ma5'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).mean())
    df['volatility'] = df.groupby('ticker')['return'].transform(lambda x: x.rolling(20).std())
    df['target'] = (df.groupby('ticker')['close'].shift(-1) > df['close']).astype(int)

    df = df.dropna()

    # ================= 1. Gradient Boosting Classifier =================
    print("=== Training Gradient Boosting Classifier ===")
    features = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma20', 'volatility']
    X = df[features]
    y = df['target']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    gb = GradientBoostingClassifier(
        n_estimators=200, 
        max_depth=5, 
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)

    print("=== Gradient Boosting Performance ===")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    df.loc[X_test.index, "gb_pred"] = y_pred

    # ================= 2. Local Outlier Factor (LOF) =================
    print("\n=== Detecting Anomalies with LOF ===")
    lof_features = ['return', 'volume', 'volatility']
    lof_df = df[lof_features].fillna(0)

    lof = LocalOutlierFactor(
        n_neighbors=20, 
        contamination=0.02,
        novelty=False
    )
    df['anomaly'] = lof.fit_predict(lof_df)  # -1 = anomaly, 1 = normal

    print("Anomaly counts:")
    print(df['anomaly'].value_counts())
    
    # LOF scores (negative outlier factor)
    df['lof_score'] = lof.negative_outlier_factor_

    # ================= 3. Gaussian Mixture Models (GMM) =================
    print("\n=== Clustering with Gaussian Mixture Models ===")
    agg = df.groupby('ticker').agg({
        'return': 'mean',
        'volatility': 'mean',
        'volume': 'mean'
    }).reset_index()

    X_cluster = agg[['return', 'volatility', 'volume']].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    gmm = GaussianMixture(
        n_components=4, 
        covariance_type='full',
        random_state=42
    )
    agg['cluster'] = gmm.fit_predict(X_scaled)
    agg['cluster_proba'] = gmm.predict_proba(X_scaled).max(axis=1)  # Probability of assigned cluster

    print("=== GMM Clusters ===")
    print(agg.head())
    print("\nCluster distribution:")
    print(agg['cluster'].value_counts().sort_index())

    # ================= 4. ARIMA / SARIMA Forecasting =================
    print("\n=== Running ARIMA/SARIMA Forecasts ===")
    
    # Select top tickers for ARIMA (to avoid excessive computation)
    top_tickers = df['ticker'].value_counts().head(5).index.tolist()
    
    arima_results = []
    
    for ticker in top_tickers:
        print(f"\nForecasting for {ticker}...")
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.set_index('trade_date')
        
        # Use closing prices for time series
        ts = ticker_data['close']
        
        # Split into train/test (last 30 days for testing)
        train_size = len(ts) - 30
        if train_size < 50:  # Need enough data for ARIMA
            print(f"Skipping {ticker} - insufficient data")
            continue
            
        train, test = ts[:train_size], ts[train_size:]
        
        try:
            # Try SARIMA first (with seasonal component)
            # order=(p,d,q), seasonal_order=(P,D,Q,s)
            model = SARIMAX(
                train, 
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 5),  # 5-day seasonality
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)
            
            # Forecast
            forecast = fitted_model.forecast(steps=len(test))
            
            # Calculate metrics
            mae = np.mean(np.abs(test.values - forecast.values))
            rmse = np.sqrt(np.mean((test.values - forecast.values)**2))
            mape = np.mean(np.abs((test.values - forecast.values) / test.values)) * 100
            
            print(f"  Model: SARIMA(1,1,1)(1,1,1,5)")
            print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            
            # Store results
            for date, actual, pred in zip(test.index, test.values, forecast.values):
                arima_results.append({
                    'ticker': ticker,
                    'date': date,
                    'actual': actual,
                    'forecast': pred,
                    'model': 'SARIMA',
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape
                })
                
        except Exception as e:
            print(f"  SARIMA failed, trying ARIMA: {str(e)[:50]}")
            
            try:
                # Fall back to simple ARIMA
                model = ARIMA(train, order=(1, 1, 1))
                fitted_model = model.fit()
                
                forecast = fitted_model.forecast(steps=len(test))
                
                mae = np.mean(np.abs(test.values - forecast.values))
                rmse = np.sqrt(np.mean((test.values - forecast.values)**2))
                mape = np.mean(np.abs((test.values - forecast.values) / test.values)) * 100
                
                print(f"  Model: ARIMA(1,1,1)")
                print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
                
                for date, actual, pred in zip(test.index, test.values, forecast.values):
                    arima_results.append({
                        'ticker': ticker,
                        'date': date,
                        'actual': actual,
                        'forecast': pred,
                        'model': 'ARIMA',
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape
                    })
                    
            except Exception as e2:
                print(f"  ARIMA also failed: {str(e2)[:50]}")
                continue

    # ================= Save Results =================
    os.makedirs(DATA_DIR, exist_ok=True)
    
    df.to_csv(OUTPUT_GB_LOF, index=False)
    print(f"\n✅ Saved Gradient Boosting + LOF results to: {OUTPUT_GB_LOF}")
    
    agg.to_csv(OUTPUT_CLUSTER, index=False)
    print(f"✅ Saved GMM clustering results to: {OUTPUT_CLUSTER}")
    
    if arima_results:
        arima_df = pd.DataFrame(arima_results)
        arima_df.to_csv(OUTPUT_ARIMA, index=False)
        print(f"✅ Saved ARIMA/SARIMA forecasts to: {OUTPUT_ARIMA}")
    else:
        print("⚠️  No ARIMA forecasts generated")

    print("\n=== Summary ===")
    print(f"Total records processed: {len(df)}")
    print(f"Anomalies detected (LOF): {(df['anomaly'] == -1).sum()}")
    print(f"Clusters (GMM): {agg['cluster'].nunique()}")
    if arima_results:
        print(f"ARIMA forecasts generated for {len(top_tickers)} tickers")


if __name__ == "__main__":
    run_models()