# 📊 Stock Market Dashboard + Machine Learning

This project provides a complete pipeline for analyzing and forecasting stock market data.  
It includes **ETL (Extract–Transform–Load)**, **Database storage**, **Machine Learning models**, and an **interactive Dashboard** built with Dash & Plotly.

---

## 🚀 Features
- **ETL Pipeline**: Extract data from CSV/APIs, clean and transform it, then load into a database.
- **Database Integration**: Centralized storage for structured stock market data.
- **Machine Learning Models**:
  - ARIMA (time-series forecasting)
  - LSTM (deep learning)
  - Prophet (trend and seasonality)
  - Gaussian Mixture Models & GB-LOF (anomaly detection)
- **Dashboard**:
  - OHLC & line charts
  - Forecast visualization
  - Anomaly detection highlights
  - Interactivity with ticker & date selection

---

## 📂 Project Structure
```
Stock_Market_Dashboard+ML/
│
├── dashboard/          # Dash application (UI + callbacks)
├── etl/                # ETL scripts
├── ml/                 # Machine learning models
├── db/                 # Database schema & scripts
├── utils/              # Helper functions
├── data/               # Raw & processed datasets
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/stock-market-dashboard-ml.git
   cd stock-market-dashboard-ml
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

1. **Run ETL** to prepare the dataset:
   ```bash
   python etl/run_etl.py
   ```

2. **Initialize Database**:
   ```bash
   python db/init_db.py
   ```

3. **Train Models**:
   ```bash
   python ml/train_models.py
   ```

4. **Run Dashboard**:
   ```bash
   python dashboard/app.py
   ```

Then open your browser at **http://127.0.0.1:8050/**.

---

## 📊 Example Dashboard
Features include:
- Stock price visualization (OHLC, line charts)
- Forecasts (ARIMA, LSTM, Prophet)
- Anomaly detection with alerts

---

## 📈 Dataset
- Historical stock data (CSV/APIs)
- Fields: `trade_date`, `ticker`, `open`, `high`, `low`, `close`, `volume`
- Cleaned and structured in **ETL**, stored in **database** for reusability

---

## 📌 Roadmap / Future Trends
- Add Reinforcement Learning for trading strategy simulation
- Expand to cryptocurrency datasets
- Deploy dashboard on cloud (Heroku/AWS)
- Real-time streaming data integration

---

## 📚 References
- [Dash Documentation](https://dash.plotly.com/)  
- [Prophet Forecasting Model](https://facebook.github.io/prophet/)  
- [ARIMA Guide - Statsmodels](https://www.statsmodels.org/stable/tsa.html)  

---

## 📎 Appendix
- **ETL Flow**: Raw data → Cleaning → Feature Engineering → Database  
- **Database Schema**: see `db/schema.sql`  
- **Additional Charts**: Extended visualization in `/reports/`
