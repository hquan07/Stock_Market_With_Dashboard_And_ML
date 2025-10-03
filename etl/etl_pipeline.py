import pandas as pd
import mysql.connector
from utils.paths import RAW_DATASET, EXTRACTED, TRANSFORMED


def run_extract():
    print("📥 Extracting data from:", RAW_DATASET)
    df = pd.read_csv(RAW_DATASET)

    df = df[["date", "open", "high", "low", "close", "volume", "Name"]]

    df.to_csv(EXTRACTED, index=False)
    print(f"✅ Extracted data saved to {EXTRACTED}")


def run_transform():
    print("🔄 Transforming data from:", EXTRACTED)
    df = pd.read_csv(EXTRACTED)

    df.rename(columns={"date": "trade_date", "Name": "ticker"}, inplace=True)

    df["daily_return"] = df.groupby("ticker")["close"].pct_change()

    df = df.dropna()

    df.to_csv(TRANSFORMED, index=False)
    print(f"✅ Transformed data saved to {TRANSFORMED}")


def run_load():
    print("📤 Loading data into MySQL from:", TRANSFORMED)
    df = pd.read_csv(TRANSFORMED)

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Huyquan160720040@",
        database="stock_db"
    )
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute(
            """
            INSERT INTO prices (ticker, trade_date, open, high, low, close, volume, daily_return)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row["ticker"],
                row["trade_date"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
                row.get("daily_return"),
            )
        )

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Data loaded into MySQL")


def run_pipeline():
    run_extract()
    run_transform()
    run_load()
    print("🚀 ETL pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()