from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_DATASET = DATA_DIR / "all_stocks_5yr.csv"

EXTRACTED = DATA_DIR / "stocks_extracted.csv"

TRANSFORMED = DATA_DIR / "stocks_transformed.csv"
