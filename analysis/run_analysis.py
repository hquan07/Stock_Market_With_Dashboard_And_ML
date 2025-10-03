import subprocess
import argparse

def run_script(script_name):
    print(f"\n👉 Running {script_name} ...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ {script_name} completed successfully!")
    else:
        print(f"❌ {script_name} failed with error:\n{result.stderr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master runner for analysis pipeline")
    parser.add_argument("--lstm", action="store_true", help="Run LSTM forecasting as well")
    args = parser.parse_args()

    # 1. Statistics first
    run_script("statistics.py")

    # 2. Forecasting models
    run_script("forecasting_arima.py")
    run_script("forecasting_prophet.py")

    # 3. Optional: LSTM
    if args.lstm:
        run_script("forecasting_lstm.py")
