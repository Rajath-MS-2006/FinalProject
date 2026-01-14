import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import requests
import numpy as np


# -----------------------------------------------------
# Load & clean data (safe)
# -----------------------------------------------------
def load_data(data_dir="data/analyzed.csv", window_days=45):
    df = pd.read_csv(data_dir)

    # Parse timestamp safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Remove timezone — REQUIRED for Prophet
    try:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    except:
        try:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        except:
            pass

    # Keep only last X days (default 45)
    latest = df["timestamp"].max()
    cutoff = latest - pd.Timedelta(days=window_days)
    df = df[df["timestamp"] >= cutoff]

    # Sort chronologically
    df = df.sort_values("timestamp")

    return df[["timestamp", "score"]].dropna()



# -----------------------------------------------------
# Run Prophet Forecast — CLEAN & STABLE VERSION
# -----------------------------------------------------
def run_prophet_forecast(df, periods=14):

    # Ensure tz-naive
    try:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    except:
        try:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        except:
            pass

    # REAL daily aggregation — NO resampling, NO interpolation
    df["date"] = df["timestamp"].dt.date

    daily = (
        df.groupby("date")["score"]
          .mean()
          .reset_index()
    )
    daily["ds"] = pd.to_datetime(daily["date"])
    daily["y"] = daily["score"]
    daily = daily[["ds", "y"]].sort_values("ds")

    # Prophet model
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.08,
        seasonality_prior_scale=6,
        seasonality_mode="additive"
    )

    # Sinusoidal weekly pattern
    m.add_seasonality(
        name="weekly_cycle",
        period=7,
        fourier_order=5
    )

    m.fit(daily)

    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)

    return forecast, daily


# -----------------------------------------------------
# Generate Forecast Plot
# -----------------------------------------------------
def generate_forecast_plot(forecast, daily, lock, out_path):

    with lock:
        plt.figure(figsize=(12, 6))

        # Confidence interval
        plt.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            color="skyblue",
            alpha=0.3
        )

        # Forecast line
        plt.plot(forecast["ds"], forecast["yhat"], color="blue", label="Forecast", linewidth=2)

        # Actual points
        plt.scatter(daily["ds"], daily["y"], color="black", s=16, label="Actual")

        plt.title("AI Market Sentiment Forecast", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Sentiment Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig(out_path, dpi=130)
        plt.close()


# -----------------------------------------------------
# Slack Alert
# -----------------------------------------------------
def send_slack_forecast_alert(_, summary):
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        print("Slack webhook missing")
        return False

    message = {
        "text": (
            "*AI Sentiment Forecast Alert*\n"
            f"Latest Sentiment: {summary['last_y']}\n"
        )
    }

    r = requests.post(webhook, json=message)
    return r.status_code == 200
