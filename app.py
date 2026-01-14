import os
import threading
import queue
import base64
import json
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, Response, send_from_directory, request
import pandas as pd

import matplotlib
matplotlib.use("Agg")

# Modules
import data_fetch_and_analyse as m2
import forecast_and_slack as m3

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
load_dotenv()

DATA_DIR = "data"
PLOTS_DIR = "plots"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "sentiment-dashboard"

plot_lock = threading.Lock()
clients = []

# -------------------------------------------------------------------
# SSE
# -------------------------------------------------------------------
def broadcast(event_type, payload):
    msg = json.dumps({
        "type": event_type,
        "payload": payload,
        "ts": datetime.now().isoformat()
    })

    for q in clients[:]:
        try:
            q.put(msg, block=False)
        except:
            try:
                clients.remove(q)
            except:
                pass


@app.route("/stream")
def stream():
    def gen(q):
        yield 'data: {"type":"connected","payload":{"msg":"connected"}}\n\n'
        while True:
            try:
                msg = q.get(timeout=25)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    q = queue.Queue()
    clients.append(q)
    return Response(gen(q), mimetype="text/event-stream")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def read_analyzed_df():
    path = os.path.join(DATA_DIR, "analyzed.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except:
        return pd.DataFrame()

    if "score" in df:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


# -------------------------------------------------------------------
# Static plot route
# -------------------------------------------------------------------
@app.route("/plots/<filename>")
def plots_static(filename):
    return send_from_directory(PLOTS_DIR, filename)


# -------------------------------------------------------------------
# Timeline JSON for Plotly
# -------------------------------------------------------------------
@app.route("/chart/timeline_data")
def timeline_data():
    df = read_analyzed_df()
    if df.empty:
        return jsonify({"dates": [], "positive": [], "neutral": [], "negative": []})

    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date

    pivot = df.pivot_table(
        index="date",
        columns="label",
        values="score",
        aggfunc="count",
        fill_value=0
    ).reset_index()

    return jsonify({
        "dates": pivot["date"].astype(str).tolist(),
        "positive": pivot.get("positive", [0]*len(pivot)).tolist(),
        "neutral": pivot.get("neutral", [0]*len(pivot)).tolist(),
        "negative": pivot.get("negative", [0]*len(pivot)).tolist()
    })


# -------------------------------------------------------------------
# Pipeline Runner
# -------------------------------------------------------------------
pipeline_running = False
pipeline_lock = threading.Lock()

# Disable automatic slack alerts inside analysis
m2.send_slack_alert = lambda *_: None


def run_pipeline():
    global pipeline_running
    with pipeline_lock:
        if pipeline_running:
            broadcast("log", {"msg": "Pipeline already running"})
            return
        pipeline_running = True

    try:
        broadcast("progress", {"pct": 0})
        broadcast("log", {"msg": "Fetching 50 NewsAPI articles..."})

        news = m2.fetch_newsapi_articles(m2.AI_QUERIES, total_records=50)
        broadcast("log", {"msg": f"Fetched {len(news)} news items"})
        broadcast("progress", {"pct": 10})

        broadcast("log", {"msg": "Fetching 50 Reddit posts..."})
        reddit = m2.fetch_reddit_posts(m2.REDDIT_SUBREDDITS, total_records=50)
        broadcast("log", {"msg": f"Fetched {len(reddit)} reddit posts"})
        broadcast("progress", {"pct": 20})

        raw = pd.DataFrame(news + reddit)
        raw.to_csv(os.path.join(DATA_DIR, "raw.csv"), index=False)
        broadcast("log", {"msg": "Saved raw.csv"})
        broadcast("progress", {"pct": 30})

        broadcast("log", {"msg": "Running sentiment analysis..."})
        analyzed = m2.analyze_sentiments(
            raw, batch_size=10,
            progress_cb=lambda pct, msg: (
    broadcast("log", {"msg": f"[Analysis] {msg}"}),
    broadcast("progress", {"pct": int(30 + pct*0.4)})
)

        )

        if isinstance(analyzed, pd.DataFrame):
            analyzed.to_csv(os.path.join(DATA_DIR, "analyzed.csv"), index=False)
            broadcast("log", {"msg": f"Analysis done: {len(analyzed)} rows"})
        else:
            analyzed = pd.DataFrame()
            broadcast("log", {"msg": "Analysis failed"})

        broadcast("progress", {"pct": 70})
        broadcast("log", {"msg": "Generating charts..."})

        m2.generate_all_charts(analyzed, PLOTS_DIR, plot_lock)
        broadcast("progress", {"pct": 90})

        broadcast("log", {"msg": "Pipeline completed successfully"})
        broadcast("progress", {"pct": 100})
        broadcast("status", {"status": "completed"})

    except Exception as e:
        broadcast("log", {"msg": f"ERROR: {e}"})
        broadcast("status", {"status": "error"})
    finally:
        pipeline_running = False

@app.route("/start_pipeline", methods=["POST"])
def start_pipeline():
    global pipeline_running
    if pipeline_running:
        return jsonify({"status": "already_running"})

    threading.Thread(target=run_pipeline, daemon=True).start()
    return jsonify({"status": "started"})


# -------------------------------------------------------------------
# Forecast
# -------------------------------------------------------------------
@app.route("/run_forecast", methods=["POST"])
def run_forecast():
    def task():
        broadcast("log", {"msg": "Preparing forecast..."})

        df = m3.load_data()
        if df is None or df.empty:
            broadcast("log", {"msg": "No data to forecast"})
            broadcast("status", {"forecast": "error"})
            return

        forecast_df, daily_df = m3.run_prophet_forecast(df, periods=14)

        out_path = os.path.join(PLOTS_DIR, "prophet_sentiment_forecast.png")
        m3.generate_forecast_plot(forecast_df, daily_df, plot_lock, out_path)

        # Preview in UI
        with open(out_path, "rb") as f:
            img64 = base64.b64encode(f.read()).decode()
            broadcast("forecast", {"image_b64": img64})

        broadcast("log", {"msg": "Forecast ready"})
        broadcast("status", {"forecast": "ok", "plot": "/plots/prophet_sentiment_forecast.png"})

    threading.Thread(target=task, daemon=True).start()
    return jsonify({"status": "started"})


# -------------------------------------------------------------------
# Slack Alert
# -------------------------------------------------------------------
@app.route("/send_forecast_alert", methods=["POST"])
def send_alert():
    try:
        df = m3.load_data()
        if df.empty:
            return jsonify({"status": "no_data"}), 400

        forecast_df, daily_df = m3.run_prophet_forecast(df, periods=14)

        # 14-day forecast uses 15-min intervals = 96 points per day
        window = 14 * 96

        last_y = daily_df["y"].iloc[-1] if len(daily_df) else 0
        future_mean = forecast_df["yhat"].tail(window).mean()

        summary = {
    "last_y": round(last_y, 4),
}


        ok = m3.send_slack_forecast_alert(None, summary)
        return jsonify({"status": "sent" if ok else "failed", "summary": summary})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# -------------------------------------------------------------------
# UI Routes
# -------------------------------------------------------------------
@app.route("/")
def index():
    df = read_analyzed_df()
    stats = {
        "total_records": len(df),
        "positive_count": len(df[df["label"] == "positive"]) if not df.empty else 0,
        "neutral_count": len(df[df["label"] == "neutral"]) if not df.empty else 0,
        "negative_count": len(df[df["label"] == "negative"]) if not df.empty else 0,
    }
    return render_template("index.html", stats=stats)


@app.route("/data")
def data_view():
    df = read_analyzed_df()
    return render_template("data_view.html", data=df.fillna("").to_dict(orient="records"))


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
