# data_fetch_and_analyse.py
import os
import re
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import praw
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import google.generativeai as genai


load_dotenv()

# ------------------- ENV -------------------
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "ai-sentiment-bot")

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "raw.csv")
ANALYZED_FILE = os.path.join(DATA_DIR, "analyzed.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------- QUERIES -------------------
AI_QUERIES = [
    "artificial intelligence",
    "machine learning",
    "generative AI",
    "AI industry trends",
    "AI startups",
    "deep learning"
]

REDDIT_SUBREDDITS = [
    "generativeAI",
    "MachineLearning",
    "deep_learning",
    "datascience",
    "learnmachinelearning",
    "OpenAI",
    "GPT3"
]

# ------------------- CLEAN TEXT -------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", str(text))
    text = re.sub(r"[\r\n]+", " ", text)
    return text.strip()


def is_ai_related(text):
    txt = text.lower()
    for q in AI_QUERIES:
        if q.lower() in txt:
            return True, q
    return False, ""


# ------------------- NEWSAPI -------------------
def fetch_newsapi_articles(queries=AI_QUERIES, total_records=50):
    articles = []
    if not NEWS_API_KEY:
        print("NEWS_API_KEY missing")
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    base_url = "https://newsapi.org/v2/everything"

    for q in queries:
        if len(articles) >= total_records:
            break

        params = {
            "q": q,
            "language": "en",
            "from": from_date,
            "to": to_date,
            "sortBy": "publishedAt",
            "pageSize": 20,
            "page": 1,
            "apiKey": NEWS_API_KEY
        }

        try:
            r = requests.get(base_url, params=params, timeout=10)
            data = r.json()

            for a in data.get("articles", []):
                if len(articles) >= total_records:
                    break

                text = f"{a.get('title','')} {a.get('description','')} {a.get('content','')}"
                articles.append({
                    "platform": "newsapi",
                    "timestamp": a.get("publishedAt"),
                    "text": clean_text(text),
                    "url": a.get("url"),
                    "query": q
                })

        except Exception as e:
            print("NewsAPI error:", e)

        time.sleep(0.2)

    return articles[:total_records]


# ------------------- REDDIT -------------------
def fetch_reddit_posts(subreddits=REDDIT_SUBREDDITS, total_records=50):
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        print("Reddit keys missing")
        return []

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    out = []

    for sub in subreddits:
        if len(out) >= total_records:
            break

        try:
            for post in reddit.subreddit(sub).new(limit=150):
                txt = f"{post.title} {post.selftext}"
                rel, matched = is_ai_related(txt)
                if not rel:
                    continue

                out.append({
                    "platform": "reddit",
                    "timestamp": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                    "text": clean_text(txt),
                    "url": f"https://reddit.com{post.permalink}",
                    "query": matched
                })

                if len(out) >= total_records:
                    break

        except Exception as e:
            print("Reddit error:", e)

        time.sleep(0.3)

    return out[:total_records]


# ------------------- GEMINI -------------------
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("models/gemini-flash-latest")


def gemini_batch_sentiment(texts):
    if not texts:
        return []

    # LIMIT LENGTH FOR FREE TIER
    cleaned = []
    for t in texts:
        t = clean_text(t)
        if len(t) > 240:
            t = t[:240] + "..."
        cleaned.append(t)

    # STRICT PROMPT
    prompt = """
You are a strict sentiment classifier.

For each text, classify sentiment as ONLY:
- "positive"
- "negative"
- "neutral"

Return a JSON ARRAY with objects formatted EXACTLY like:
{"id": n, "label": "positive|neutral|negative", "score": -1..1}

Scoring rules (VERY IMPORTANT):
- If label = "positive": score MUST be between +0.40 and +1.00
- If label = "neutral":  score MUST be between -0.20 and +0.20
- If label = "negative": score MUST be between -1.00 and -0.40

Rules:
- Praise, optimism, excitement → positive
- Criticism, fear, anger, decline → negative
- Purely factual or uncertain → neutral
- DO NOT default everything to neutral
- DO NOT include explanations
- Return ONLY JSON
"""

    for i, t in enumerate(cleaned):
        prompt += f"\nTEXT {i}: '''{t}'''"

    # CALL GEMINI
    try:
        resp = model.generate_content(prompt)
        raw = resp.text.strip()

        # Detect quota error
        if "quota" in raw.lower() or "limit" in raw.lower():
            print("⚠ Gemini FREE TIER QUOTA EXCEEDED")
            return [{"id": i, "label": "quota_error", "score": 0} for i in range(len(texts))]

        # Extract JSON
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))

    except Exception as e:
        print("Gemini error:", e)

    # Final fallback — mark clearly (not neutral)
    return [{"id": i, "label": "error", "score": 0} for i in range(len(texts))]


# ------------------- SENTIMENT ANALYSIS -------------------
def analyze_sentiments(df, batch_size=50, progress_cb=None):
    """
    FREE TIER version:
    - Batches through Gemini
    - Stops on quota error
    - Writes a clean, Prophet-safe analyzed.csv
    """

    if df is None or df.empty:
        return pd.DataFrame(columns=["platform", "timestamp", "query", "text", "label", "score", "url"])

    # Ensure expected columns exist
    for col in ["platform", "timestamp", "query", "text", "url"]:
        if col not in df.columns:
            df[col] = ""

    results = []
    texts = df["text"].apply(clean_text).tolist()
    total = len(texts)

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_df = df.iloc[i:i + batch_size]

        sentiments = gemini_batch_sentiment(batch_texts)

        # Quota exhausted → stop completely
        if sentiments and sentiments[0].get("label") == "quota_error":
            print("STOPPING EARLY — GEMINI FREE TIER LIMIT REACHED")
            break

        # Error → skip batch safely
        if not sentiments or sentiments[0]["label"] == "error":
            print(f"Skipping batch {i}-{i + batch_size} (Gemini error)")
            continue

        # Save results
        for j, sent in enumerate(sentiments):
            if j >= len(batch_df):
                break
            row = batch_df.iloc[j]
            results.append({
                "platform": row["platform"],
                "timestamp": row["timestamp"],
                "query": row["query"],
                "text": clean_text(row["text"]),
                "label": sent.get("label", "neutral"),
                "score": sent.get("score", 0),
                "url": row["url"]
            })

        # Progress callback
        if progress_cb:
            pct = int(((i + len(batch_texts)) / total) * 100)
            try:
                progress_cb(pct, f"Analyzed {min(i + batch_size, total)}/{total}")
            except:
                pass

        # FREE TIER limit protection
        time.sleep(1.5)

    out = pd.DataFrame(results)

    # ---------- NEW: CLEANING FOR PROPHET SAFETY ----------
    if not out.empty:
        # Timestamp to datetime
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"])

        # Drop timezone (Prophet requirement)
        try:
            out["timestamp"] = out["timestamp"].dt.tz_convert(None)
        except:
            try:
                out["timestamp"] = out["timestamp"].dt.tz_localize(None)
            except:
                pass

        # Score numeric & clipped
        out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0)
        out["score"] = out["score"].clip(-1, 1)

        # Remove duplicates (same platform + url + timestamp)
        out = out.drop_duplicates(subset=["platform", "url", "timestamp"])

        # Sort by time
        out = out.sort_values("timestamp")

    out.to_csv(ANALYZED_FILE, index=False)
    return out


# ------------------- CHART GENERATION -------------------
def generate_all_charts(df, out_dir, plot_lock):
    os.makedirs(out_dir, exist_ok=True)

    wc_path = os.path.join(out_dir, "wordcloud.png")
    dist_path = os.path.join(out_dir, "distribution.png")
    time_path = os.path.join(out_dir, "timeline.png")

    # --------------------------------------- WORDCLOUD ---------------------------------------
    try:
        text = " ".join(df["text"].dropna().astype(str))
        with plot_lock:
            plt.figure(figsize=(10, 4))
            if text.strip():
                wc = WordCloud(
                    width=900,
                    height=450,
                    background_color="white",
                    stopwords=STOPWORDS
                ).generate(text)
                plt.imshow(wc, interpolation="bilinear")
            else:
                plt.text(0.5, 0.5, "No text available", ha="center")
            plt.axis("off")
            plt.savefig(wc_path, dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print("Wordcloud error:", e)

    # ------------------------------------- PIE CHART -----------------------------------------
    try:
        counts = df["label"].fillna("neutral").value_counts()
        labels = ["positive", "neutral", "negative"]
        sizes = [counts.get(l, 0) for l in labels]

        with plot_lock:
            plt.figure(figsize=(6, 4))
            if sum(sizes) > 0:
                plt.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=["#4caf50", "#9e9e9e", "#f44336"]
                )
            else:
                plt.text(0.5, 0.5, "No data", ha="center")
            plt.savefig(dist_path, dpi=150)
            plt.close()

    except Exception as e:
        print("Pie chart error:", e)

    # ---------------------------------------- TIMELINE ---------------------------------------
    try:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["date"] = df["timestamp"].dt.date

        pivot = df.pivot_table(
            index="date",
            columns="label",
            values="score",
            aggfunc="count",
            fill_value=0
        )

        with plot_lock:
            pivot.plot(kind="area", figsize=(10, 4), stacked=True, alpha=0.8)
            plt.title("Daily Sentiment Timeline")
            plt.tight_layout()
            plt.savefig(time_path, dpi=150)
            plt.close()
    except Exception as e:
        print("Timeline error:", e)

    return {
        "wordcloud": "/plots/wordcloud.png",
        "distribution": "/plots/distribution.png",
        "timeline": "/plots/timeline.png"
    }
