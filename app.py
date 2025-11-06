# app.py — YouTube Sentiment Pro (Render-Ready & Optimized)
import os
import io
import json
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from googleapiclient.discovery import build
import isodate
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import numpy as np
import seaborn as sns

# BERTopic optional
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except Exception:
    BERTOPIC_AVAILABLE = False

# ---------------- CONFIG ----------------
API_KEY = os.getenv("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")
if not API_KEY or API_KEY.startswith("YOUR"):
    raise RuntimeError("Please set a valid YouTube Data API key.")

MAX_COMMENTS_FAST = 100
ENABLE_HEAVY_MODE = False

youtube = build("youtube", "v3", developerKey=API_KEY, static_discovery=False)

os.makedirs("static", exist_ok=True)
os.makedirs("static/images", exist_ok=True)
os.makedirs("output", exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "devkey_js"

# ---------------- Helpers ----------------
def safe_int(x):
    try:
        return int(x)
    except:
        return 0

# ---------- Channel & Videos ----------
def fetch_channel(channel_id):
    try:
        req = youtube.channels().list(part="snippet,contentDetails,statistics", id=channel_id)
        res = req.execute()
        items = res.get("items", [])
        if not items:
            return None, "Channel not found or no public data"
        it = items[0]
        ch = {
            "title": it["snippet"].get("title", ""),
            "description": it["snippet"].get("description", ""),
            "subscribers": safe_int(it["statistics"].get("subscriberCount", 0)),
            "total_views": safe_int(it["statistics"].get("viewCount", 0)),
            "total_videos": safe_int(it["statistics"].get("videoCount", 0)),
            "uploads_playlist": it["contentDetails"]["relatedPlaylists"].get("uploads")
        }
        return ch, None
    except Exception as e:
        return None, f"Channel fetch error: {e}"

def fetch_latest_videos(playlist_id, max_results=10):
    try:
        req = youtube.playlistItems().list(part="snippet,contentDetails", playlistId=playlist_id, maxResults=max_results)
        res = req.execute()
        items = res.get("items", [])
        if not items:
            return pd.DataFrame()
        video_ids = [it["contentDetails"]["videoId"] for it in items if it.get("contentDetails")]
        if not video_ids:
            return pd.DataFrame()
        vreq = youtube.videos().list(part="snippet,statistics,contentDetails", id=",".join(video_ids))
        vres = vreq.execute()
        rows = []
        for it in vres.get("items", []):
            rows.append({
                "video_id": it["id"],
                "title": it["snippet"].get("title",""),
                "publishedAt": it["snippet"].get("publishedAt"),
                "views": safe_int(it["statistics"].get("viewCount",0)),
                "likes": safe_int(it["statistics"].get("likeCount",0)),
                "comments": safe_int(it["statistics"].get("commentCount",0)),
                "duration_sec": int(isodate.parse_duration(it["contentDetails"]["duration"]).total_seconds())
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="publishedAt", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        print("fetch_latest_videos error:", e)
        return pd.DataFrame()

# ---------- Comments ----------
def fetch_comments(video_id, max_comments=300):
    comments = []
    try:
        nextPageToken = None
        while len(comments) < max_comments:
            req = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=nextPageToken,
                textFormat="plainText"
            )
            res = req.execute()
            for it in res.get("items", []):
                sn = it["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": sn.get("textDisplay", ""),
                    "likeCount": int(sn.get("likeCount", 0)),
                    "publishedAt": sn.get("publishedAt")
                })
                if len(comments) >= max_comments:
                    break
            nextPageToken = res.get("nextPageToken")
            if not nextPageToken:
                break
    except Exception as e:
        print("fetch_comments error:", e)
    return comments

def fetch_comments_cached(video_id, max_comments=MAX_COMMENTS_FAST, use_cache=True):
    cache_file = os.path.join("output", f"comments_{video_id}.json")
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if len(data) > max_comments:
                    return data[:max_comments]
                return data
        except Exception:
            pass
    comments = fetch_comments(video_id, max_comments=max_comments)
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(comments, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return comments

# ---------- Sentiment ----------
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    print("⚠️ VADER not available:", e)
    sia = None

def analyze_comments(comments):
    rows = []
    for c in comments:
        text = c.get("text", "")
        likes = int(c.get("likeCount", 0))
        pub = c.get("publishedAt")
        if sia:
            comp = sia.polarity_scores(text)["compound"]
            label = "Positive" if comp >= 0.05 else ("Negative" if comp <= -0.05 else "Neutral")
            s = comp
        else:
            p = TextBlob(text).sentiment.polarity
            label = "Positive" if p > 0 else ("Negative" if p < 0 else "Neutral")
            s = p
        rows.append({
            "comment": text,
            "likeCount": likes,
            "publishedAt": pub,
            "label": label,
            "score": s
        })
    df = pd.DataFrame(rows)
    if "publishedAt" in df.columns:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
    return df

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_id = request.form.get("video_id", "").strip()
        if video_id:
            comments = fetch_comments_cached(video_id)
            df = analyze_comments(comments)
            if df.empty:
                flash("No comments found.")
                return render_template("index.html")
            counts = df["label"].value_counts().to_dict()
            return render_template("index.html", sentiment=counts)
    return render_template("index.html")

# ---------- Download ----------
@app.route("/download_excel")
def download_excel():
    p = "output/data.xlsx"
    if os.path.exists(p):
        return send_file(p, as_attachment=True)
    flash("No file found. Please run analysis first.", "error")
    return redirect(url_for("index"))

# ---------- Run (Render Fix) ----------
if __name__ == "__main__":
    import sys, time
    try:
        os.makedirs("output", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        os.makedirs("static/images", exist_ok=True)

        port = int(os.environ.get("PORT", 10000))
        print(f"✅ Starting Flask app on 0.0.0.0:{port}")
        sys.stdout.flush()

        time.sleep(2)  # small delay so Render detects port properly

        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    except Exception as e:
        print("❌ Error starting app:", e, file=sys.stderr)
        sys.stderr.flush()
        while True:
            print("⚠️ Keeping service alive for Render port scan...")
            sys.stdout.flush()
            time.sleep(5)
