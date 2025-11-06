# app.py — YouTube Comment Sentiment (Render + Flask route fix)
import os
import sys
import threading
import time
from flask import Flask, render_template, request, send_file, redirect, url_for, flash

# ---------------- Early Flask Binding ----------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "devkey_js"

def start_flask():
    """Start Flask early so Render detects open port before heavy imports load."""
    port = int(os.environ.get("PORT", 10000))
    print(f"✅ Early Flask startup on 0.0.0.0:{port}")
    sys.stdout.flush()
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# Start Flask early in a background thread
threading.Thread(target=start_flask, daemon=True).start()
time.sleep(3)  # Give Render time to detect open port

# ---------------- Heavy Imports (load after Flask binding) ----------------
import io
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from googleapiclient.discovery import build
import isodate
from datetime import datetime, timedelta, timezone
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import seaborn as sns
import numpy as np

# Optional BERTopic imports
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except Exception:
    BERTOPIC_AVAILABLE = False

# ---------------- CONFIG ----------------
API_KEY = os.getenv("YOUTUBE_API_KEY", "")
if not API_KEY:
    print("⚠️ Warning: YOUTUBE_API_KEY not set. Continuing for demo/testing mode.")
else:
    print("✅ YouTube API Key loaded.")

MAX_COMMENTS_FAST = 100
ENABLE_HEAVY_MODE = False

youtube = None
if API_KEY:
    youtube = build("youtube", "v3", developerKey=API_KEY, static_discovery=False)

os.makedirs("static", exist_ok=True)
os.makedirs("static/images", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ---------------- Helpers ----------------
def safe_int(x):
    try:
        return int(x)
    except:
        return 0

# ---------- Fetch Channel ----------
def fetch_channel(channel_id):
    if not youtube:
        return None, "YouTube API not initialized (no API key)."
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
            "uploads_playlist": it["contentDetails"]["relatedPlaylists"].get("uploads"),
        }
        return ch, None
    except Exception as e:
        return None, f"Channel fetch error: {e}"

# ---------- Fetch Videos ----------
def fetch_latest_videos(playlist_id, max_results=10):
    if not youtube:
        return pd.DataFrame()
    try:
        req = youtube.playlistItems().list(part="snippet,contentDetails", playlistId=playlist_id, maxResults=max_results)
        res = req.execute()
        items = res.get("items", [])
        if not items:
            return pd.DataFrame()
        video_ids = [it["contentDetails"]["videoId"] for it in items if it.get("contentDetails")]
        vreq = youtube.videos().list(part="snippet,statistics,contentDetails", id=",".join(video_ids))
        vres = vreq.execute()
        rows = []
        for it in vres.get("items", []):
            rows.append({
                "video_id": it["id"],
                "title": it["snippet"].get("title", ""),
                "publishedAt": it["snippet"].get("publishedAt"),
                "views": safe_int(it["statistics"].get("viewCount", 0)),
                "likes": safe_int(it["statistics"].get("likeCount", 0)),
                "comments": safe_int(it["statistics"].get("commentCount", 0)),
                "duration_sec": int(isodate.parse_duration(it["contentDetails"]["duration"]).total_seconds())
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="publishedAt", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        print("fetch_latest_videos error:", e)
        return pd.DataFrame()

# ---------- Fetch Comments ----------
def fetch_comments(video_id, max_comments=200):
    if not youtube:
        return []
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
                    "publishedAt": sn.get("publishedAt"),
                })
                if len(comments) >= max_comments:
                    break
            nextPageToken = res.get("nextPageToken")
            if not nextPageToken:
                break
    except Exception as e:
        print("fetch_comments error:", e)
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
        rows.append({"comment": text, "likeCount": likes, "publishedAt": pub, "label": label, "score": s})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
    return df

# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def index():
    context = {"channel_info": None, "videos": None, "sentiment": None,
               "top_comments": None, "topic_list": None}
    if request.method == "POST":
        channel_id = request.form.get("channel_id", "").strip()
        video_id = request.form.get("video_id", "").strip()
        manual_vid = request.form.get("manual_video_id", "").strip()

        if channel_id:
            ch, err = fetch_channel(channel_id)
            if err:
                flash(err, "error")
                return render_template("index.html", **context)
            df_v = fetch_latest_videos(ch["uploads_playlist"], max_results=10)
            context.update({"channel_info": ch, "videos": df_v.to_dict("records")})
            return render_template("index.html", **context)

        vid = manual_vid or video_id
        if vid:
            comments = fetch_comments(vid)
            if not comments:
                flash("No comments found for this video.", "error")
                return render_template("index.html", **context)
            df = analyze_comments(comments)
            summary = df["label"].value_counts().to_dict()
            context.update({"sentiment": summary})
            return render_template("index.html", **context)
    return render_template("index.html", **context)

@app.route("/download_excel")
def download_excel():
    p = "output/data.xlsx"
    if os.path.exists(p):
        return send_file(p, as_attachment=True)
    flash("No Excel file found. Please run analysis first.", "error")
    return redirect(url_for("index"))

@app.route("/download_pdf")
def download_pdf():
    p = "output/report.pdf"
    if os.path.exists(p):
        return send_file(p, as_attachment=True)
    flash("No PDF found. Please run analysis first.", "error")
    return redirect(url_for("index"))

# ---------- KEEP ALIVE LOOP ----------
while True:
    print("🟢 Flask app running — Render port bound successfully.")
    sys.stdout.flush()
    time.sleep(30)
