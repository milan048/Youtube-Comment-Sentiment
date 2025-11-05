# app.py — YouTube Sentiment Pro (same features, optimized for speed)
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

# BERTopic optional (kept lazy if available)
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except Exception:
    BERTOPIC_AVAILABLE = False

# ---------------- CONFIG ----------------
# Put your API Key here (or set YOUTUBE_API_KEY environment variable)
API_KEY = os.getenv("YOUTUBE_API_KEY", "AIzaSyCGUxhvjFogC5qImxqpVlXuNzM01BUjS4Q")

if not API_KEY or API_KEY.startswith("PUT"):
    raise RuntimeError("Please set a valid YouTube Data API key in API_KEY or YOUTUBE_API_KEY env var.")
# Performance / dev tuning
MAX_COMMENTS_FAST = 100          # fast default during development
ENABLE_HEAVY_MODE = False        # set True to allow heavy operations by default

# Build YouTube client (static_discovery for compat)
youtube = build("youtube", "v3", developerKey=API_KEY, static_discovery=False)

# create folders
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

# ---------- Comments (extended) ----------
def fetch_comments(video_id, max_comments=300):
    """
    returns list of dicts: {'text','likeCount','publishedAt'}
    (Raw API fetch — use fetch_comments_cached in app routes to speed dev)
    """
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

# ---------- caching for speed (dev) ----------
def fetch_comments_cached(video_id, max_comments=MAX_COMMENTS_FAST, use_cache=True):
    cache_file = os.path.join("output", f"comments_{video_id}.json")
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # if cached more than needed, trim
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

# ---------- Sentiment engines ----------
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Try to init VADER quickly
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    hf_pipe = None
    print("✅ VADER loaded (fast mode). HF lazy.")
except Exception as e:
    print("⚠️ VADER not available:", e)
    sia = None
    hf_pipe = None

def analyze_comments(comments, use_hf=False):
    """
    returns DataFrame columns: comment, likeCount, publishedAt, label, score
    use_hf: bool (if True, lazy-load HF and use batched inference)
    """
    global hf_pipe, sia
    rows = []

    # lazy-load HF
    if use_hf and hf_pipe is None:
        try:
            from transformers import pipeline
            hf_pipe = pipeline("sentiment-analysis", model=HF_MODEL)
            print("✅ HF pipeline loaded")
        except Exception as e:
            print("HF load failed:", e)
            hf_pipe = None

    BATCH = 32
    texts = [c.get("text","") for c in comments]

    if use_hf and hf_pipe:
        outs = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            attempt = 0
            out = None

            while attempt < 3:
                try:
                    out = hf_pipe(batch, truncation=True)
                    break
                except Exception as e:
                    attempt += 1
                    print(f"HF Error attempt {attempt}/3:", e)
                    import time; time.sleep(2)

            if out is None:
                print("⚠ HF fallback for this batch")
                out = [{"label":"NEUTRAL","score":0.0} for _ in batch]

            outs.extend(out)

        for idx, c in enumerate(comments):
            r = outs[idx]
            lab = r.get("label","").lower()
            score = float(r.get("score",0.0))
            if "pos" in lab:
                label = "Positive"
                s = score
            elif "neg" in lab:
                label = "Negative"
                s = -score
            else:
                label = "Neutral"
                s = 0.0

            rows.append({
                "comment": c.get("text",""),
                "likeCount": int(c.get("likeCount",0)),
                "publishedAt": c.get("publishedAt"),
                "label": label,
                "score": s
            })

    else:
        for c in comments:
            text = c.get("text","")
            likes = int(c.get("likeCount",0))
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


# ---------- Topic modeling (fallback) ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

if BERTOPIC_AVAILABLE:
    try:
        SENT_EMBED = "all-MiniLM-L6-v2"
        sentence_model = SentenceTransformer(SENT_EMBED)
        bert_model = BERTopic(embedding_model=sentence_model, verbose=False)
    except Exception as e:
        print("BERTopic init error:", e)
        bert_model = None
else:
    bert_model = None

def quick_topics_tfidf(comments, k=6):
    texts = [str(t) for t in comments if t and str(t).strip()]
    if not texts:
        return []
    try:
        vec = TfidfVectorizer(max_features=1500, stop_words='english')
        X = vec.fit_transform(texts)
        km = KMeans(n_clusters=min(k, max(2, len(texts)//2)), random_state=42).fit(X)
        terms = vec.get_feature_names_out()
        topics = []
        for i in range(min(k, km.n_clusters)):
            center = km.cluster_centers_[i]
            top_idx = center.argsort()[-8:][::-1]
            words = ", ".join(terms[j] for j in top_idx)
            topics.append({"topic": f"Topic {i}", "freq": int((km.labels_==i).sum()), "words": words})
        return topics
    except Exception as e:
        print("quick_topics error:", e)
        return []

def run_topic_modeling(comments_list, top_n=8):
    texts = [str(c) for c in comments_list if c and str(c).strip()]
    if not texts:
        return []
    if bert_model is None:
        return quick_topics_tfidf(texts, k=top_n)
    try:
        topics, probs = bert_model.fit_transform(texts)
        info = bert_model.get_topic_info()
        info = info[info.Topic != -1].head(top_n)
        results = []
        for _, row in info.iterrows():
            t = int(row.Topic)
            words = bert_model.get_topic(t)
            words_clean = ", ".join([w for w,_ in words[:8]])
            results.append({"topic": row.Name if "Name" in row and row.Name else f"Topic {t}", "freq": int(row.Count), "words": words_clean})
        return results
    except Exception as e:
        print("BERTopic run error:", e)
        return quick_topics_tfidf(texts, k=top_n)

# ---------- Time-series & correlations ----------
def compute_time_series_metrics(df_comments):
    if df_comments is None or df_comments.empty:
        return None, None, None
    df = df_comments.copy()
    if "publishedAt" in df.columns:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
    df = df.dropna(subset=["publishedAt"]).copy()
    if df.empty:
        return None, None, None
    df["date"] = df["publishedAt"].dt.tz_convert("UTC").dt.date
    # ensure 'score' exists
    if "score" not in df.columns:
        if "sentiment_score" in df.columns:
            df["score"] = df["sentiment_score"]
        elif "polarity" in df.columns:
            df["score"] = df["polarity"]
        else:
            df["score"] = 0.0
    daily_sent = df.groupby("date")["score"].mean()
    daily_vol = df.groupby("date").size()
    merged = pd.DataFrame({"daily_sentiment": daily_sent, "daily_volume": daily_vol}).fillna(0)
    return daily_sent, daily_vol, merged

def save_time_series_plots(daily_sent, daily_vol):
    # remove old files if no data
    if daily_sent is None or daily_sent.empty:
        for p in ("static/sentiment_over_time.png","static/volume_vs_sentiment.png","static/sentiment_vs_likes.png"):
            if os.path.exists(p): os.remove(p)
        return
    plt.figure(figsize=(8,3))
    plt.plot(daily_sent.index, daily_sent.values, marker='o', color="#ff0000")
    plt.title("Avg Sentiment Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/sentiment_over_time.png", dpi=150)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(9,3))
    ax2 = ax1.twinx()
    ax1.bar(daily_vol.index, daily_vol.values, alpha=0.6, label="Comment Volume", color="#777")
    ax2.plot(daily_sent.index, daily_sent.values, color="#ff0000", marker='o', label="Avg Sentiment")
    ax1.set_xticklabels([str(d) for d in daily_vol.index], rotation=45)
    ax1.set_title("Comment Volume vs Avg Sentiment")
    fig.tight_layout()
    fig.savefig("static/volume_vs_sentiment.png", dpi=150)
    plt.close()

def save_sentiment_vs_likes(df_comments):
    if df_comments is None or df_comments.empty:
        if os.path.exists("static/sentiment_vs_likes.png"): os.remove("static/sentiment_vs_likes.png")
        return
    plt.figure(figsize=(6,4))
    sns.scatterplot(x="likeCount", y="score", data=df_comments, alpha=0.6)
    plt.title("Sentiment vs Comment Likes")
    plt.xlabel("Comment Likes")
    plt.ylabel("Sentiment Score")
    plt.tight_layout()
    plt.savefig("static/sentiment_vs_likes.png", dpi=150)
    plt.close()

# ---------- Views/likes/wordcloud ----------
def save_views_likes_last30_graphs(df):
    for p in ("static/views.png","static/likes.png","static/last30.png"):
        if os.path.exists(p): os.remove(p)
    if df is None or df.empty:
        return
    df2 = df.copy()
    df2["short_title"] = df2["title"].apply(lambda t: t if len(t)<=40 else t[:37]+"...")
    plt.figure(figsize=(10,4))
    plt.bar(df2["short_title"], df2["views"], color="#ff0000")
    plt.xticks(rotation=45, ha='right')
    plt.title("Top 10 Videos — Views")
    plt.tight_layout()
    plt.savefig("static/views.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10,4))
    plt.bar(df2["short_title"], df2["likes"], color="#ff4444")
    plt.xticks(rotation=45, ha='right')
    plt.title("Top 10 Videos — Likes")
    plt.tight_layout()
    plt.savefig("static/likes.png", dpi=150)
    plt.close()

    df2["published_dt"] = pd.to_datetime(df2["publishedAt"], errors="coerce", utc=True)
    recent_v = df2[df2["published_dt"] >= (datetime.now(timezone.utc)-timedelta(days=30))]
    if recent_v.empty:
        if os.path.exists("static/last30.png"): os.remove("static/last30.png")
    else:
        plt.figure(figsize=(8,3))
        plt.bar(recent_v["short_title"].apply(lambda t: t if len(t)<=30 else t[:27]+"..."), recent_v["views"], color="#9c27b0")
        plt.xticks(rotation=45, ha='right')
        plt.title("Views on Videos Published in Last 30 Days")
        plt.tight_layout()
        plt.savefig("static/last30.png", dpi=150)
        plt.close()

def save_sentiment_images(df_comments):
    for p in ("static/sentiment_pie.png","static/polarity_line.png","static/wordcloud.png"):
        if os.path.exists(p) and (df_comments is None or df_comments.empty):
            os.remove(p)
    if df_comments is None or df_comments.empty:
        return
    counts = df_comments["label"].value_counts().reindex(["Positive","Neutral","Negative"], fill_value=0)
    plt.figure(figsize=(4,4))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=["#ff0000","#9e9e9e","#ff5252"])
    plt.title("Sentiment Distribution")
    plt.tight_layout()
    plt.savefig("static/sentiment_pie.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6,3))
    plt.plot(df_comments.index, df_comments["score"], marker='o', linestyle='-')
    plt.title("Comment Polarity (intensity)")
    plt.xlabel("Comment index")
    plt.ylabel("Polarity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("static/polarity_line.png", dpi=150)
    plt.close()

    text = " ".join(df_comments["comment"].astype(str).tolist())
    if text.strip():
        wc = WordCloud(width=900, height=400, background_color="black", colormap="viridis").generate(text)
        wc.to_file("static/wordcloud.png")

def top_n_comments(df_comments, n=5):
    if df_comments is None or df_comments.empty:
        return {"positive":[], "negative":[], "neutral":[]}
    pos = df_comments[df_comments["label"]=="Positive"].sort_values("score", ascending=False).head(n)["comment"].tolist()
    neg = df_comments[df_comments["label"]=="Negative"].sort_values("score", ascending=True).head(n)["comment"].tolist()
    neu_df = df_comments[df_comments["label"]=="Neutral"].copy()
    if not neu_df.empty:
        neu_df["abs_pol"] = neu_df["score"].abs()
        neu = neu_df.sort_values("abs_pol").head(n)["comment"].tolist()
    else:
        neu = []
    return {"positive": pos, "negative": neg, "neutral": neu}

# Excel helpers
def tz_naive_for_excel(df):
    if df is None:
        return df
    df2 = df.copy()
    try:
        for col in df2.columns:
            if pd.api.types.is_datetime64_any_dtype(df2[col]):
                # drop tz info
                try:
                    df2[col] = df2[col].dt.tz_convert(None)
                except Exception:
                    try:
                        df2[col] = df2[col].dt.tz_localize(None)
                    except Exception:
                        pass
    except Exception:
        pass
    return df2

def save_excel(videos_df=None, comments_df=None, path="output/data.xlsx"):
    videos_safe = tz_naive_for_excel(videos_df) if videos_df is not None else None
    comments_safe = tz_naive_for_excel(comments_df) if comments_df is not None else None
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if videos_safe is not None and not videos_safe.empty:
            videos_safe.to_excel(writer, sheet_name="Videos", index=False)
        if comments_safe is not None and not comments_safe.empty:
            comments_safe.to_excel(writer, sheet_name="Comments", index=False)
    return path

# PDF generation (same style)
def generate_pdf(channel_info, videos_df, stats30, comments_df, topc, filename="output/report.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    margin = 36
    y = height - margin
    logo = "static/images/logo.jpg"
    if os.path.exists(logo):
        try:
            c.drawImage(ImageReader(logo), margin, y-40, width=80, height=40)
        except:
            pass
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin+100, y-20, "YouTube Analytics & Sentiment Report")
    y -= 50
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Channel: {channel_info.get('title','-')}")
    y -= 14
    c.drawString(margin, y, f"Subscribers: {channel_info.get('subscribers','-')}  Total Views: {channel_info.get('total_views','-')}  Total Videos: {channel_info.get('total_videos','-')}")
    y -= 18
    c.drawString(margin, y, f"Last 30d (approx): Views: {stats30.get('views_30d',0)}  Likes: {stats30.get('likes_30d',0)}  Videos: {stats30.get('videos_in_30d',0)}")
    y -= 24
    if os.path.exists("static/views.png"):
        c.drawImage(ImageReader("static/views.png"), margin, y-200, width=260, height=120)
    if os.path.exists("static/likes.png"):
        c.drawImage(ImageReader("static/likes.png"), margin+280, y-200, width=260, height=120)
    y -= 220
    if os.path.exists("static/sentiment_pie.png"):
        c.drawImage(ImageReader("static/sentiment_pie.png"), margin, y-140, width=200, height=120)
    if os.path.exists("static/wordcloud.png"):
        c.drawImage(ImageReader("static/wordcloud.png"), margin+220, y-140, width=300, height=120)
    y -= 160
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Top Positive Comments:")
    y -= 14
    c.setFont("Helvetica", 9)
    for t in topc.get("positive", []):
        c.drawString(margin+6, y, "- " + (t[:120] + ("..." if len(t) > 120 else "")))
        y -= 12
        if y < 80:
            c.showPage()
            y = height - margin
    y -= 6
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Top Negative Comments:")
    y -= 14
    c.setFont("Helvetica", 9)
    for t in topc.get("negative", []):
        c.drawString(margin+6, y, "- " + (t[:120] + ("..." if len(t) > 120 else "")))
        y -= 12
        if y < 80:
            c.showPage()
            y = height - margin
    y -= 6
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Top Neutral Comments:")
    y -= 14
    c.setFont("Helvetica", 9)
    for t in topc.get("neutral", []):
        c.drawString(margin+6, y, "- " + (t[:120] + ("..." if len(t) > 120 else "")))
        y -= 12
        if y < 80:
            c.showPage()
            y = height - margin
    c.save()
    return filename

def compute_last_30d(df):
    try:
        if df is None or df.empty:
            return {"views_30d":0, "likes_30d":0, "videos_in_30d":0}

        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=30)

        # Ensure the column is converted to datetime with timezone
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], utc=True, errors='coerce')

        last30 = df[df['publishedAt'] >= cutoff]

        view_series = last30.get('viewCount', pd.Series([0]*len(last30)))
        like_series = last30.get('likeCount', pd.Series([0]*len(last30)))

        return {
            "views_30d": int(pd.to_numeric(view_series, errors='coerce').fillna(0).sum()),
            "likes_30d": int(pd.to_numeric(like_series, errors='coerce').fillna(0).sum()),
            "videos_in_30d": len(last30)
        }

    except Exception as e:
        print("compute_last_30d error:", e)
        return {"views_30d":0, "likes_30d":0, "videos_in_30d":0}

# ========== Routes (same endpoints as before) ==========
@app.route("/", methods=["GET","POST"])
def index():
    context = {"channel_info": None, "videos": None, "stats30": None, "sentiment": None, "top_comments": None, "topic_list": None}
    if request.method == "POST":
        channel_id = request.form.get("channel_id", "").strip()
        select_vid = request.form.get("video_id", "").strip()        # dropdown/video-select
        manual_vid = request.form.get("manual_video_id", "").strip() # manual input
        # Channel analyze
        if channel_id:
            ch, err = fetch_channel(channel_id)
            if err:
                flash(err, "error")
                return render_template("index.html", **context)
            df_v = fetch_latest_videos(ch["uploads_playlist"], max_results=10)
            # compute last 30d stats (using publishedAt parsing inside function)
            stats30 = compute_last_30d(df_v) if df_v is not None else {"views_30d":0,"likes_30d":0,"videos_in_30d":0}
            save_views_likes_last30_graphs(df_v)
            try:
                save_excel(videos_df=df_v, comments_df=None, path="output/data.xlsx")
            except Exception as e:
                print("excel save error:", e)
            context.update({"channel_info": ch, "videos": df_v.to_dict("records") if df_v is not None else None, "stats30": stats30})
            return render_template("index.html", **context)

        # Video analyze (dropdown or manual)
        vid = manual_vid or select_vid
        if vid:
            # use cached fetch for speed
            raw_comments = fetch_comments_cached(vid, max_comments=MAX_COMMENTS_FAST, use_cache=True)

            # ✅ Hybrid mode: limit for speed if too many comments
            if len(raw_comments) > 400:
                raw_comments = raw_comments[:400]

            if not raw_comments:
                flash("No comments found for this video (maybe comments disabled or private).", "error")
                return render_template("index.html", **context)

            # ✅ Read HF option from UI checkbox
            use_hf_flag = request.form.get("use_hf", "off") == "on"

            # ✅ Run sentiment
            dfc = analyze_comments(raw_comments, use_hf=use_hf_flag)
            save_sentiment_images(dfc)
          # ⚡ Skip topic modeling on large comment sets to save time
            if len(raw_comments) > 400:
                 topic_list = []
            else:
                 topic_list = run_topic_modeling([c["text"] for c in raw_comments], top_n=6)

            if topic_list:
                topics = [t["topic"] for t in topic_list]
                freqs = [t["freq"] for t in topic_list]
                plt.figure(figsize=(6,3))
                plt.barh(topics[::-1], freqs[::-1], color="#ff0000")
                plt.title("Top Topics")
                plt.tight_layout()
                plt.savefig("static/topic_bar.png", dpi=150)
                plt.close()
            else:
                if os.path.exists("static/topic_bar.png"): os.remove("static/topic_bar.png")

            daily_sent, daily_vol, merged = compute_time_series_metrics(dfc)
            save_time_series_plots(daily_sent, daily_vol)
            save_sentiment_vs_likes(dfc)
            tops = top_n_comments(dfc, n=5)
            try:
                save_excel(videos_df=None, comments_df=dfc, path="output/data.xlsx")
            except Exception as e:
                print("excel save error advanced:", e)
            # keep 'sentiment' summary compatible with old templates
            summary = dfc["label"].value_counts().to_dict() if "label" in dfc.columns else {}
            context.update({"sentiment": summary, "top_comments": tops, "topic_list": topic_list})
            return render_template("index.html", **context)

    return render_template("index.html", **context)

@app.route("/download_excel")
def download_excel():
    p = "output/data.xlsx"
    if os.path.exists(p):
        return send_file(p, as_attachment=True)
    flash("No data file found. Please run analysis first.", "error")
    return redirect(url_for("index"))

@app.route("/download_pdf")
def download_pdf():
    ch = {"title":"(report)", "subscribers":"-", "total_views":"-", "total_videos":"-"}
    dfv = None
    dfc = None
    if os.path.exists("output/data.xlsx"):
        try:
            dfv = pd.read_excel("output/data.xlsx", sheet_name="Videos")
        except Exception:
            dfv = None
        try:
            dfc = pd.read_excel("output/data.xlsx", sheet_name="Comments")
        except Exception:
            dfc = None
    topc = {"positive":[],"negative":[],"neutral":[]}
    if dfc is not None and not dfc.empty:
        if "score" not in dfc.columns:
            dfc["score"] = dfc.get("sentiment_score", dfc.get("polarity", 0.0))
        if "score" in dfc.columns:
            dfc["sentiment"] = dfc["score"].apply(lambda p: "Positive" if p>0 else ("Negative" if p<0 else "Neutral"))
        topc = top_n_comments(dfc, n=5)
    stats30 = {"views_30d":0,"likes_30d":0,"videos_in_30d":0}
    pdf_path = generate_pdf(ch, dfv, stats30, dfc, topc, filename="output/report.pdf")
    return send_file(pdf_path, as_attachment=True)

# --------------- Run ----------------
if __name__ == "__main__":
    # quick check: ensure output directories exist
    os.makedirs("output", exist_ok=True)
    app.run(debug=True)
