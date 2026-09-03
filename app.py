import ast
import re
import os
import html
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_OK = True
except ImportError:
    SEMANTIC_OK = False

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CineMind — AI Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CINEMATIC UI
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Bebas+Neue&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- Background ---------- */
.stApp {
    background:
      radial-gradient(circle at 85% 0%, rgba(229,9,20,.18), transparent 32%),
      radial-gradient(circle at 5% 15%, rgba(120,80,255,.10), transparent 28%),
      radial-gradient(circle at 50% 100%, rgba(0,180,255,.06), transparent 40%),
      #060606;
    color: #f5f5f5;
}

.block-container {
    max-width: 1480px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
    height: 0rem;
}

header[data-testid="stHeader"] * {
    display: none;
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #e50914; }

/* ---------- Top brand bar ---------- */
.brand-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 4px 18px;
}
.brand-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.1rem;
    line-height: 1.4;
    padding-top: 6px;
    padding-bottom: 4px;
    letter-spacing: 2px;
    background: linear-gradient(90deg, #ff3b3b, #ff8a5c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}
.brand-tag {
    color: #888;
    font-size: .82rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
}

/* ---------- Hero ---------- */
.hero {
    position: relative;
    min-height: 420px;
    border-radius: 26px;
    padding: 70px 65px;
    margin-bottom: 34px;
    overflow: hidden;
    background:
      linear-gradient(90deg, rgba(0,0,0,.97) 0%, rgba(0,0,0,.82) 42%, rgba(0,0,0,.15) 100%),
      radial-gradient(circle at 78% 25%, rgba(229,9,20,.40), transparent 40%),
      linear-gradient(135deg, #191919, #070707);
    border: 1px solid rgba(255,255,255,.08);
    box-shadow: 0 30px 90px rgba(0,0,0,.55);
}

.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: radial-gradient(rgba(255,255,255,.035) 1px, transparent 1px);
    background-size: 26px 26px;
    opacity: .5;
    pointer-events: none;
}

.hero-kicker {
    display: inline-block;
    color: #ff3b3b;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-size: .82rem;
    padding: 5px 14px;
    border: 1px solid rgba(229,9,20,.4);
    border-radius: 999px;
    background: rgba(229,9,20,.08);
    margin-bottom: 14px;
}

.hero h1 {
    font-size: 4.4rem;
    line-height: 1.0;
    margin: 14px 0;
    font-weight: 900;
    letter-spacing: -1px;
    text-shadow: 0 10px 40px rgba(0,0,0,.6);
}

.hero p {
    max-width: 660px;
    color: #cfcfcf;
    font-size: 1.1rem;
    line-height: 1.65;
    position: relative;
}

.hero-stats {
    margin-top: 18px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    position: relative;
}

.stat-pill {
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.12);
    backdrop-filter: blur(6px);
    padding: 8px 16px;
    border-radius: 999px;
    font-size: .9rem;
    font-weight: 600;
    color: #eee;
}

/* ---------- Section titles ---------- */
.section-title {
    font-size: 1.7rem;
    font-weight: 800;
    margin: 30px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,.15), transparent);
    margin-left: 8px;
}

/* ---------- Movie cards ---------- */
.movie-card {
    background: linear-gradient(160deg, #141414, #0c0c0c);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px;
    padding: 10px;
    transition: all .25s cubic-bezier(.2,.8,.2,1);
    height: 100%;
    position: relative;
}

.movie-card:hover {
    transform: translateY(-7px) scale(1.015);
    border-color: rgba(229,9,20,.6);
    box-shadow: 0 18px 45px rgba(0,0,0,.55), 0 0 0 1px rgba(229,9,20,.25);
}

.poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    border-radius: 12px;
    background: linear-gradient(160deg,#232323,#151515);
}

.poster-placeholder {
    aspect-ratio: 2/3;
    border-radius: 12px;
    background: linear-gradient(160deg,#232323,#141414);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3.2rem;
    color: #444;
    border: 1px dashed rgba(255,255,255,.12);
}

.movie-title {
    font-size: 1.02rem;
    font-weight: 700;
    margin-top: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.meta {
    color: #9c9c9c;
    font-size: .82rem;
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.rating-chip {
    background: rgba(255,193,7,.14);
    color: #ffc107;
    border: 1px solid rgba(255,193,7,.25);
    border-radius: 8px;
    padding: 1px 7px;
    font-weight: 700;
    font-size: .78rem;
}

.badge {
    display: inline-block;
    background: rgba(229,9,20,.14);
    color: #ff6970;
    border: 1px solid rgba(229,9,20,.28);
    border-radius: 999px;
    padding: 4px 10px;
    margin: 6px 4px 0 0;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .2px;
}

/* ---------- Detail box ---------- */
.detail-box {
    background: linear-gradient(150deg,#181818,#0b0b0b);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 22px;
    padding: 28px;
    margin: 14px 0 26px;
    box-shadow: 0 20px 60px rgba(0,0,0,.5);
}

/* ---------- Search ---------- */
.search-wrap {
    background: rgba(18,18,18,.9);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 18px;
    padding: 14px 16px;
    box-shadow: 0 12px 30px rgba(0,0,0,.35);
}

div.stTextInput > div > div > input {
    background: #101010 !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 12px !important;
    color: #f5f5f5 !important;
    padding: 12px 16px !important;
    font-size: 1rem !important;
}
div.stTextInput > div > div > input:focus {
    border-color: #e50914 !important;
    box-shadow: 0 0 0 3px rgba(229,9,20,.18) !important;
}

/* ---------- Buttons ---------- */
div.stButton > button {
    border-radius: 11px;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,.14);
    transition: .2s;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#e50914,#ff3b3b);
    border: none;
    box-shadow: 0 8px 22px rgba(229,9,20,.35);
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(90deg,#f40612,#ff5a5a);
    box-shadow: 0 10px 28px rgba(229,9,20,.5);
    transform: translateY(-1px);
}

div.stButton > button[kind="secondary"]:hover {
    border-color: rgba(229,9,20,.5);
    color: #ff6970;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #0d0d0d;
    border-radius: 14px;
    padding: 6px;
    border: 1px solid rgba(255,255,255,.06);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 18px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg,#e50914,#ff3b3b) !important;
    color: #fff !important;
}

/* ---------- Sliders / selects ---------- */
div[data-baseweb="select"] > div {
    background: #101010 !important;
    border-color: rgba(255,255,255,.12) !important;
    border-radius: 10px !important;
}

/* ---------- Footer ---------- */
.developer-footer {
    margin-top: 70px;
    padding: 40px 20px 20px;
    text-align: center;
    border-top: 1px solid rgba(255,255,255,.10);
    background: linear-gradient(180deg, transparent, rgba(229,9,20,.03));
}

.footer-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 2px;
    color: #ffffff;
}

.footer-subtitle {
    font-size: 1rem;
    color: #aaa;
    margin-top: 4px;
}

.footer-developed {
    font-size: .85rem;
    color: #777;
    margin-top: 26px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.footer-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 6px;
}

.footer-company {
    color: #e50914;
    font-weight: 600;
    margin-top: 4px;
}

.footer-tech {
    color: #666;
    font-size: .78rem;
    margin-top: 16px;
    letter-spacing: .5px;
}
div[data-testid="stToast"] {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    right: auto !important;
    bottom: auto !important;
    transform: translate(-50%, -50%) !important;
    font-size: 1.3rem !important;
    padding: 24px 32px !important;
    min-width: 380px !important;
    background: #1a1a1a !important;
    border: 1px solid rgba(229,9,20,.5) !important;
    box-shadow: 0 20px 60px rgba(0,0,0,.6) !important;
    z-index: 9999 !important;
}

div[data-testid="stToast"] p {
    font-size: 1.3rem !important;
    font-weight: 600 !important;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def parse_names(value, limit=None):
    if pd.isna(value):
        return []
    try:
        data = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    result = []
    for item in data:
        if isinstance(item, dict) and item.get("name"):
            result.append(str(item["name"]))
            if limit and len(result) >= limit:
                break
    return result

def parse_director(value):
    if pd.isna(value):
        return []
    try:
        data = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [
        str(x["name"]) for x in data
        if isinstance(x, dict) and x.get("job") == "Director" and x.get("name")
    ][:1]

def clean_tokens(items):
    out = []
    for x in items:
        token = re.sub(r"[^a-zA-Z0-9]+", "", str(x).lower())
        if token:
            out.append(token)
    return out

def normalize(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or values.max() == values.min():
        return np.full(len(values), .5)
    return (values - values.min()) / (values.max() - values.min())

def movie_year(row):
    date = str(row.get("release_date", ""))
    return date[:4] if len(date) >= 4 and date[:4].isdigit() else "—"

def get_tmdb_api_key():
    """
    Get TMDB API key from Streamlit secrets or environment variables.
    """
    try:
        api_key = st.secrets.get("TMDB_API_KEY")
        if api_key:
            return api_key
    except Exception:
        pass
    return os.getenv("TMDB_API_KEY")


@st.cache_data(ttl=86400, show_spinner=False)
def get_tmdb_poster(movie_id):
    """
    Fetch poster directly from TMDB using movie ID.
    Cached for 24 hours so we don't repeatedly call the API.
    """
    api_key = get_tmdb_api_key()

    if not api_key or pd.isna(movie_id):
        return None

    try:
        import requests

        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(movie_id)}",
            params={"api_key": api_key},
            timeout=10
        )

        if response.status_code != 200:
            return None

        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except Exception as e:
        print("TMDB poster error:", e)

    return None


def poster_url(movie_id, row=None):
    """
    Get movie poster.

    Priority:
    1. poster_path from CSV, if available
    2. TMDB API using movie ID
    3. None
    """
    if row is not None and "poster_path" in row.index:
        poster_path = row["poster_path"]
        if pd.notna(poster_path):
            poster_path = str(poster_path).strip()
            if poster_path and poster_path.lower() not in ["nan", "none", "null", "na", ""]:
                if not poster_path.startswith("/"):
                    poster_path = "/" + poster_path
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

    return get_tmdb_poster(movie_id)


def trailer_url(movie_id):
    api_key = get_tmdb_api_key()
    if not api_key:
        return None
    try:
        import requests
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(movie_id)}/videos",
            params={"api_key": api_key},
            timeout=5
        )
        if r.ok:
            videos = r.json().get("results", [])
            for v in videos:
                if v.get("site") == "YouTube" and v.get("type") in ["Trailer", "Teaser"]:
                    return "https://www.youtube.com/watch?v=" + v["key"]
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def load_data(movies_path, credits_path):
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path).rename(columns={"movie_id": "id"})
    cols = [c for c in ["id", "cast", "crew"] if c in credits.columns]
    movies = movies.merge(credits[cols], on="id", how="left")

    movies["title"] = movies["title"].fillna("Unknown")
    movies["overview"] = movies["overview"].fillna("")
    movies["genres_list"] = movies["genres"].apply(parse_names)
    movies["keywords_list"] = movies["keywords"].apply(parse_names)
    movies["cast_list"] = movies["cast"].apply(lambda x: parse_names(x, 5))
    movies["director_list"] = movies["crew"].apply(parse_director)

    for c in ["genres_list", "keywords_list", "cast_list", "director_list"]:
        movies[c.replace("_list", "_tokens")] = movies[c].apply(clean_tokens)

    movies["tags"] = (
        movies["overview"].str.lower() + " " +
        movies["genres_tokens"].apply(lambda x: " ".join(x) + " ") +
        movies["genres_tokens"].apply(lambda x: " ".join(x) + " ") +
        movies["keywords_tokens"].apply(lambda x: " ".join(x) + " ") +
        movies["cast_tokens"].apply(lambda x: " ".join(x) + " ") +
        movies["director_tokens"].apply(lambda x: " ".join(x))
    )

    movies["semantic_text"] = (
        movies["title"] + ". " + movies["overview"] + ". " +
        movies["genres_list"].apply(lambda x: ", ".join(x)) + ". " +
        movies["keywords_list"].apply(lambda x: ", ".join(x))
    )

    for c in ["vote_average", "vote_count", "popularity"]:
        movies[c] = pd.to_numeric(movies[c], errors="coerce").fillna(0)

    movies["rating_score"] = normalize(movies["vote_average"])
    movies["popularity_score"] = normalize(np.log1p(movies["popularity"]))
    movies["vote_confidence"] = normalize(np.log1p(movies["vote_count"]))

    return movies.drop_duplicates("title").reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def build_tfidf(tags):
    vec = TfidfVectorizer(
        max_features=15000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )
    return vec, vec.fit_transform(tags)

@st.cache_resource(show_spinner=False)
def build_semantic_model():
    if not SEMANTIC_OK:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def build_embeddings(texts):
    model = build_semantic_model()
    if model is None:
        return None
    return model.encode(
        list(texts),
        show_progress_bar=False,
        normalize_embeddings=True
    )

def find_movie(title, movies):
    title = title.strip().lower()
    exact = movies[movies["title"].str.lower() == title]
    if len(exact):
        return int(exact.index[0])
    partial = movies[movies["title"].str.lower().str.contains(re.escape(title), na=False)]
    return int(partial.index[0]) if len(partial) else None

def get_recommendations(idx, movies, tfidf_matrix, n=10, extra=None, less=None):
    base = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    scores = (
        .55 * base +
        .20 * movies["rating_score"].values +
        .10 * movies["popularity_score"].values +
        .15 * movies["vote_confidence"].values
    )
    extra = [x.lower().replace(" ", "") for x in (extra or [])]
    less = [x.lower().replace(" ", "") for x in (less or [])]

    for i, row in movies.iterrows():
        genres = set(row["genres_tokens"])
        if set(extra) & genres:
            scores[i] += .15
        if set(less) & genres:
            scores[i] -= .10

    result = movies.copy()
    result["recommendation_score"] = scores
    result["content_similarity"] = base
    return result[
        (result.index != idx)
    ].sort_values("recommendation_score", ascending=False).head(n)

def render_row(result, key_prefix, movies):
    if result.empty:
        st.info("No recommendations found.")
        return

    cols = st.columns(min(6, len(result)))
    for i, (_, row) in enumerate(result.iterrows()):
        with cols[i % len(cols)]:
            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
            poster = poster_url(row["id"], row=row)
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown(
                    "<div class='poster-placeholder'>🎬</div>",
                    unsafe_allow_html=True
                )
            st.markdown(
                f"<div class='movie-title'>{html.escape(str(row['title']))}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='meta'>"
                f"<span class='rating-chip'>⭐ {row['vote_average']:.1f}</span>"
                f"<span>{movie_year(row)}</span></div>",
                unsafe_allow_html=True
            )
            genres = row["genres_list"][:3]
            st.markdown(
                "".join(f"<span class='badge'>{html.escape(g)}</span>" for g in genres),
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Details", key=f"{key_prefix}_{i}", use_container_width=True):
                st.session_state["selected_movie"] = int(row.name)

def render_details(idx, movies):
    if idx is None:
        return
    row = movies.loc[idx]
    poster = poster_url(row["id"], row=row)
    st.markdown("<div class='detail-box'>", unsafe_allow_html=True)

    left, right = st.columns([1, 2])
    with left:
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.markdown("<div class='poster-placeholder' style='aspect-ratio:2/3;'>🎬</div>", unsafe_allow_html=True)
    with right:
        st.markdown(f"## {html.escape(str(row['title']))}")
        st.markdown(
            f"⭐ **{row['vote_average']:.1f}/10** &nbsp;&nbsp; "
            f"📅 **{movie_year(row)}** &nbsp;&nbsp; "
            f"🔥 **{row['popularity']:.0f} popularity**",
            unsafe_allow_html=True
        )
        st.write(row["overview"] or "No overview available.")

        genres, director, cast = (
            ", ".join(row["genres_list"]) or "Unknown",
            ", ".join(row["director_list"]) or "Unknown",
            ", ".join(row["cast_list"][:5]) or "Unknown"
        )
        st.markdown(f"**Genres:** {genres}")
        st.markdown(f"**Director:** {director}")
        st.markdown(f"**Cast:** {cast}")

        trailer = trailer_url(row["id"])
        if trailer:
            st.link_button("▶️ Watch Trailer", trailer)
        else:
            st.caption("Trailer button becomes active when a TMDB API key is configured.")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# DATA
# =========================================================
st.sidebar.header("Dataset")
movies_path = st.sidebar.text_input("Movies CSV", "tmdb_5000_movies.csv")
credits_path = st.sidebar.text_input("Credits CSV", "tmdb_5000_credits.csv")

if not os.path.exists(movies_path) or not os.path.exists(credits_path):
    st.markdown(
    '<div class="detail-box">'
    '<h3>📁 Dataset not found</h3>'
    '<p>Place <b>tmdb_5000_movies.csv</b> and <b>tmdb_5000_credits.csv</b> '
    'next to <b>app.py</b>, or enter their full paths in the sidebar.</p>'
    '</div>',
    unsafe_allow_html=True
    )
    st.stop()

with st.spinner("Loading your cinema universe..."):
    movies = load_data(movies_path, credits_path)
    tfidf, tfidf_matrix = build_tfidf(movies["tags"])

# =========================================================
# BRAND BAR
# =========================================================
st.markdown(
'<div class="brand-bar">'
'<div class="brand-logo"><span style="-webkit-text-fill-color:initial;">🎬</span> CINEMIND</div>'
'<div class="brand-tag">AI-Powered Movie Discovery</div>'
'</div>',
unsafe_allow_html=True
)

# =========================================================
# HERO
# =========================================================
hero_movie = "Interstellar" if "Interstellar" in movies["title"].values else movies.iloc[0]["title"]
hero_idx = find_movie(hero_movie, movies)
hero = movies.loc[hero_idx]

st.markdown(
'<div class="hero">'
'<span class="hero-kicker">✨ Featured Pick</span>'
f'<h1>{html.escape(str(hero["title"]))}</h1>'
f'<p>{html.escape(str(hero["overview"])[:420])}</p>'
'<div class="hero-stats">'
f'<div class="stat-pill">⭐ {hero["vote_average"]:.1f}/10</div>'
f'<div class="stat-pill">📅 {movie_year(hero)}</div>'
f'<div class="stat-pill">🎭 {", ".join(hero["genres_list"][:3]) or "—"}</div>'
'</div>'
'</div>',
unsafe_allow_html=True
)

# =========================================================
# TOP SEARCH
# =========================================================
st.markdown('<div class="section-title">🔎 What do you want to watch?</div>', unsafe_allow_html=True)
search = st.text_input(
    "",
    placeholder="Try: 'mind-bending sci-fi about space and time'...",
    label_visibility="collapsed"
)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    min_rating = st.slider("Minimum rating", 0.0, 10.0, 6.0, .5)
with c2:
    count = st.slider("Movies per row", 4, 12, 6)
with c3:
    mood = st.selectbox(
        "Quick mood",
        ["Any mood", "😄 Happy", "😢 Emotional", "😱 Scared", "❤️ Romantic",
         "🤯 Mind-blown", "🔥 Excited", "😌 Relaxed"]
    )

if search:
    if SEMANTIC_OK:
        model = build_semantic_model()
        embeddings = build_embeddings(movies["semantic_text"])
        q = model.encode([search], normalize_embeddings=True)
        scores = cosine_similarity(q, embeddings)[0]
    else:
        q = tfidf.transform([search])
        scores = cosine_similarity(q, tfidf_matrix)[0]

    final = (
        .75 * scores +
        .15 * movies["rating_score"].values +
        .10 * movies["popularity_score"].values
    )
    result = movies.copy()
    result["recommendation_score"] = final
    result = result[result["vote_average"] >= min_rating].sort_values(
        "recommendation_score", ascending=False
    ).head(count)

    st.markdown('<div class="section-title">✨ Best matches for your search</div>', unsafe_allow_html=True)
    render_row(result, "search", movies)
else:
    similar = get_recommendations(hero_idx, movies, tfidf_matrix, n=count)
    st.markdown('<div class="section-title">🔥 Trending-style Picks</div>', unsafe_allow_html=True)
    render_row(similar, "hero", movies)

# =========================================================
# MAIN NAV
# =========================================================
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Discover", "❤️ My Taste", "🎭 Mood", "🎛️ Control"
])

with tab1:
    st.markdown('<div class="section-title">🎬 Pick a movie you love</div>', unsafe_allow_html=True)
    chosen = st.selectbox(
        "Movie",
        movies["title"].tolist(),
        label_visibility="collapsed",
        key="discover_movie"
    )
    if st.button("✨ Find Similar Movies", type="primary", key="discover_btn"):
        idx = find_movie(chosen, movies)
        result = get_recommendations(idx, movies, tfidf_matrix, n=12)
        st.markdown(f"### Because you liked **{chosen}**")
        render_row(result, "discover", movies)

with tab2:
    st.markdown('<div class="section-title">❤️ Build your Movie DNA</div>', unsafe_allow_html=True)

    def enforce_limit():
        if len(st.session_state.liked) > 5:
            st.session_state.liked = st.session_state.liked[:5]
            st.toast("You've reached your limit of 5 movies. Remove one to pick a different movie.", icon="⚠️")

    liked = st.multiselect(
        "Select up to 5 movies you love",
        movies["title"].tolist(),
        key="liked",
        on_change=enforce_limit
    )

    if len(liked) < 5:
        st.caption(f"🎬 {len(liked)}/5 selected")
    else:
        st.success("✅ You've picked all 5! Click below to build your profile.")

    if liked and st.button("🧬 Build My Taste Profile", type="primary"):
        ids = [find_movie(x, movies) for x in liked]
        profile = tfidf_matrix[ids].mean(axis=0)
        scores = cosine_similarity(np.asarray(profile), tfidf_matrix)[0]
        final = .70 * scores + .20 * movies["rating_score"].values + .10 * movies["popularity_score"].values
        result = movies.copy()
        result["recommendation_score"] = final
        result = result[
            (~result.index.isin(ids)) & (result["vote_average"] >= min_rating)
        ].sort_values("recommendation_score", ascending=False).head(12)
        render_row(result, "taste", movies)
    

with tab3:
    st.markdown('<div class="section-title">🎭 Movies for your mood</div>', unsafe_allow_html=True)
    moods = {
        "😄 Happy": "comedy family music fun friendship",
        "😢 Emotional": "drama love family relationship emotional",
        "😱 Scared": "horror thriller mystery dark psychological",
        "❤️ Romantic": "romance love relationship wedding couple",
        "🤯 Mind-blown": "sciencefiction mystery space time psychological",
        "🔥 Excited": "action adventure fight war superhero",
        "😌 Relaxed": "comedy family friendship music feelgood"
    }
    chosen_mood = st.selectbox("Mood", list(moods), key="mood")
    if st.button("🎭 Find My Movies", type="primary"):
        vec = tfidf.transform([moods[chosen_mood]])
        scores = cosine_similarity(vec, tfidf_matrix)[0]
        final = .60 * scores + .25 * movies["rating_score"].values + .15 * movies["popularity_score"].values
        result = movies.copy()
        result["recommendation_score"] = final
        result = result[result["vote_average"] >= min_rating].sort_values(
            "recommendation_score", ascending=False
        ).head(12)
        render_row(result, "mood", movies)

with tab4:
    st.markdown('<div class="section-title">🎛️ Control your recommendations</div>', unsafe_allow_html=True)
    control_movie = st.selectbox("Starting movie", movies["title"].tolist(), key="control")
    genres = sorted({g for x in movies["genres_list"] for g in x})
    more = st.multiselect("➕ More of", genres)
    less = st.multiselect("➖ Less of", genres)

    if st.button("🎛️ Apply My Preferences", type="primary"):
        idx = find_movie(control_movie, movies)
        result = get_recommendations(idx, movies, tfidf_matrix, n=12, extra=more, less=less)
        render_row(result, "control", movies)

# =========================================================
# DETAILS MODAL
# =========================================================
if "selected_movie" in st.session_state:
    idx = st.session_state["selected_movie"]
    if hasattr(st, "dialog"):
        @st.dialog("🎬 Movie Details")
        def movie_dialog():
            render_details(idx, movies)
            if st.button("Close"):
                del st.session_state["selected_movie"]
                st.rerun()
        movie_dialog()
    else:
        st.markdown('<div class="section-title">🎬 Movie Details</div>', unsafe_allow_html=True)
        render_details(idx, movies)

st.markdown(
'<div class="developer-footer">'
'<div class="footer-title">CINEMIND</div>'
'<div class="footer-subtitle">AI-Powered Movie Discovery</div>'
'<div class="footer-developed">Developed by</div>'
'<div class="footer-name">SHRUSTI DIGGAVI</div>'
'<div class="footer-company">IPEC Solutions Private Limited, Bangalore</div>'
'<div class="footer-tech">Built with Python • Machine Learning • NLP • Streamlit</div>'
'</div>',
unsafe_allow_html=True
)
