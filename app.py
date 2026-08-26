
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
      radial-gradient(circle at 80% 5%, rgba(229,9,20,.14), transparent 28%),
      radial-gradient(circle at 10% 20%, rgba(120,80,255,.08), transparent 25%),
      #080808;
    color: #f5f5f5;
}

.block-container {
    max-width: 1450px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

.hero {
    min-height: 390px;
    border-radius: 22px;
    padding: 65px 60px;
    margin-bottom: 30px;
    background:
      linear-gradient(90deg, rgba(0,0,0,.96) 0%, rgba(0,0,0,.78) 45%, rgba(0,0,0,.20) 100%),
      radial-gradient(circle at 75% 30%, rgba(229,9,20,.35), transparent 38%),
      linear-gradient(135deg, #181818, #090909);
    border: 1px solid rgba(255,255,255,.08);
    box-shadow: 0 25px 80px rgba(0,0,0,.5);
}

.hero-kicker {
    color: #e50914;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-size: .85rem;
}

.hero h1 {
    font-size: 4.1rem;
    line-height: 1.02;
    margin: 12px 0;
    font-weight: 800;
}

.hero p {
    max-width: 650px;
    color: #c8c8c8;
    font-size: 1.08rem;
    line-height: 1.65;
}

.section-title {
    font-size: 1.65rem;
    font-weight: 800;
    margin: 28px 0 14px;
}

.movie-card {
    background: #121212;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px;
    padding: 9px;
    transition: .2s ease;
    height: 100%;
}

.movie-card:hover {
    transform: translateY(-5px);
    border-color: rgba(229,9,20,.55);
    box-shadow: 0 12px 35px rgba(0,0,0,.5);
}

.poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    border-radius: 10px;
    background: #202020;
}

.movie-title {
    font-size: 1rem;
    font-weight: 700;
    margin-top: 10px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.meta {
    color: #aaa;
    font-size: .82rem;
    margin-top: 4px;
}

.badge {
    display: inline-block;
    background: rgba(229,9,20,.16);
    color: #ff6970;
    border: 1px solid rgba(229,9,20,.25);
    border-radius: 999px;
    padding: 4px 9px;
    margin: 3px 2px 0 0;
    font-size: .72rem;
}

.detail-box {
    background: linear-gradient(135deg,#171717,#0d0d0d);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 20px;
    padding: 25px;
    margin: 12px 0 25px;
}

.search-wrap {
    background: rgba(20,20,20,.85);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 16px;
    padding: 10px;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: 700;
}

div.stButton > button[kind="primary"] {
    background: #e50914;
    border: none;
}

div.stButton > button[kind="primary"]:hover {
    background: #f40612;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #0d0d0d;
    border-radius: 12px;
    padding: 5px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
}

footer {visibility: hidden;}
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

    # Streamlit Cloud / local secrets
    try:
        api_key = st.secrets.get("TMDB_API_KEY")
        if api_key:
            return api_key
    except Exception:
        pass

    # Environment variable
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
            params={
                "api_key": api_key
            },
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

    # ---------------------------------------------------------
    # OPTION 1: poster_path exists in CSV
    # ---------------------------------------------------------
    if row is not None and "poster_path" in row.index:

        poster_path = row["poster_path"]

        if pd.notna(poster_path):

            poster_path = str(poster_path).strip()

            if poster_path and poster_path.lower() not in [
                "nan",
                "none",
                "null",
                "na",
                ""
            ]:

                if not poster_path.startswith("/"):
                    poster_path = "/" + poster_path

                return f"https://image.tmdb.org/t/p/w500{poster_path}"

    # ---------------------------------------------------------
    # OPTION 2: Fetch from TMDB using movie ID
    # ---------------------------------------------------------
    return get_tmdb_poster(movie_id)
    
    
def trailer_url(movie_id):
    api_key = os.getenv("TMDB_API_KEY")
    try:
        if not api_key:
            api_key = st.secrets.get("TMDB_API_KEY")
    except Exception:
        pass
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
    credits = pd.read_csv(credits_path).rename(columns={"movie_id":"id"})
    cols = [c for c in ["id","cast","crew"] if c in credits.columns]
    movies = movies.merge(credits[cols], on="id", how="left")

    movies["title"] = movies["title"].fillna("Unknown")
    movies["overview"] = movies["overview"].fillna("")
    movies["genres_list"] = movies["genres"].apply(parse_names)
    movies["keywords_list"] = movies["keywords"].apply(parse_names)
    movies["cast_list"] = movies["cast"].apply(lambda x: parse_names(x, 5))
    movies["director_list"] = movies["crew"].apply(parse_director)

    for c in ["genres_list","keywords_list","cast_list","director_list"]:
        movies[c.replace("_list","_tokens")] = movies[c].apply(clean_tokens)

    movies["tags"] = (
        movies["overview"].str.lower() + " " +
        movies["genres_tokens"].apply(lambda x: " ".join(x)+" ") +
        movies["genres_tokens"].apply(lambda x: " ".join(x)+" ") +
        movies["keywords_tokens"].apply(lambda x: " ".join(x)+" ") +
        movies["cast_tokens"].apply(lambda x: " ".join(x)+" ") +
        movies["director_tokens"].apply(lambda x: " ".join(x))
    )

    movies["semantic_text"] = (
        movies["title"] + ". " + movies["overview"] + ". " +
        movies["genres_list"].apply(lambda x: ", ".join(x)) + ". " +
        movies["keywords_list"].apply(lambda x: ", ".join(x))
    )

    for c in ["vote_average","vote_count","popularity"]:
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
        ngram_range=(1,2),
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
    extra = [x.lower().replace(" ","") for x in (extra or [])]
    less = [x.lower().replace(" ","") for x in (less or [])]

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
            poster = poster_url(row["id"], row=row)
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown(
                    "<div class='poster' style='display:flex;align-items:center;"
                    "justify-content:center;font-size:4rem;'>🎬</div>",
                    unsafe_allow_html=True
                )
            st.markdown(
                f"<div class='movie-title'>{html.escape(str(row['title']))}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='meta'>⭐ {row['vote_average']:.1f} • {movie_year(row)}</div>",
                unsafe_allow_html=True
            )
            genres = row["genres_list"][:3]
            st.markdown(
                "".join(f"<span class='badge'>{html.escape(g)}</span>" for g in genres),
                unsafe_allow_html=True
            )

            if st.button("Details", key=f"{key_prefix}_{i}", use_container_width=True):
                st.session_state["selected_movie"] = int(row.name)

def render_details(idx, movies):
    if idx is None:
        return
    row = movies.loc[idx]
    poster = poster_url(row["id"], row=row)
    st.markdown("<div class='detail-box'>", unsafe_allow_html=True)

    left, right = st.columns([1,2])
    with left:
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.markdown("### 🎬 Poster unavailable")
    with right:
        st.markdown(f"## {html.escape(str(row['title']))}")
        st.markdown(
            f"⭐ **{row['vote_average']:.1f}/10** &nbsp;&nbsp; "
            f"📅 **{movie_year(row)}** &nbsp;&nbsp; "
            f"🔥 **{row['popularity']:.0f} popularity**"
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
    st.markdown("""
    <div class="detail-box">
    <h3>📁 Dataset not found</h3>
    <p>Place <b>tmdb_5000_movies.csv</b> and <b>tmdb_5000_credits.csv</b>
    next to <b>app.py</b>, or enter their full paths in the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.spinner("Loading your cinema universe..."):
    movies = load_data(movies_path, credits_path)
    tfidf, tfidf_matrix = build_tfidf(movies["tags"])

# =========================================================
# HERO
# =========================================================
hero_movie = "Interstellar" if "Interstellar" in movies["title"].values else movies.iloc[0]["title"]
hero_idx = find_movie(hero_movie, movies)
hero = movies.loc[hero_idx]

st.markdown(f"""
<div class="hero">
<div class="hero-kicker">AI MOVIE DISCOVERY</div>
<h1>{html.escape(str(hero["title"]))}</h1>
<p>{html.escape(str(hero["overview"])[:420])}</p>
<p><b>⭐ {hero["vote_average"]:.1f}/10</b> &nbsp; • &nbsp;
{movie_year(hero)} &nbsp; • &nbsp; {", ".join(hero["genres_list"][:3])}</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TOP SEARCH
# =========================================================
st.markdown('<div class="section-title">🔎 What do you want to watch?</div>', unsafe_allow_html=True)
search = st.text_input(
    "",
    placeholder="Try: 'mind-bending sci-fi about space and time'...",
    label_visibility="collapsed"
)

c1, c2, c3 = st.columns([1,1,1])
with c1:
    min_rating = st.slider("Minimum rating", 0.0, 10.0, 6.0, .5)
with c2:
    count = st.slider("Movies per row", 4, 12, 6)
with c3:
    mood = st.selectbox(
        "Quick mood",
        ["Any mood","😄 Happy","😢 Emotional","😱 Scared","❤️ Romantic",
         "🤯 Mind-blown","🔥 Excited","😌 Relaxed"]
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
    liked = st.multiselect(
        "Select up to 8 movies you love",
        movies["title"].tolist(),
        max_selections=8,
        key="liked"
    )
    if liked and st.button("🧬 Build My Taste Profile", type="primary"):
        ids = [find_movie(x, movies) for x in liked]
        profile = tfidf_matrix[ids].mean(axis=0)
        scores = cosine_similarity(np.asarray(profile), tfidf_matrix)[0]
        final = .70*scores + .20*movies["rating_score"].values + .10*movies["popularity_score"].values
        result = movies.copy()
        result["recommendation_score"] = final
        result = result[
            (~result.index.isin(ids)) & (result["vote_average"] >= min_rating)
        ].sort_values("recommendation_score", ascending=False).head(12)
        render_row(result, "taste", movies)

with tab3:
    st.markdown('<div class="section-title">🎭 Movies for your mood</div>', unsafe_allow_html=True)
    moods = {
        "😄 Happy":"comedy family music fun friendship",
        "😢 Emotional":"drama love family relationship emotional",
        "😱 Scared":"horror thriller mystery dark psychological",
        "❤️ Romantic":"romance love relationship wedding couple",
        "🤯 Mind-blown":"sciencefiction mystery space time psychological",
        "🔥 Excited":"action adventure fight war superhero",
        "😌 Relaxed":"comedy family friendship music feelgood"
    }
    chosen_mood = st.selectbox("Mood", list(moods), key="mood")
    if st.button("🎭 Find My Movies", type="primary"):
        vec = tfidf.transform([moods[chosen_mood]])
        scores = cosine_similarity(vec, tfidf_matrix)[0]
        final = .60*scores + .25*movies["rating_score"].values + .15*movies["popularity_score"].values
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

st.markdown("---")
st.caption("CineMind • AI-powered movie discovery • Built with Python, NLP, ML & Streamlit")
