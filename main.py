import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models.popularity import load_data
from models.content_based import content_based_filtering


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
for key, default in {
    "page": "home",
    "selected_movie": None,
    "search_query": "",
    "genre_filter": "All",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Data Loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_base_data():
    movies = pd.read_csv('data/movie.csv').sample(frac=0.5, random_state=42)
    ratings = pd.read_csv('data/rating.csv').sample(frac=0.5, random_state=42)
    stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()
    movies = movies.merge(stats, on='movieId', how='left')
    movies['avg_rating'] = movies['avg_rating'].fillna(0)
    movies['rating_count'] = movies['rating_count'].fillna(0)
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['clean_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
    movies['search_text'] = movies['clean_title'] + ' ' + movies['genres'].str.replace('|', ' ')
    return movies


@st.cache_resource
def build_tfidf(_movies):
    vec = TfidfVectorizer(ngram_range=(1, 2), analyzer='word')
    mat = vec.fit_transform(_movies['search_text'])
    return vec, mat


@st.cache_data
def load_popularity_data():
    pop_df = load_data()
    if pop_df is None:
        st.error("Could not load popularity data.")
        st.stop()
    return pop_df


@st.cache_resource
def load_content_model():
    cosine_sim, movie_content = content_based_filtering()
    return cosine_sim, movie_content


# ── Trie Autocomplete ─────────────────────────────────────────────────────────
class TrieNode:
    def __init__(self):
        self.children = {}
        self.movies = []

class AutoComplete:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, title, movie_id, popularity):
        node = self.root
        for char in title.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.movies.append((popularity, title, movie_id))

    def search(self, prefix, top_n=5):
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        sorted_movies = sorted(node.movies, key=lambda x: -x[0])
        seen, results = set(), []
        for _, title, mid in sorted_movies:
            if mid not in seen:
                seen.add(mid)
                results.append({'movieId': mid, 'title': title})
            if len(results) == top_n:
                break
        return results


@st.cache_resource
def build_autocomplete(_movies):
    ac = AutoComplete()
    for _, row in _movies.iterrows():
        pop = row['avg_rating'] * np.log1p(row['rating_count'])
        ac.insert(row['clean_title'], row['movieId'], pop)
    return ac


# ── Helpers ───────────────────────────────────────────────────────────────────
def do_search(query, movies, vec, mat, top_n=10, genre_filter=None):
    query_vec = vec.transform([query])
    scores = cosine_similarity(query_vec, mat).flatten()
    df = movies.copy()
    df['score'] = scores
    results = df[df['score'] > 0.01].copy()
    if genre_filter and genre_filter != "All":
        results = results[results['genres'].str.contains(genre_filter, case=False)]
    results['final_score'] = (
        results['score'] * 0.6 +
        (results['avg_rating'] / 5) * 0.25 +
        (np.log1p(results['rating_count']) / 10) * 0.15
    )
    return results.sort_values('final_score', ascending=False).head(top_n)


def get_similar(title, cosine_sim, movie_content, movies_df, top_n=6):
    indices = pd.Series(movie_content.index, index=movie_content['title']).drop_duplicates()
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]
    sim_rows = movie_content.iloc[[i[0] for i in sim_scores]][['title', 'genres']]
    enriched = sim_rows.merge(
        movies_df[['title', 'avg_rating', 'rating_count', 'year']],
        on='title', how='left'
    )
    enriched['avg_rating'] = enriched['avg_rating'].fillna(0)
    enriched['rating_count'] = enriched['rating_count'].fillna(0)
    enriched['year'] = enriched['year'].fillna('')
    return enriched


def stars(r):
    f = int(round(r))
    return "★" * f + "☆" * (5 - f)


def pills(genres_str):
    # handle both "Crime|Drama" and "Crime Drama" formats
    if '|' in genres_str:
        tags = [g.strip() for g in genres_str.split('|') if g.strip()]
    else:
        tags = [g.strip() for g in genres_str.split() if g.strip()]
    return " ".join([f'<span class="genre-pill">{g}</span>' for g in tags[:5]])


def safe_year(val):
    s = str(val)
    return s if s not in ('nan', 'None', '') else ''


def nav(movie_row):
    st.session_state.selected_movie = movie_row.to_dict()
    st.session_state.page = "detail"


# ── Load Data ─────────────────────────────────────────────────────────────────
try:
    movies_df = load_base_data()
    vectorizer, tfidf_matrix = build_tfidf(movies_df)
    ac = build_autocomplete(movies_df)
    pop_df = load_popularity_data()
except FileNotFoundError as e:
    st.error(f"⚠️ Data files not found. Ensure `data/movie.csv` and `data/rating.csv` exist.\n\n`{e}`")
    st.stop()

try:
    cosine_sim, movie_content = load_content_model()
    content_ok = True
except FileNotFoundError:
    content_ok = False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MOVIE DETAIL
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "detail" and st.session_state.selected_movie:
    movie = st.session_state.selected_movie

    if st.button("← Back to results"):
        st.session_state.page = "home"
        st.rerun()

    # Safely extract all fields with fallbacks
    title = movie.get('title', 'Unknown')
    genres = movie.get('genres', '')
    year = safe_year(movie.get('year', ''))
    avg = float(movie.get('avg_rating', 0))
    cnt = int(movie.get('rating_count', 0))
    pct = (avg / 5) * 100

    # Build year html separately to avoid f-string nesting issues
    year_html = f'<div class="detail-year">{year}</div>' if year else ''

    # Hero card
    st.markdown(f"""
<div class="detail-hero">
    {year_html}
    <div class="detail-title">{title}</div>
    <div style="margin:0.6rem 0 1rem">{pills(genres)}</div>
    <div style="display:flex;align-items:flex-end;gap:2.5rem;flex-wrap:wrap;">
        <div>
            <div class="big-rating">{avg:.1f}</div>
            <div class="rating-label">Avg Rating</div>
            <div class="rating-bar-bg" style="width:140px">
                <div class="rating-bar-fill" style="width:{pct:.1f}%"></div>
            </div>
        </div>
        <div>
            <div class="big-rating" style="font-size:1.8rem;color:#888">{cnt:,}</div>
            <div class="rating-label">Total Ratings</div>
        </div>
        <div>
            <div style="font-size:1.4rem;color:#e8c547;letter-spacing:2px">{stars(avg)}</div>
            <div class="rating-label">Stars</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Similar movies
    st.markdown('<div class="section-label">Similar Movies You Might Like</div>', unsafe_allow_html=True)

    if not content_ok:
        st.warning("⚠️ `data/tag.csv` not found — similar movies unavailable.")
    else:
        similar = get_similar(title, cosine_sim, movie_content, movies_df)

        if similar is None or similar.empty:
            st.markdown("""
            <div class="empty-state">
                <div class="icon">🎭</div>
                No similar movies found for this title.
            </div>""", unsafe_allow_html=True)
        else:
            for i, (_, row) in enumerate(similar.iterrows()):
                yr = safe_year(row.get('year', ''))
                r_avg = float(row['avg_rating'])
                r_cnt = int(row['rating_count'])
                r_pct = (r_avg / 5) * 100
                yr_html = f'<span>{yr}</span>' if yr else ''

                col_card, col_btn = st.columns([6, 1])
                with col_card:
                    st.markdown(f"""
<div class="sim-card">
    <div style="display:flex;justify-content:space-between;align-items:start">
        <div>
            <div class="result-title">{row['title']}</div>
            <div style="margin:3px 0 5px">{pills(row['genres'])}</div>
            <div class="result-meta">
                <span class="star">{stars(r_avg)}</span>
                <span>{r_avg:.1f}/5</span>
                <span>{r_cnt:,} ratings</span>
                {yr_html}
            </div>
        </div>
        <div style="text-align:right">
            <div style="font-size:0.65rem;color:#333;margin-bottom:3px">MATCH</div>
            <div style="font-size:1.1rem;color:#e8c547;font-family:'DM Serif Display',serif">#{i+1}</div>
        </div>
    </div>
    <div class="rating-bar-bg">
        <div class="rating-bar-fill" style="width:{r_pct:.1f}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)
                with col_btn:
                    st.write("")
                    st.write("")
                    if st.button("Open →", key=f"sim_{i}"):
                        nav(row)
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
<div style="padding:1.5rem 0 1.2rem">
    <div class="hero-title">Cine<span>Match</span></div>
    <div class="hero-sub">Search movies · click to explore · discover what's similar.</div>
</div>
""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔍  Search", "🏆  Top Rated"])

    # ── SEARCH TAB ────────────────────────────────────────────────────────────
    with tab1:
        col_q, col_g = st.columns([3, 1])
        with col_q:
            st.markdown('<div class="section-label">Search Movies</div>', unsafe_allow_html=True)
            query = st.text_input("", value=st.session_state.search_query,
                                  placeholder="title, genre, keyword...",
                                  label_visibility="collapsed", key="q_input")
            st.session_state.search_query = query
        with col_g:
            st.markdown('<div class="section-label">Genre</div>', unsafe_allow_html=True)
            all_genres = sorted({g for gs in movies_df['genres'].dropna() for g in gs.split('|')})
            opts = ["All"] + all_genres
            genre_choice = st.selectbox("", opts,
                                        index=opts.index(st.session_state.genre_filter)
                                        if st.session_state.genre_filter in opts else 0,
                                        label_visibility="collapsed", key="g_input")
            st.session_state.genre_filter = genre_choice

        # Autocomplete
        if query and len(query) >= 2:
            suggestions = ac.search(query, top_n=4)
            if suggestions:
                st.markdown('<div class="section-label" style="margin-top:0.3rem">Suggestions</div>',
                            unsafe_allow_html=True)
                sug_cols = st.columns(len(suggestions))
                for i, s in enumerate(suggestions):
                    with sug_cols[i]:
                        if st.button(s['title'], key=f"sug_{i}", use_container_width=True):
                            st.session_state.search_query = s['title']
                            st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if query:
            results = do_search(query, movies_df, vectorizer, tfidf_matrix,
                                top_n=10, genre_filter=genre_choice)
            if results.empty:
                st.markdown("""
<div class="empty-state"><div class="icon">🎬</div>No movies found.</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="section-label">{len(results)} results · click Open to see details & similar movies</div>',
                    unsafe_allow_html=True)

                for i, (_, row) in enumerate(results.iterrows()):
                    yr = safe_year(row.get('year', ''))
                    avg = float(row['avg_rating'])
                    cnt = int(row['rating_count'])
                    yr_html = f'<span>{yr}</span>' if yr else ''

                    col_card, col_btn = st.columns([6, 1])
                    with col_card:
                        st.markdown(f"""
<div class="result-card">
    <div class="result-title">{row['title']}</div>
    <div style="margin:3px 0 5px">{pills(row['genres'])}</div>
    <div class="result-meta">
        <span class="star">{stars(avg)}</span>
        <span>{avg:.1f}/5</span>
        <span>{cnt:,} ratings</span>
        {yr_html}
        <span style="color:#e8c547">Score: {row['final_score']:.3f}</span>
    </div>
</div>
""", unsafe_allow_html=True)
                    with col_btn:
                        st.write("")
                        st.write("")
                        if st.button("Open →", key=f"res_{i}"):
                            nav(row)
                            st.rerun()
        else:
            st.markdown("""
<div class="empty-state"><div class="icon">🔍</div>Start typing to search movies.</div>
""", unsafe_allow_html=True)

    # ── TOP RATED TAB ─────────────────────────────────────────────────────────
    with tab2:
        s1, s2, s3 = st.columns(3)
        for box, num, label in [
            (s1, str(len(pop_df)), "Top Rated Films"),
            (s2, f"{pop_df['avg_rating'].mean():.1f}★", "Avg Rating"),
            (s3, f"{int(pop_df['rating_count'].max()):,}", "Most Rated"),
        ]:
            with box:
                st.markdown(f"""
<div class="stat-box">
    <div class="stat-num">{num}</div>
    <div class="stat-label">{label}</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col_lbl, col_sort = st.columns([3, 1])
        with col_lbl:
            st.markdown('<div class="section-label">Highly Rated · click Open to explore</div>',
                        unsafe_allow_html=True)
        with col_sort:
            sort_by = st.selectbox("", ["Rating ↓", "Popularity ↓"],
                                   label_visibility="collapsed", key="pop_sort")

        display_pop = pop_df.sort_values(
            'avg_rating' if sort_by == "Rating ↓" else 'rating_count',
            ascending=False
        ).head(20)

        for i, (_, row) in enumerate(display_pop.iterrows()):
            avg = float(row['avg_rating'])
            cnt = int(row['rating_count'])
            pct = (avg / 5) * 100
            col_card, col_btn = st.columns([6, 1])
            with col_card:
                st.markdown(f"""
<div class="result-card">
    <div class="result-title">{row['title']}</div>
    <div style="margin:3px 0 5px">{pills(row['genres'])}</div>
    <div class="result-meta">
        <span class="star">{stars(avg)}</span>
        <span>{avg:.1f}/5</span>
        <span>{cnt:,} ratings</span>
    </div>
    <div class="rating-bar-bg">
        <div class="rating-bar-fill" style="width:{pct:.1f}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)
            with col_btn:
                st.write("")
                st.write("")
                if st.button("Open →", key=f"pop_{i}"):
                    nav(row)
                    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#222;font-size:0.78rem;">
    CineMatch · TF-IDF Search · Content-Based Filtering · Popularity Ranking
</div>
""", unsafe_allow_html=True)