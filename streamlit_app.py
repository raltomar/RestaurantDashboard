"""
Restaurant Data Dashboard — Streamlit App

Install dependencies:
    pip install streamlit pandas plotly folium streamlit-folium requests geopy statsmodels

Run:
    streamlit run streamlit_app.py
"""

import os
import sqlite3
import requests

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
import streamlit as st

try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False

# ── Constants ────────────────────────────────────────────────────────────────

DB_URL = "https://github.com/raltomar/RestaurantsWebScrape/raw/refs/heads/main/restaurants_data.db"
DB_PATH = "restaurants_data.db"

CATEGORY_CLUSTERS = [
    ("Asian",             ["Sushi", "Japanese", "Chinese", "Thai", "Filipino", "Indian", "Mongolian", "Korean", "Vietnamese", "Asian"]),
    ("Mexican & Latin",   ["Mexican", "Latin American", "Caribbean", "Cuban", "Spanish"]),
    ("Italian & Pizza",   ["Italian", "Pizza"]),
    ("Seafood",           ["Seafood", "Fish & Seafood"]),
    ("Mediterranean",     ["Mediterranean", "Greek", "Middle Eastern", "French"]),
    ("Bars & Nightlife",  ["Cocktail Lounge", "Brew Pub", "Sports Bar", "Tavern", "Night Club", "Wine Bar"]),
    ("Coffee & Bakery",   ["Coffee", "Breakfast", "Brunch", "Bakeries", "Bakery", "Donut", "Bagel", "Cafeteria"]),
    ("Barbecue",          ["Barbecue", "Hawaiian"]),
    ("American",          ["American", "Family Style", "Steak", "Hamburger", "Chicken", "Home Cooking", "Buffet", "Sandwich"]),
    ("Bars",              ["Bar"]),
    ("Desserts",          ["Dessert", "Ice Cream"]),
    ("Health & Specialty",["Health Food", "Juice"]),
]

def assign_cluster(cat_string):
    if not cat_string or (isinstance(cat_string, float) and pd.isna(cat_string)):
        return "Other"
    s = str(cat_string).lower()
    for cluster_name, keywords in CATEGORY_CLUSTERS:
        if any(kw.lower() in s for kw in keywords):
            return cluster_name
    return "Other"

# ── Data loading ─────────────────────────────────────────────────────────────

def _fetch_db():
    if not os.path.exists(DB_PATH):
        r = requests.get(DB_URL, stream=True)
        r.raise_for_status()
        with open(DB_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


@st.cache_data
def load_data():
    _fetch_db()
    conn = sqlite3.connect(DB_PATH)
    restaurants = pd.read_sql("SELECT * FROM Restaurants", conn)
    reviews = pd.read_sql("SELECT * FROM Reviews", conn)
    conn.close()

    # Numeric casting
    for col in ["score", "ta score", "number of reviews", "ta number of reviews"]:
        restaurants[col] = pd.to_numeric(restaurants[col], errors="coerce")
    for col in ["rating", "sentiment_score"]:
        reviews[col] = pd.to_numeric(reviews[col], errors="coerce")

    # String cleanup
    for col in ["name", "address", "categories", "hours", "phone"]:
        if col in restaurants.columns:
            restaurants[col] = restaurants[col].astype(str).str.strip().replace("nan", None)

    # Category clustering
    restaurants["categories_list"] = (
        restaurants["categories"]
        .fillna("")
        .str.split(",")
        .apply(lambda xs: [x.strip() for x in xs if x.strip()])
    )
    restaurants["cluster"] = restaurants["categories"].apply(assign_cluster)
    restaurants["primary_category"] = restaurants["cluster"]
    restaurants["combined_score"] = restaurants[["score", "ta score"]].mean(axis=1)
    restaurants["total_reviews"] = (
        restaurants["number of reviews"].fillna(0)
        + restaurants["ta number of reviews"].fillna(0)
    )

    # Use lat/lon already in the database
    restaurants = restaurants.rename(columns={"latitude": "lat", "longitude": "lon"})

    categories_long = restaurants[["id", "name", "score", "ta score", "cluster"]].copy()
    categories_long = categories_long.rename(columns={"cluster": "category"})

    return restaurants, reviews, categories_long


# ── Chart factories ───────────────────────────────────────────────────────────

def score_to_color(s):
    if pd.isna(s): return "gray"
    if s >= 4.5: return "green"
    if s >= 4.0: return "lightgreen"
    if s >= 3.5: return "orange"
    return "red"


def make_map(df):
    df_map = df.dropna(subset=["lat", "lon"]).copy()
    if df_map.empty:
        return folium.Map(location=[0, 0], zoom_start=2)
    center = [df_map["lat"].mean(), df_map["lon"].mean()]
    m = folium.Map(location=center, zoom_start=13)
    cluster = MarkerCluster().add_to(m)
    for _, row in df_map.iterrows():
        popup_html = (
            f"<b>{row['name']}</b><br>"
            f"Score: {row.get('score', 'N/A')} | TA: {row.get('ta score', 'N/A')}<br>"
            f"<i>{row.get('categories', 'N/A')}</i><br>"
            f"Hours: {row.get('hours', 'N/A')}<br>"
            f"Phone: {row.get('phone', 'N/A')}"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7, fill=True,
            fill_color=score_to_color(row.get("score")),
            color=score_to_color(row.get("score")),
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=row["name"],
        ).add_to(cluster)
    return m


def make_rating_hist(df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["score"].dropna(), name="Score",
        opacity=0.7, marker_color="#2A9D8F",
        xbins=dict(start=0, end=5, size=0.25),
    ))
    fig.add_trace(go.Histogram(
        x=df["ta score"].dropna(), name="TA Score",
        opacity=0.6, marker_color="#E76F51",
        xbins=dict(start=0, end=5, size=0.25),
    ))
    fig.update_layout(
        barmode="overlay",
        title="Score Distribution: Yellow Pages vs TripAdvisor",
        xaxis_title="Score (0–5)", yaxis_title="Number of Restaurants",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    fig.add_annotation(
        text=f"n = {df['score'].notna().sum()} restaurants",
        xref="paper", yref="paper", x=0.01, y=0.98,
        showarrow=False, font=dict(size=11, color="gray"),
    )
    return fig


def make_category_bar(df_long):
    counts = df_long["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    fig = px.bar(
        counts, x="count", y="category", orientation="h",
        color="count", color_continuous_scale="Teal",
        title="Restaurants by Cuisine Cluster",
        labels={"count": "Number of Restaurants", "category": "Cuisine"},
        template="plotly_white",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    return fig


def make_cross_source_scatter(df):
    df_s = df.dropna(subset=["score", "ta score"]).copy()
    if df_s.empty:
        return go.Figure().update_layout(title="No data with both scores")
    fig = px.scatter(
        df_s, x="score", y="ta score", color="primary_category",
        hover_data={"name": True, "score": True, "ta score": True, "primary_category": False},
        title="Score Comparison: Yellow Pages vs TripAdvisor",
        labels={"score": "Score (YP)", "ta score": "TA Score", "primary_category": "Cuisine"},
        template="plotly_white", opacity=0.75,
    )
    corr = df_s["score"].corr(df_s["ta score"])
    fig.add_annotation(
        text=f"Pearson r = {corr:.3f}",
        xref="paper", yref="paper", x=0.02, y=0.95,
        showarrow=False, font=dict(size=12), bgcolor="lightyellow", bordercolor="gray",
    )
    lim = [
        min(df_s["score"].min(), df_s["ta score"].min()) - 0.1,
        max(df_s["score"].max(), df_s["ta score"].max()) + 0.1,
    ]
    fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                  line=dict(color="gray", dash="dash", width=1))
    return fig


def make_sentiment_scatter(reviews_df):
    df_r = reviews_df.dropna(subset=["sentiment_score", "rating"]).copy()
    if df_r.empty:
        return go.Figure().update_layout(title="No review data")
    try:
        import statsmodels  # noqa: F401
        fig = px.scatter(
            df_r, x="sentiment_score", y="rating", color="source",
            opacity=0.5, trendline="ols",
            title="Review Sentiment Score vs Star Rating",
            labels={"sentiment_score": "Sentiment Score", "rating": "Star Rating"},
            template="plotly_white",
        )
    except ImportError:
        fig = px.scatter(
            df_r, x="sentiment_score", y="rating", color="source",
            opacity=0.5,
            title="Review Sentiment Score vs Star Rating",
            labels={"sentiment_score": "Sentiment Score", "rating": "Star Rating"},
            template="plotly_white",
        )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def make_top_table(df, n=20):
    cols = ["name", "score", "ta score", "number of reviews", "categories"]
    df_top = df.sort_values("score", ascending=False).head(n)[cols].reset_index(drop=True)
    n_rows = len(df_top)
    row_colors = ["#edf6f4" if i % 2 == 0 else "#ffffff" for i in range(n_rows)]
    col_fill = [row_colors] * len(cols)
    fig = go.Figure(data=[go.Table(
        columnwidth=[25, 180, 70, 70, 80, 280],
        header=dict(
            values=["#", "Name", "Score", "TA Score", "# Reviews", "Categories"],
            fill_color="#264653",
            font=dict(color="#ffffff", size=12, family="Arial"),
            align="left",
            height=36,
        ),
        cells=dict(
            values=[
                list(range(1, n_rows + 1)),
                df_top["name"].tolist(),
                df_top["score"].round(2).tolist(),
                df_top["ta score"].round(2).tolist(),
                df_top["number of reviews"].fillna(0).astype(int).tolist(),
                df_top["categories"].tolist(),
            ],
            fill_color=col_fill,
            align="left",
            font=dict(color="#1a1a1a", size=11, family="Arial"),
            height=30,
        ),
    )])
    fig.update_layout(
        title=f"Top {n} Restaurants by Score",
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ── Filtering ────────────────────────────────────────────────────────────────

def filter_restaurants(df, cuisines, min_score, min_revs):
    mask = pd.Series([True] * len(df), index=df.index)
    if cuisines:
        mask &= df["cluster"].isin(cuisines)
    mask &= df["score"].fillna(0) >= min_score
    mask &= df["number of reviews"].fillna(0) >= min_revs
    return df[mask]


# ── App layout ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Restaurant Dashboard",
    page_icon="🍽️",
    layout="wide",
)

restaurants, reviews, categories_long = load_data()

all_categories = sorted(categories_long["category"].dropna().unique().tolist())
all_sources = sorted(reviews["source"].dropna().unique().tolist())
max_reviews = int(restaurants["number of reviews"].max() or 0)

# Sidebar
with st.sidebar:
    st.title("🍽️ Restaurant Dashboard")
    st.markdown("Filter the data across all tabs.")

    selected_cuisines = st.multiselect(
        "Cuisines", options=all_categories, default=[],
        help="Leave blank to show all cuisines",
    )
    min_score = st.slider("Min score", 0.0, 5.0, 0.0, 0.1)
    min_revs = st.slider("Min review count", 0, max_reviews, 0)
    selected_sources = st.multiselect(
        "Review sources", options=all_sources, default=all_sources,
    )
    with st.expander("About this dashboard"):
        st.markdown(
            "**Data source:** Restaurant listings scraped from Yellow Pages and TripAdvisor "
            "by Raphael Altomar. Stored in a SQLite database.\n\n"
            "**Coordinates:** Lat/lon sourced directly from the scraped database. "
            "Restaurants without coordinates are excluded from the map.\n\n"
            "**Sentiment scores:** Pre-computed NLP scores in the Reviews table, ranging −1 (negative) to +1 (positive).\n\n"
            "**Code:** [github.com/raltomar/RestaurantsWebScrape](https://github.com/raltomar/RestaurantsWebScrape)"
        )

# Apply filters
df_f = filter_restaurants(restaurants, selected_cuisines, min_score, min_revs)
sources = selected_sources if selected_sources else all_sources
rev_f = reviews[reviews["source"].isin(sources)]
cat_f = categories_long[categories_long["id"].isin(df_f["id"])]

# Tabs
tab_overview, tab_map, tab_charts, tab_reviews = st.tabs(
    ["Overview", "Map", "Charts", "Reviews"]
)

# ── Tab: Overview ─────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown(
        "A snapshot of the filtered restaurant dataset. Metrics update as you adjust "
        "the sidebar filters."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Restaurants", len(df_f))
    c2.metric("Avg Score (YP)", f"{df_f['score'].mean():.2f}" if df_f['score'].notna().any() else "—")
    c3.metric("Avg TA Score", f"{df_f['ta score'].mean():.2f}" if df_f['ta score'].notna().any() else "—")
    c4.metric("Total Reviews", f"{int(df_f['total_reviews'].sum()):,}")

    st.divider()
    st.markdown("#### Top Restaurants by Score")
    st.plotly_chart(make_top_table(df_f), use_container_width=True)

# ── Tab: Map ──────────────────────────────────────────────────────────────────
with tab_map:
    st.markdown(
        "Each marker represents a restaurant. Color indicates score: "
        "**green** ≥4.5 · **light green** ≥4.0 · **orange** ≥3.5 · **red** <3.5. "
        "Click a cluster to zoom in, then click a marker to see details."
    )
    if not HAS_ST_FOLIUM:
        st.warning(
            "`streamlit-folium` is not installed. Run `pip install streamlit-folium` "
            "then restart the app to enable the interactive map."
        )
    else:
        m = make_map(df_f)
        map_data = st_folium(m, width=None, height=550, returned_objects=["last_object_clicked_tooltip"])

        clicked_name = (map_data or {}).get("last_object_clicked_tooltip")
        if clicked_name:
            match = df_f[df_f["name"] == clicked_name]
            if not match.empty:
                row = match.iloc[0]
                st.divider()
                st.subheader(f"📍 {row['name']}")
                col_a, col_b = st.columns(2)
                col_a.markdown(
                    f"**Categories:** {row.get('categories', '—')}  \n"
                    f"**Hours:** {row.get('hours', '—')}  \n"
                    f"**Phone:** {row.get('phone', '—')}  \n"
                    f"**Address:** {row.get('address', '—')}"
                )
                col_b.markdown(
                    f"**Score:** {row.get('score', '—')}  \n"
                    f"**TA Score:** {row.get('ta score', '—')}  \n"
                    f"**Reviews:** {int(row.get('number of reviews', 0) or 0)}  \n"
                    f"**TA Reviews:** {int(row.get('ta number of reviews', 0) or 0)}"
                )
                rest_reviews = reviews[reviews["restaurant_id"] == row["id"]]
                if not rest_reviews.empty:
                    st.markdown("**Reviews:**")
                    st.dataframe(
                        rest_reviews[["source", "title", "rating", "sentiment_score", "text"]],
                        use_container_width=True,
                        hide_index=True,
                    )

# ── Tab: Charts ───────────────────────────────────────────────────────────────
with tab_charts:
    st.markdown(
        "Six interactive visualizations covering score distributions, cuisine breakdown, "
        "cross-platform score agreement, and review sentiment analysis."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_rating_hist(df_f), use_container_width=True)
    with col2:
        st.plotly_chart(make_category_bar(cat_f), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(make_cross_source_scatter(df_f), use_container_width=True)
    with col4:
        st.plotly_chart(make_sentiment_scatter(rev_f), use_container_width=True)

# ── Tab: Reviews ──────────────────────────────────────────────────────────────
with tab_reviews:
    st.markdown(
        "Browse all reviews filtered by source platform. Select a specific restaurant "
        "to drill into its individual reviews and rating distribution."
    )
    st.markdown(f"**{len(rev_f):,} reviews** from: {', '.join(sources)}")

    restaurant_names = sorted(df_f["name"].dropna().unique().tolist())
    selected_restaurant = st.selectbox(
        "Drill into a specific restaurant (optional)",
        options=["— Show all —"] + restaurant_names,
    )

    if selected_restaurant != "— Show all —":
        match = df_f[df_f["name"] == selected_restaurant]
        if not match.empty:
            rid = match.iloc[0]["id"]
            drill = rev_f[rev_f["restaurant_id"] == rid]
            st.markdown(f"**{len(drill)} reviews** for *{selected_restaurant}*")
            if not drill.empty:
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    rating_hist = px.histogram(
                        drill, x="rating", nbins=10,
                        title="Rating Distribution",
                        template="plotly_white",
                        color_discrete_sequence=["#2A9D8F"],
                    )
                    st.plotly_chart(rating_hist, use_container_width=True)
                with col_r2:
                    st.dataframe(
                        drill[["source", "title", "rating", "sentiment_score", "text"]],
                        use_container_width=True,
                        hide_index=True,
                    )
    else:
        st.dataframe(
            rev_f[["source", "title", "rating", "sentiment_score", "text"]].head(200),
            use_container_width=True,
            hide_index=True,
        )
        if len(rev_f) > 200:
            st.caption(f"Showing first 200 of {len(rev_f):,} reviews. Use the restaurant selector above to drill in.")
