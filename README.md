# Restaurant Data Dashboard

An interactive data dashboard analyzing restaurant listings scraped from Google Maps and TripAdvisor for a single city. Built for Data Wrangling Project 2.

## Dataset

The dataset was scraped using Python (BeautifulSoup + requests) and stored in a SQLite database (`restaurants_data.db`). It contains two tables:

- **Restaurants** — 283 restaurants with name, address, phone, categories, Google score, TripAdvisor score, review counts, and hours
- **Reviews** — ~947 individual reviews with source platform, title, text, star rating, and NLP sentiment score

The scraping code lives in the companion repo: [RestaurantsWebScrape](https://github.com/raltomar/RestaurantsWebScrape)

## Visualizations

| # | Chart | Description |
|---|-------|-------------|
| 1 | **Map** | Folium MarkerCluster map — markers colored green→red by Google score |
| 2 | **Score Distribution** | Overlaid histograms: Google vs TripAdvisor scores |
| 3 | **Cuisine Categories** | Horizontal bar chart of top-N cuisine types |
| 4 | **Cross-Source Comparison** | Scatter: Google score vs TA score with Pearson correlation |
| 5 | **Sentiment vs Rating** | Scatter: NLP sentiment score vs star rating with OLS trendline |
| 6 | **Top 20 Table** | Ranked table of highest-rated restaurants |

## Interactive Elements

The dashboard includes 6 interactive controls:
1. Cuisine dropdown (filter by food type)
2. Minimum score slider
3. Minimum review count slider
4. Review source selector (Google / TripAdvisor)
5. Top-N categories slider
6. Map toggle checkbox

## Files

```
RestaurantDashboard/
├── Dashboard.ipynb          # Main analysis notebook
├── streamlit_app.py         # Streamlit web app
├── restaurants_data.db      # SQLite database
├── geocode_cache.csv        # Cached lat/lon for all addresses
└── DW_Project_Sources/      # Reference notebooks (Plotly, Folium, Dash)
```

## Running the Streamlit App

```bash
pip install streamlit pandas plotly folium streamlit-folium requests geopy statsmodels
streamlit run streamlit_app.py
```

## Running the Notebook

Open `Dashboard.ipynb` in JupyterLab. Run all cells top-to-bottom. The database is downloaded automatically on first run. Geocoding is cached to `geocode_cache.csv` and skipped on subsequent runs.

---

*Data scraped and dashboard built by Raphael Altomar — Data Wrangling Project 2*
