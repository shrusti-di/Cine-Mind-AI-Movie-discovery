# CineMind Pro — Netflix-style Streamlit UI
Deployed the project on livemy.app : https://vh-prod-bollywood-look-alike-ai-main-5cc657-9a004260.livemy.site/
## Files

Put these together:

```text
CineMind/
├── app.py
├── requirements.txt
├── tmdb_5000_movies.csv
└── tmdb_5000_credits.csv
```
DATASET LINK : https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## UI features

- Dark cinematic Netflix/Prime-style interface
- Hero movie section
- Large poster cards
- Horizontal-style recommendation rows
- Global natural-language search
- Similar movie recommendations
- Movie DNA / personalized recommendations
- Mood recommendations
- Controllable recommendations
- Movie details dialog
- TMDB poster integration
- YouTube trailer button
- Hybrid recommendation scoring

## TMDB posters and trailers

The app works without an API key, but posters/trailers require a TMDB API key.

For local use, set:

```bash
# Windows PowerShell
$env:TMDB_API_KEY="YOUR_KEY"
```

Or create:

```text
.streamlit/secrets.toml
```

with:

```toml
TMDB_API_KEY = "YOUR_KEY"
```

For Streamlit Cloud, add `TMDB_API_KEY` under App Settings → Secrets.

The app uses TMDB only for poster/trailer metadata; the recommendation engine continues to use your local CSV data.
