# 🚦 TraffiCast AI — Event-Driven Congestion Intelligence

**Flipkart Gridlock Hackathon 2.0 · Problem Statement 2**
Forecast event-related traffic impact → recommend optimal **manpower, barricading & diversion**,
and **learn from every event**. Trained **only** on the provided Astram event dataset (no external data).

---

## What it does

| Module | Question answered | Validated performance |
|---|---|---|
| Road-closure predictor | Will this event need a road closure? | **ROC-AUC ≈ 0.82** |
| Long-blocker predictor | Will it block the road > 3 hours? | **ROC-AUC ≈ 0.86** |
| Severity classifier | short / medium / long clearance | **≈ 76% accuracy** |
| Hotspot intelligence | Where do events recur? (DBSCAN) | 50 clusters |
| Surge detector | Is an unplanned gathering forming? | 6-hour z-score |
| Resource optimizer | How to allocate a limited officer pool? | impact-first allocation |
| Post-event learning | Were we right? Improve. | feedback log + retrain |

The dashboard has 7 pages: Command Center, Simulate Event, Resource Optimizer,
Hotspot Intelligence, Ask TraffiCast (NLQ), Post-Event Learning, Model Trust.

---

## Run locally

```bash
pip install -r requirements.txt
# place the provided CSV here:  data/astram_event_data.csv
streamlit run app.py
```

First launch trains the models (~20s) and caches them to `artifacts/`.
Later launches load the cached bundle instantly.

### Configuration (optional env vars)
- `TRAFFICAST_CSV` — path to the event CSV (default `data/astram_event_data.csv`)
- `TRAFFICAST_ARTIFACTS` — where to cache models (default `artifacts/`)

### MapmyIndia (Mappls) routing — optional, for the Diversion Planner
External map/routing **APIs are permitted** by the hackathon admin (they are used at
display time only, NOT as training data). To enable live diversion routing, add a key:

```toml
# .streamlit/secrets.toml
MAPPLS_REST_KEY = "your_rest_api_key"
# or OAuth:
# MAPPLS_CLIENT_ID = "..."
# MAPPLS_CLIENT_SECRET = "..."
```
Without a key the Diversion Planner runs in safe fallback mode (straight-line preview),
so the rest of the app always works.

**Which Mappls APIs we use:** Routes & Navigation (Directions/Route API) for diversions,
optionally the Maps SDK traffic layer for the base map and Geocoding for address lookup.

## Deploy on Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo (include `data/astram_event_data.csv`, < 50 MB).
2. On share.streamlit.io → New app → point to `app.py`.
3. Done — share the public URL in your submission.

---

## Files
- `app.py` — Streamlit UI (7 pages)
- `model.py` — feature engineering, training, inference, recommendation, hotspots, surge, explainability
- `requirements.txt`, `README.md`
- `artifacts/` — cached model bundle (auto-created)

## Design honesty (a strength)
This dataset is **event incident reports**, not road-flow telemetry, so we forecast event
**impact & resource need** instead of fabricating traffic flow. Exact-minute duration is
intrinsically noisy (administrative auto-close), so we reframed it as the decision-relevant
**"blocks > 3h?"** question — far more accurate and more useful to officers.
Every prediction is explainable (feature contributions) → built for adoption by Bengaluru Traffic Police.
