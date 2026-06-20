# 🚦 TraffiCast AI — Event-Driven Congestion Intelligence

**Flipkart Gridlock Hackathon 2.0 · Problem Statement 2**
Forecast event-related traffic impact → recommend optimal **manpower, barricading & diversion**, and **learn from every event**. Trained **only** on the provided Astram event dataset (no external data).

---

## 🌟 The Solution: Overview

**TraffiCast AI** is a decision-support command center designed for the Bengaluru Traffic Police. By translating historical event incident reports into actionable operational directives, TraffiCast AI prevents gridlocks before they form. 

### Why TraffiCast AI? (Design Honesty)
Unlike traditional navigation systems that rely on real-time vehicle GPS telemetry, the provided dataset contains **event incident reports** (accidents, VIP movements, rallies, breakdowns). TraffiCast AI aligns with this data structure by forecasting **event-related impact, duration, and resource requirements** instead of fabricating traffic flow rates. 
- **Reframed Target Questions**: Predicting exact-minute durations is noisy due to administrative auto-close patterns. We reframed the classification to focus on the decision-relevant question: **"Will this block the road for more than 3 hours?"** (resulting in a highly robust model with **ROC-AUC ≈ 0.86**).
- **Adoption-First Design**: Every model prediction is explainable via local feature contributions (SHAP-style explanations), allowing officers to trust and audit the machine learning logic.

---

## 🛠️ Key Technical Architecture & ML Engine

The backend (`model.py`) trains a set of highly optimized gradient-boosted trees utilizing **LightGBM** (with a fallback to scikit-learn's **HistGradientBoosting** if LightGBM is not installed):

1. **Closure Classifier (`closure_clf`)**: Predicts the probability that an event will require a full or partial road closure. **(ROC-AUC ≈ 0.82)**
2. **Long-Blocker Classifier (`longblock_clf`)**: Predicts the probability that an event will block the road for more than 180 minutes. **(ROC-AUC ≈ 0.86)**
3. **Severity Classifier (`severity_clf`)**: Predicts clearance severity (Low, Moderate, High, Critical) using multi-class classification. **(Accuracy ≈ 76%)**
4. **Duration Regressor (`duration_reg`)**: Continuous regression model estimating the exact incident clearance duration in minutes.

### Feature Engineering Pipeline
- **Spatiotemporal Features**: Cyclic representations of time (Sin/Cos transformations of hour, day of week, month) and distance to the Bengaluru city center.
- **Natural Language Parsing**: Text-mined keywords from raw description texts (`breakdown`, `accident`, `rally`, etc.) and address components (`flyover`, `metro`, `signal`, etc.).
- **Spatial BallTree KNN**: Looks up nearby historical events within a spatial index to calculate historical local closure rates and localized incident density.
- **Target Encoding**: Out-of-fold target encoding for high-cardinality categorical columns (police stations, corridors, zones, event types).

---

## 🖥️ The 10 Dashboard Pages & Features

The Streamlit interface (`app.py`) provides 10 specialized operational modules:

### 1. Command Center
- **Live Operating Picture**: Aggregates and displays all logged events.
- **Interactive Spatiotemporal Mapping**: Leverages Leaflet/Folium (or Plotly Mapbox as a fallback) to map incidents, complete with marker clustering, impact-tier color-coding, and detailed popup cards.
- **Operational Metrics**: Real-time counts of critical events, road closure requirements, and average impact scores.

### 2. Simulate Event (What-if Sandbox)
- **Interactive Prototyping**: Planners input hypothetical event parameters (cause, latitude/longitude, timing) to simulate traffic impact before it occurs.
- **Explainable Predictions**: Displays local feature contributions, showing officers exactly which factor (e.g., peak hour, incident cause) pushed the risk probability up or down.

### 3. Congestion Contagion (Hawkes Process Emulator)
- **Gridlock R0 Modeling**: Formulates gridlock spread using a Hawkes Point Process model, predicting the average number of secondary bottlenecks spawned by a single incident.
- **Upstream Quarantine Intercepts (EIQ)**: Recommends intercept blockades 1.5 - 3 km upstream in opposing directions to divert incoming vehicles before they enter the expanding congestion envelope.

### 4. Resource Optimizer
- **Manpower Allocation**: Solves an optimization problem to dispatch officers from a limited pool across various Traffic Police Station Hubs.
- **Impact-First Dispatch**: Prioritizes dispatches to incidents where the difference between predicted impact and mitigated impact is maximized.

### 5. Diversion Planner (MapmyIndia Routes API)
- **Live Alternative Routing**: Generates multiple routes bypassing the blocked area using MapmyIndia (Mappls) Directions APIs.
- **Cleansed Diversions**: Recommends the route that stays farthest away from the incident coordinate, providing the cleanest detour to sign-post. Runs in a straight-line fallback mode if no API key is configured.

### 6. Hotspot Intelligence
- **DBSCAN Spatial Clustering**: Runs density-based spatial clustering to group historical incidents, revealing recurring chokepoints and high-frequency corridors.
- **Zone Load Profiling**: Renders temporal load heatmaps by weekday and hour for targeted police staging.

### 7. Ask TraffiCast (NLQ Grounded Assistant)
- **Generative AI RAG**: Uses Google Gemini to act as a grounded traffic strategist, answering complex policy and operational queries.
- **Robust Local Fallback**: If no API key is present, a local rule-based regex parsing engine runs analytics, filters the dataset, and automatically renders relevant bar and line charts.

### 8. Advanced Traffic Dynamics
- **Prophecy Breaker (Adversarial Routing)**: Simulates driver route-switching behavior to find a stable Nash Equilibrium, preventing alternative routes from collapsing due to sudden navigation redirects.
- **Stress Contagion**: Models driver stress and road aggression propagation using an SIR (Susceptible-Infected-Recovered) epidemiological model, suggesting deployment details to calm affected corridors.
- **Induced Demand Oracle**: Models long-term travel-time degradation using traffic elasticity coefficients, warning planners of the latent demand loop where widening roads erodes benefit over time.
- **Route Diversity Defender**: Computes Shannon entropy of route choices, distributing traffic patterns to minimize network fragility.
- **Emergency Wake Healer**: Reconstructs traffic platoon flow in the wake of emergency vehicles (e.g., ambulances) by recommending custom green-signal extensions at downstream intersections.

### 9. Post-Event Learning
- **Closed-Loop Feedback**: Officers log the actual outcome of incidents (actual duration, whether a closure was required).
- **Local Model Retraining**: Ingests logged outcomes and retrains LightGBM models locally, updating the cached artifacts for future predictions.

### 10. Model Trust & Performance
- **Operational Auditing**: Displays model validation metrics (ROC-AUC, PR-AUC, multi-class accuracy, and MAE).
- **Global Feature Importance**: Plots global LightGBM feature importances to show what factors drive road closure decisions overall.

---

## 🚀 Installation & Running Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Place Dataset**:
   Place the event log CSV file at: `data/astram_event_data.csv`
3. **Run Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   *Note: The first launch will train the LightGBM models (~20s) and cache them in `artifacts/`. Subsequent launches will start instantly.*

### Environmental Configuration
- `TRAFFICAST_CSV`: Set to override the path to the dataset CSV (default: `data/astram_event_data.csv`).
- `TRAFFICAST_ARTIFACTS`: Set to override the model caching directory (default: `artifacts/`).
- `MAPPLS_REST_KEY`: Set to enable live MapmyIndia Route Directions routing. Alternatively, configure in `.streamlit/secrets.toml`.
- `GEMINI_API_KEY`: Set to enable the LLM RAG Strategist in the "Ask TraffiCast" module.

---

## 📜 Dataset Compliance & Hackathon Rules
As per the hackathon rules and FAQs regarding external data:
- **Trained ONLY on Provided Data**: All predictive models (road closure, long block, severity, duration regression), spatial hotspot clustering (DBSCAN), local density estimators (BallTree KNN), and spatiotemporal Hawkes processes are trained and calculated **strictly on the provided Astram event dataset** (`data/astram_event_data.csv`).
- **No External Data Leakage**: There is zero dependency on external traffic speed telemetry, public road flow data, or secondary datasets for model training or inference.
- **Allowed Display-Time APIs**: The optional MapmyIndia (Mappls) directions API and Google Gemini API are used purely at display-time / inference-time for visualization (diversion routing and RAG policy briefing) and do not violate dataset constraints. If keys are omitted, the application runs entirely offline in fallback mode.
