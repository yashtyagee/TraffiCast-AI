---
title: TraffiCast AI
emoji: 🚦
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
startup_duration_timeout: 10m
pinned: false
---

# 🚦 TraffiCast AI — Event-Driven Congestion Intelligence

**Flipkart Gridlock Hackathon 2.0 · Problem Statement 2**
Forecast event-related traffic impact → recommend optimal **manpower, barricading & diversion**, and **learn from every event**. Trained **only** on the provided Astram event dataset (no external data).

---

## 🌟 The Solution: Overview

**TraffiCast AI** is a decision-support command center designed for the Bengaluru Traffic Police. By translating historical event incident reports into actionable operational directives, TraffiCast AI prevents gridlocks before they form.

### Why TraffiCast AI? (Design Honesty)
Unlike traditional navigation systems that rely on real-time vehicle GPS telemetry, the provided dataset contains **event incident reports** (accidents, VIP movements, rallies, breakdowns). TraffiCast AI aligns with this data structure by forecasting **event-related impact, duration, and resource requirements** instead of fabricating traffic flow rates.
- **Reframed Target Questions**: Predicting exact-minute durations is noisy due to administrative auto-close patterns. We reframed the classification to focus on the decision-relevant question: **"Will this block the road for more than 3 hours?"** (resulting in a highly robust model with **ROC-AUC ≈ 0.86**).
- **Temporal Train/Test Split**: Models are validated using a strict chronological 80/20 split (train on earlier events, test on later events) to prevent data leakage.
- **Adoption-First Design**: Every model prediction is explainable via local feature contributions (SHAP-style explanations), allowing officers to trust and audit the machine learning logic.

---

## 🛠️ Key Technical Architecture & ML Engine

The backend (`model.py`) trains a set of highly optimized gradient-boosted trees utilizing **LightGBM** (with a fallback to scikit-learn's **HistGradientBoosting** if LightGBM is not installed):

1. **Closure Classifier (`closure_clf`)**: Predicts the probability that an event will require a full or partial road closure. **(ROC-AUC ≈ 0.82)**
2. **Long-Blocker Classifier (`longblock_clf`)**: Predicts the probability that an event will block the road for more than 180 minutes. **(ROC-AUC ≈ 0.86)**
3. **Severity Classifier (`severity_clf`)**: Predicts clearance severity (short / medium / long) using multi-class classification. **(Accuracy ≈ 76%)**
4. **Duration Regressor (`duration_reg`)**: Continuous regression model estimating the actual incident clearance duration in minutes.

### Feature Engineering Pipeline
- **Spatiotemporal Features**: Cyclic representations of time (Sin/Cos transformations of hour, day of week, month) and distance to the Bengaluru city center.
- **Natural Language Parsing**: Text-mined keywords from raw description texts (`breakdown`, `accident`, `rally`, etc.) and address components (`flyover`, `metro`, `signal`, etc.).
- **Spatial BallTree KNN**: Looks up nearby historical events within a spatial index to calculate historical local closure rates and localized incident density.
- **Target Encoding**: Out-of-fold target encoding for high-cardinality categorical columns (police stations, corridors, zones, event types).

---

## 🖥️ The 12 Dashboard Pages & Features

The Streamlit interface (`app.py`) provides 12 specialized operational modules:

### 1. Command Center
- **Live Operating Picture**: Aggregates and displays all logged events scored with a unified Event Impact Score (EIS 0–100).
- **Interactive Spatiotemporal Mapping**: Leverages Leaflet/Folium (or Plotly Mapbox as a fallback) to map incidents, complete with marker clustering, impact-tier color-coding, and detailed popup cards.
- **Event Spatiotemporal Risk Heatmap**: 7×24 grid showing historical average EIS across days of the week and hours of the day.
- **Per-Event SHAP Explanations**: Select any event to see which features pushed its closure probability up or down.

### 2. Simulate Event (What-if Sandbox)
- **Interactive Prototyping**: Planners input hypothetical event parameters (cause, latitude/longitude, timing) to simulate traffic impact before it occurs.
- **Prescriptive Counterfactuals**: Perturbs model inputs (e.g., "pre-position crane", "set upstream diversions") to show the duration/closure reduction achievable from each intervention.
- **Historical Similarity Precedents**: Finds the top 5 geographically and semantically similar past incidents.
- **Projected Severity Timeline**: Model-driven 3-hour severity evolution projection based on predicted duration.

### 3. Event Playbook Generator (Core PS2)
- **Per-Cause Operational Playbook**: Select any event cause to get its historical P50/P90 duration, road closure rate, and top impacted corridors.
- **Pre-Approved Resource Targets**: Recommended officers, barricades, and crane deployment probability.
- **Copyable SOP Briefing Card**: Auto-generated Standard Operating Procedure text ready for dispatch channels.

### 4. Shift Bandobast Planner (Core PS2)
- **Optimal Manpower Allocation**: Inspector inputs a date + available officer count → gets a printable, zone-level deployment sheet per shift.
- **Historical Load Profile**: Distributes the officer pool proportionally to zone × weekday × shift-hour-block historical load (event count × (1 + road closure rate)).
- **Printable Deployment Briefing Sheet**: Copy-paste ready text for field distribution.

### 5. Public Diversion Advisory (Core PS2)
- **Data-Driven Alternative Corridors**: Selects corridors in the same zone with lower historical incident frequency.
- **Tweet-Ready Public Broadcast Notice**: Auto-generated advisory text for social media and dispatch channels.
- **Route Bypass Map**: Visualizes bypass routes using MapmyIndia Directions API (straight-line fallback if no API key).

### 6. Conflict + Dispatch (Core PS2)
- **Conflict Zone Detection**: Identifies clusters of simultaneous events within a configurable spatial radius.
- **Compounded Event Impact Score**: Aggregates overlapping event severity with diminishing returns.
- **Nearest-Hub Greedy Dispatch**: Matches officers from real police station hubs (coordinates derived from dataset) to active incidents, minimizing travel cost.

### 7. Waterlogging / Chokepoint Analyzer
- **Unplanned Event Cause Analysis**: Filters the dataset for waterlogging/water-related incidents and computes clearance time, closure rates, and DBSCAN-clustered hotspots.
- **Pre-Monsoon Mitigation Checklist**: Actionable recommendations for pump pre-positioning and drain maintenance.

### 8. Crane Pre-Positioning
- **Accident/Breakdown Resource Deployment**: Identifies top breakdown corridors and overlays police station hub locations.
- **Actionable Pre-Position Briefing**: Recommends which station should pre-position recovery vehicles near the highest-incident corridors.

### 9. Upstream Risk Buffer
- **Spatial Incident Query**: Analyses historical event density within a configurable radius around any point.
- **Risk Factor Calculation**: Computes localized density-based risk levels from historical data.
- **Upstream Intercept Actions**: Directional recommendations for pre-emptive traffic interception.

### 10. Post-Event Learning
- **Closed-Loop Feedback**: Officers log the actual outcome of incidents (actual duration, whether a closure was required).
- **Learning Payoff Tracking**: Measures model error reduction from incorporating feedback.
- **Local Model Retraining**: Ingests logged outcomes and retrains LightGBM models locally, updating the cached artifacts for future predictions.

### 11. Model Trust & Performance
- **Operational Auditing**: Displays model validation metrics (ROC-AUC, PR-AUC, multi-class accuracy, and MAE) from the temporal test split.
- **Global Feature Importance**: Plots global LightGBM feature importances with human-readable labels to show what factors drive road closure decisions overall.
- **Design Honesty Statement**: Transparent disclaimer about what the models can and cannot do.

### 12. Astram Query Center
- **Offline Traffic Analytics Engine**: Answers natural language queries about corridors, risk profiles, zone comparisons, and resource planning using pre-compiled pandas aggregates directly on the Astram dataset.
- **Auto-Visualization**: Automatically renders relevant bar charts, line charts, and tables based on the query keywords.

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

---

## 📜 Dataset Compliance & Hackathon Rules
As per the hackathon rules and FAQs regarding external data:
- **Trained ONLY on Provided Data**: All predictive models (road closure, long block, severity, duration regression), spatial hotspot clustering (DBSCAN), local density estimators (BallTree KNN), and zone load profiles are trained and calculated **strictly on the provided Astram event dataset** (`data/astram_event_data.csv`).
- **No External Data Leakage**: There is zero dependency on external traffic speed telemetry, public road flow data, or secondary datasets for model training or inference.
- **Allowed Display-Time APIs**: The optional MapmyIndia (Mappls) directions API is used purely at display-time / inference-time for route visualization and does not violate dataset constraints. If keys are omitted, the application runs entirely offline in fallback mode.
- **Temporal Validation**: Models are validated with a strict chronological 80/20 train/test split (no random shuffling) to prevent temporal leakage.
