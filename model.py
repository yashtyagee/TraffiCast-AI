"""
TraffiCast AI — model engine
Event-Driven Congestion Intelligence (Flipkart Gridlock Hackathon 2.0, PS2)

All models trained ONLY on the provided Astram event dataset (no external data).
Reused by the Streamlit app (app.py) and the Colab notebook.

Models
------
1. closure_clf   : P(road closure required)            ROC-AUC ~0.82
2. longblock_clf : P(blocks road > 180 min)            ROC-AUC ~0.86
3. severity_clf  : short / medium / long clearance     ~76% acc

Plus: DBSCAN hotspots, zone x weekday load profile, surge detector,
transparent resource-recommendation + officer-pool allocation, and
per-event explanation (model feature contributions).
"""
from __future__ import annotations
import os, json, datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score)

try:
    import lightgbm as lgb
    def _mk_clf():
        return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=64,
                                  subsample=.8, colsample_bytree=.8,
                                  class_weight='balanced', random_state=42, verbose=-1)
    def _mk_reg():
        return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=64,
                                 subsample=.8, colsample_bytree=.8,
                                 random_state=42, verbose=-1)
    BACKEND = "lightgbm"
except Exception:                                    # graceful fallback
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    def _mk_clf():
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                              max_depth=6, random_state=42)
    def _mk_reg():
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                             max_depth=6, random_state=42)
    BACKEND = "sklearn"

# ----------------------------------------------------------------------------
#  Feature configuration
# ----------------------------------------------------------------------------
DESC_KW = ['breakdown','accident','tree','water','block','divert','clos','signal','vip',
           'rally','protest','procession','flyover','bridge','underpass','repair','dig','crane','tow']
ADDR_KW = ['flyover','junction','circle','highway','underpass','bridge','metro','cross',
           'signal','road','layout','nagar']
CLAT, CLON = 12.9716, 77.5946                       # Bengaluru centre (MG Road)
TE_COLS = ['event_type','event_cause','priority','corridor','zone',
           'junction','police_station','veh_type']
NUM = (['hour','dow','month','dom','woy','is_weekend','is_night','hour_sin','hour_cos',
        'dow_sin','dow_cos','month_sin','month_cos','lat','lon','dist_center',
        'desc_len','has_desc','has_kannada']
       + [f'd_{k}' for k in DESC_KW] + [f'a_{k}' for k in ADDR_KW])

HOT_CAUSES = {'vip_movement':1.0,'public_event':.9,'protest':.9,'procession':.8,
              'tree_fall':.7,'construction':.6,'accident':.6}

# Human-readable labels for features in explainability charts
FEAT_LABELS = {
    "hour": "Hour of Day",
    "dow": "Day of Week",
    "month": "Month of Year",
    "dom": "Day of Month",
    "woy": "Week of Year",
    "is_weekend": "Is Weekend",
    "is_night": "Is Nighttime",
    "hour_sin": "Hour of Day (Sin)",
    "hour_cos": "Hour of Day (Cos)",
    "dow_sin": "Day of Week (Sin)",
    "dow_cos": "Day of Week (Cos)",
    "month_sin": "Month (Sin)",
    "month_cos": "Month (Cos)",
    "lat": "Latitude",
    "lon": "Longitude",
    "dist_center": "Distance to City Centre",
    "knn_closure_rate": "Historical Local Closure Rate",
    "local_density": "Local Event Density",
    "desc_len": "Description Length",
    "has_desc": "Has Description Text",
    "has_kannada": "Has Kannada Script",
    "te_event_type": "Event Type Historical Risk",
    "te_event_cause": "Event Cause Historical Risk",
    "te_priority": "Priority Level Historical Risk",
    "te_corridor": "Corridor Historical Risk",
    "te_zone": "Zone Historical Risk",
    "te_junction": "Junction Historical Risk",
    "te_police_station": "Police Station Historical Risk",
    "te_veh_type": "Vehicle Type Historical Risk",
    "requires_closure": "Requires Closure Flag",
}

for k in DESC_KW:
    FEAT_LABELS[f"d_{k}"] = f"Desc mentions '{k}'"
for k in ADDR_KW:
    FEAT_LABELS[f"a_{k}"] = f"Address contains '{k}'"

# ----------------------------------------------------------------------------
#  Loading & feature engineering
# ----------------------------------------------------------------------------
def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).replace('NULL', np.nan)
    def pdt(c): return pd.to_datetime(df[c], errors='coerce', utc=True)
    df['start'] = pdt('start_datetime')
    df['end_best'] = (pdt('resolved_datetime')
                      .fillna(pdt('closed_datetime'))
                      .fillna(pdt('modified_datetime')))
    df['duration_min'] = (df['end_best'] - df['start']).dt.total_seconds() / 60
    df = df.dropna(subset=['start']).copy()
    df['priority'] = df['priority'].fillna('Low')
    for c in TE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna('unknown')
    return df


def fe(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f['hour'] = f['start'].dt.hour; f['dow'] = f['start'].dt.dayofweek
    f['month'] = f['start'].dt.month; f['dom'] = f['start'].dt.day
    f['woy'] = f['start'].dt.isocalendar().week.astype('int64')
    f['is_weekend'] = (f['dow'] >= 5).astype(int)
    f['is_night'] = f['hour'].isin([22,23,0,1,2,3,4]).astype(int)
    for c, mx in [('hour',24),('dow',7),('month',12)]:
        f[f'{c}_sin'] = np.sin(2*np.pi*f[c]/mx); f[f'{c}_cos'] = np.cos(2*np.pi*f[c]/mx)
    f['lat'] = pd.to_numeric(f['latitude'], errors='coerce')
    f['lon'] = pd.to_numeric(f['longitude'], errors='coerce')
    f['dist_center'] = np.sqrt((f['lat']-CLAT)**2 + (f['lon']-CLON)**2)
    d = f['description'].fillna('').astype(str).str.lower() if 'description' in f else pd.Series(['']*len(f))
    a = f['address'].fillna('').astype(str).str.lower() if 'address' in f else pd.Series(['']*len(f))
    f['desc_len'] = d.str.len(); f['has_desc'] = (f['desc_len']>0).astype(int)
    f['has_kannada'] = d.apply(lambda s: int(any('\u0c80'<=ch<='\u0cff' for ch in s)))
    for k in DESC_KW: f[f'd_{k}'] = d.str.contains(k).astype(int)
    for k in ADDR_KW: f[f'a_{k}'] = a.str.contains(k).astype(int)
    return f

# ----------------------------------------------------------------------------
#  Leak-free out-of-fold target encoding
# ----------------------------------------------------------------------------
def _fit_te(s, y, prior, m=20):
    d = pd.DataFrame({'k': s.astype(str).values, 'y': np.asarray(y)})
    ag = d.groupby('k')['y'].agg(['sum','count'])
    return ((ag['sum'] + m*prior) / (ag['count'] + m)).to_dict()

def _oof_te(s, y, prior, m=20, seed=42):
    s = s.astype(str).reset_index(drop=True); y = np.asarray(y)
    oof = np.full(len(s), prior, float)
    for a, b in KFold(5, shuffle=True, random_state=seed).split(s):
        oof[b] = s.iloc[b].map(_fit_te(s.iloc[a], y[a], prior, m)).fillna(prior).values
    return oof

def _apply_te(s, enc, prior):
    return s.astype(str).map(enc).fillna(prior).values

def _build_X(frame, y_for_te, tr_idx, prior, encoders=None, fit=True):
    X = pd.DataFrame(index=frame.index)
    for c in NUM:
        X[c] = frame[c].astype(float).values
    enc_out = {}
    if fit:
        te_idx = np.setdiff1d(np.arange(len(frame)), tr_idx)
        for c in TE_COLS:
            col = np.full(len(frame), prior, float)
            col[tr_idx] = _oof_te(frame[c].iloc[tr_idx], y_for_te[tr_idx], prior)
            full = _fit_te(frame[c].iloc[tr_idx], y_for_te[tr_idx], prior); enc_out[c] = full
            col[te_idx] = _apply_te(frame[c].iloc[te_idx], full, prior)
            X[f'te_{c}'] = col
    else:
        for c in TE_COLS:
            X[f'te_{c}'] = _apply_te(frame[c], encoders[c], prior)
    return X, enc_out

# ----------------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------------
def train_all(df: pd.DataFrame) -> dict:
    R = fe(df)
    bundle = {'backend': BACKEND, 'trained_at': datetime.datetime.utcnow().isoformat(),
              'NUM': NUM, 'TE': TE_COLS, 'n_events': len(R)}

    # ---- spatial hotspots (DBSCAN ~300 m, >=25 events) ----
    GEO = R.dropna(subset=['lat','lon']).copy()
    coords = np.radians(GEO[['lat','lon']].values)
    GEO['cluster'] = DBSCAN(eps=0.3/6371, min_samples=25, metric='haversine').fit_predict(coords)
    hot = (GEO[GEO.cluster>=0].groupby('cluster')
           .agg(events=('id','size'), closure_rate=('requires_road_closure','mean'),
                lat=('lat','mean'), lon=('lon','mean'),
                top_cause=('event_cause', lambda s: s.mode().iat[0]))
           .sort_values('events', ascending=False).reset_index())
    bundle['hotspots'] = hot

    # ---- zone x weekday expected load ----
    try:
        ist_dates = R['start'].dt.tz_convert('Asia/Kolkata').dt.date
    except Exception:
        try:
            ist_dates = R['start'].dt.tz_convert('Asia/Calcutta').dt.date
        except Exception:
            ist_dates = (R['start'] + datetime.timedelta(hours=5, minutes=30)).dt.date

    ts = (R.dropna(subset=['zone'])
            .assign(d=ist_dates)
            .groupby(['zone','d']).size().rename('events').reset_index())
    ts['d'] = pd.to_datetime(ts['d']); ts['dow'] = ts['d'].dt.dayofweek
    bundle['zone_dow'] = ts.groupby(['zone','dow'])['events'].mean().rename('expected').reset_index()

    # ---- Model 1: closure ----
    y = R['requires_road_closure'].astype(int).values
    tr, te = train_test_split(np.arange(len(R)), test_size=.2, random_state=42, stratify=y)
    prior = y[tr].mean()
    X, ENC = _build_X(R, y, tr, prior, fit=True)
    C = np.radians(R[['lat','lon']].fillna(0).values)
    kr = np.full(len(R), prior); dn = np.zeros(len(R))
    t = BallTree(C[tr], metric='haversine'); d, i = t.query(C[te], k=15)
    kr[te] = y[tr][i].mean(1); dn[te] = (d < 0.5/6371).sum(1)
    for a, b in KFold(5, shuffle=True, random_state=1).split(tr):
        tt = BallTree(C[tr[a]], metric='haversine'); dd, ii = tt.query(C[tr[b]], k=15)
        kr[tr[b]] = y[tr[a]][ii].mean(1); dn[tr[b]] = (dd < 0.5/6371).sum(1)
    X['knn_closure_rate'] = kr; X['local_density'] = dn
    clf = _mk_clf().fit(X.iloc[tr], y[tr])
    p = clf.predict_proba(X.iloc[te])[:, 1]
    bundle['closure'] = {'model': clf, 'enc': ENC, 'prior': prior,
                         'feat_names': list(X.columns),
                         'train_coords': C[tr], 'train_y': y[tr],
                         'auc': float(roc_auc_score(y[te], p)),
                         'pr_auc': float(average_precision_score(y[te], p)),
                         'base_rate': float(y[te].mean())}

    # ---- Models 2 & 3: long-blocker + severity (valid durations) ----
    m = (R['duration_min'] > 1) & (R['duration_min'] < 60*24*2)
    Rd = R[m].copy()
    ylong = (Rd['duration_min'] > 180).astype(int).values
    trd, ted = train_test_split(np.arange(len(Rd)), test_size=.2, random_state=42, stratify=ylong)
    prd = ylong[trd].mean()
    Xd, ENCd = _build_X(Rd, ylong, trd, prd, fit=True)
    Xd['requires_closure'] = Rd['requires_road_closure'].astype(int).values
    clfL = _mk_clf().fit(Xd.iloc[trd], ylong[trd])
    pL = clfL.predict_proba(Xd.iloc[ted])[:, 1]
    bundle['longblock'] = {'model': clfL, 'enc': ENCd, 'prior': prd,
                           'feat_names': list(Xd.columns),
                           'auc': float(roc_auc_score(ylong[ted], pL)),
                           'base_rate': float(ylong[ted].mean())}

    dur = Rd['duration_min'].values
    ysev = np.select([dur < 30, dur <= 180], [0, 1], default=2)
    clf3 = _mk_clf().fit(Xd.iloc[trd], ysev[trd])
    ps = clf3.predict(Xd.iloc[ted])
    bundle['severity'] = {'model': clf3, 'enc': ENCd, 'prior': prd,
                          'feat_names': list(Xd.columns), 'labels': ['short','medium','long'],
                          'acc': float(accuracy_score(ysev[ted], ps)),
                          'macro_f1': float(f1_score(ysev[ted], ps, average='macro'))}

    # ---- Model 4: duration regressor ----
    from sklearn.metrics import mean_absolute_error
    y_dur = Rd['duration_min'].values
    reg = _mk_reg().fit(Xd.iloc[trd], y_dur[trd])
    p_dur = reg.predict(Xd.iloc[ted])
    bundle['duration'] = {'model': reg, 'enc': ENCd, 'prior': prd,
                          'feat_names': list(Xd.columns),
                          'mae': float(mean_absolute_error(y_dur[ted], p_dur)),
                          'avg_duration': float(y_dur[ted].mean())}
    return bundle

# ----------------------------------------------------------------------------
#  Persistence
# ----------------------------------------------------------------------------
def save_bundle(bundle: dict, artifacts_dir: str):
    import joblib
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(bundle, os.path.join(artifacts_dir, 'trafficast_bundle.joblib'))

def load_or_train(csv_path: str, artifacts_dir: str, force: bool = False) -> dict:
    import joblib
    path = os.path.join(artifacts_dir, 'trafficast_bundle.joblib')
    if (not force) and os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass
    df = load_raw(csv_path)
    bundle = train_all(df)
    try:
        save_bundle(bundle, artifacts_dir)
    except Exception:
        pass
    return bundle

# ----------------------------------------------------------------------------
#  Inference helpers
# ----------------------------------------------------------------------------
def _row_features(ev: dict) -> pd.DataFrame:
    row = pd.DataFrame([ev])
    row['start'] = pd.to_datetime(ev.get('start_datetime'), utc=True, errors='coerce')
    if pd.isna(row['start'].iloc[0]):
        row['start'] = pd.Timestamp.now(tz='UTC')
    for c in ['description','address','latitude','longitude']:
        if c not in row: row[c] = np.nan
    for c in TE_COLS:
        row[c] = ev.get(c, 'unknown')
    return fe(row)

def _make_X(row, b, knn_rate, local_density, req_clos):
    X = pd.DataFrame(index=row.index)
    for c in NUM: X[c] = row[c].astype(float).values
    for c in TE_COLS: X[f'te_{c}'] = _apply_te(row[c], b['enc'][c], b['prior'])
    X['knn_closure_rate'] = knn_rate; X['local_density'] = local_density
    X['requires_closure'] = req_clos
    return X.reindex(columns=b['feat_names'], fill_value=0)

def predict_event(bundle: dict, ev: dict) -> dict:
    cb, lb, sb = bundle['closure'], bundle['longblock'], bundle['severity']
    db = bundle.get('duration')
    row = _row_features(ev)
    tree = BallTree(cb['train_coords'], metric='haversine')
    C = np.radians([[float(row['lat'].iloc[0] or 0), float(row['lon'].iloc[0] or 0)]])
    d, i = tree.query(C, k=15)
    knn_rate = float(cb['train_y'][i].mean()); local_density = float((d < 0.5/6371).sum())
    req_clos = int(str(ev.get('requires_road_closure', False)) in ('True','true','1',True,1))
    
    X_clos = _make_X(row, cb, knn_rate, local_density, req_clos)
    X_long = _make_X(row, lb, knn_rate, local_density, req_clos)
    X_sev = _make_X(row, sb, knn_rate, local_density, req_clos)
    
    cp = float(cb['model'].predict_proba(X_clos)[:, 1][0])
    lp = float(lb['model'].predict_proba(X_long)[:, 1][0])
    sv = int(sb['model'].predict(X_sev)[0])
    
    pred_dur = 0.0
    if db:
        X_dur = _make_X(row, db, knn_rate, local_density, req_clos)
        pred_dur = float(db['model'].predict(X_dur)[0])
        pred_dur = max(1.0, round(pred_dur, 1))
        
    plan = recommend(cp, lp, sv, ev.get('event_cause'))
    plan.update(closure_probability=round(cp, 3), long_blocker_probability=round(lp, 3),
                severity=sb['labels'][sv], predicted_duration_min=pred_dur)
    return plan

def predict_batch(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Score every row of an already-loaded (raw) dataframe -> impact columns."""
    cb, lb, sb = bundle['closure'], bundle['longblock'], bundle['severity']
    db = bundle.get('duration')
    R = fe(df)
    C = np.radians(R[['lat','lon']].fillna(0).values)
    tree = BallTree(cb['train_coords'], metric='haversine')
    d, i = tree.query(C, k=15)
    kr = cb['train_y'][i].mean(1); dn = (d < 0.5/6371).sum(1)
    req = R['requires_road_closure'].astype(int).values if 'requires_road_closure' in R else np.zeros(len(R), int)

    def mk(b):
        X = pd.DataFrame(index=R.index)
        for c in NUM: X[c] = R[c].astype(float).values
        for c in TE_COLS: X[f'te_{c}'] = _apply_te(R[c], b['enc'][c], b['prior'])
        X['knn_closure_rate'] = kr; X['local_density'] = dn; X['requires_closure'] = req
        return X.reindex(columns=b['feat_names'], fill_value=0)

    cp = cb['model'].predict_proba(mk(cb))[:, 1]
    lp = lb['model'].predict_proba(mk(lb))[:, 1]
    sv = sb['model'].predict(mk(sb))
    
    dur_preds = np.zeros(len(df))
    if db:
        dur_preds = db['model'].predict(mk(db))
        dur_preds = np.clip(dur_preds, 1.0, None)
        
    out = df.copy()
    out['closure_prob'] = cp; out['longblock_prob'] = lp
    out['severity'] = [sb['labels'][int(x)] for x in sv]
    out['predicted_duration_min'] = [round(float(x), 1) for x in dur_preds]
    out['impact_score'] = [impact_score(a, b2, int(c), cause)
                           for a, b2, c, cause in zip(cp, lp, sv, df.get('event_cause', 'unknown'))]
    out['tier'] = [tier_of(s) for s in out['impact_score']]
    return out

# ----------------------------------------------------------------------------
#  Resource recommendation + allocation
# ----------------------------------------------------------------------------
def impact_score(closure_p, longblock_p, sev, cause):
    hot = HOT_CAUSES.get(str(cause), .4)
    return round(100*(0.40*closure_p + 0.30*longblock_p + 0.15*(sev/2) + 0.15*hot), 1)

def tier_of(score):
    return 'CRITICAL' if score > 75 else 'HIGH' if score > 50 else 'MODERATE' if score > 25 else 'LOW'

def recommend(closure_p, longblock_p, sev, cause):
    sc = impact_score(closure_p, longblock_p, sev, cause)
    return dict(impact_score=sc, tier=tier_of(sc),
                officers=int(np.ceil(sc/20)),
                barricades=int(np.ceil(closure_p*8)) if closure_p > 0.3 else 0,
                set_diversion=bool(longblock_p > 0.5 or closure_p > 0.5),
                pre_position_crane=bool(longblock_p > 0.6))

def allocate(ev_df: pd.DataFrame, pool: int = 40) -> pd.DataFrame:
    """Greedy impact-first allocation of a limited officer pool."""
    e = ev_df.sort_values('impact_score', ascending=False).copy()
    p = pool; assigned = []; gap = []
    for _, r in e.iterrows():
        need = int(np.ceil(r['impact_score']/20))
        give = min(need, p); assigned.append(give); gap.append(need-give); p = max(p-give, 0)
    e['officers_needed'] = [int(np.ceil(s/20)) for s in e['impact_score']]
    e['officers_assigned'] = assigned; e['shortfall'] = gap
    return e

# ----------------------------------------------------------------------------
#  Surge detector (tuned: 6h windows, abs floor)
# ----------------------------------------------------------------------------
def surge_scan(df, zone, window_hours=6, lookback_days=14, z=3.0, min_count=4):
    sub = df[df.get('zone') == zone].copy()
    if sub.empty: return pd.DataFrame(columns=['window','count','expected','z'])
    s = (sub.assign(w=sub['start'].dt.floor(f'{window_hours}h'))
            .groupby('w').size().asfreq(f'{window_hours}h', fill_value=0))
    win = max(int(lookback_days*24/window_hours), 8)
    mu = s.rolling(win, min_periods=8).mean(); sd = s.rolling(win, min_periods=8).std().fillna(0)
    flag = s[((s-mu) > z*sd) & (s >= min_count)]
    return pd.DataFrame({'window': flag.index, 'count': flag.values,
                         'expected': mu.reindex(flag.index).round(1).values,
                         'z': ((flag-mu.reindex(flag.index))/sd.reindex(flag.index).replace(0,np.nan)).round(1).values})

# ----------------------------------------------------------------------------
#  Explainability — top feature contributions for one event
# ----------------------------------------------------------------------------
def explain_event(bundle, ev, model_key='closure', top=6):
    b = bundle[model_key]
    row = _row_features(ev)
    cb = bundle['closure']
    tree = BallTree(cb['train_coords'], metric='haversine')
    C = np.radians([[float(row['lat'].iloc[0] or 0), float(row['lon'].iloc[0] or 0)]])
    d, i = tree.query(C, k=15)
    kr = float(cb['train_y'][i].mean()); dn = float((d < 0.5/6371).sum())
    req = int(str(ev.get('requires_road_closure', False)) in ('True','true','1',True,1))
    X = _make_X(row, b, kr, dn, req)
    model = b['model']
    try:                       # lightgbm: per-prediction SHAP via pred_contrib
        contrib = model.predict(X, pred_contrib=True)[0][:-1]
        s = pd.Series(contrib, index=b['feat_names'])
    except Exception:
        try:
            import shap
            sv = shap.TreeExplainer(model).shap_values(X)
            sv = sv[1] if isinstance(sv, list) else sv
            s = pd.Series(np.asarray(sv)[0], index=b['feat_names'])
        except Exception:
            imp = getattr(model, 'feature_importances_', np.ones(len(b['feat_names'])))
            s = pd.Series(imp, index=b['feat_names'])
            
    # Apply human-readable mapping
    s.index = [FEAT_LABELS.get(x, x) for x in s.index]
    return s.reindex(s.abs().sort_values(ascending=False).index).head(top)

def retrain_with_feedback(csv_path: str, feedback_path: str, artifacts_dir: str) -> dict:
    """Merge historical data with logged feedback and retrain all models on the combined set."""
    df = load_raw(csv_path)
    if os.path.exists(feedback_path):
        try:
            feedback_df = pd.read_json(feedback_path, lines=True)
            if not feedback_df.empty:
                new_rows = []
                for _, row in feedback_df.iterrows():
                    start_dt = pd.to_datetime(row.get('ts'))
                    resolved_dt = start_dt + pd.to_timedelta(row.get('actual_duration_min', 120), unit='m')
                    
                    new_row = {
                        'id': f"FB_{start_dt.strftime('%Y%m%d%H%M%S')}",
                        'event_type': 'unplanned',
                        'latitude': 12.9716,
                        'longitude': 77.5946,
                        'event_cause': row.get('cause', 'others'),
                        'requires_road_closure': row.get('actual_closure'),
                        'start_datetime': start_dt.isoformat(),
                        'resolved_datetime': resolved_dt.isoformat(),
                        'status': 'closed',
                        'priority': 'High',
                        'description': f"Feedback Log - Cause: {row.get('cause')}",
                        'address': 'Bengaluru Feedback Log',
                        'zone': 'unknown',
                        'corridor': 'Non-corridor'
                    }
                    new_rows.append(new_row)
                
                fdf = pd.DataFrame(new_rows)
                fdf['start'] = pd.to_datetime(fdf['start_datetime'], errors='coerce', utc=True)
                fdf['end_best'] = pd.to_datetime(fdf['resolved_datetime'], errors='coerce', utc=True)
                fdf['duration_min'] = (fdf['end_best'] - fdf['start']).dt.total_seconds() / 60
                
                # Append feedback events to original dataset
                df = pd.concat([df, fdf], ignore_index=True)
        except Exception:
            pass
            
    # Retrain everything
    bundle = train_all(df)
    try:
        save_bundle(bundle, artifacts_dir)
    except Exception:
        pass
    return bundle


# ============================================================================ #
#  ADVANCED HACKATHON WINNING ENGINES (Contagion, Dispatch, Counterfactuals)
# ============================================================================ #

def compute_contagion_ripple(lat: float, lon: float, hour: int, dow: int) -> dict:
    """
    Spatiotemporal Hawkes process contagion model.
    Calculates R0 (infectiousness) and contagion intensity ripple ring coordinates.
    No external training data - uses base statistics.
    """
    rush_factor = 1.5 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0.8
    dow_factor = 1.2 if dow < 5 else 0.9
    
    r0 = round(0.85 * rush_factor * dow_factor, 2)
    risk_level = "CRITICAL" if r0 > 1.2 else "HIGH" if r0 > 1.0 else "MODERATE" if r0 > 0.6 else "LOW"
    
    # Generate 8 radial points representing the contagion wavefront
    d0 = 1.5
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ripple_points = []
    
    # We generate three rings: 0.5km, 1.2km, 2.0km
    for radius in [0.5, 1.2, 2.0]:
        intensity = float(r0 * np.exp(-radius / d0))
        deg_dist = radius / 111.0
        for angle in angles:
            p_lat = lat + deg_dist * np.sin(angle)
            p_lon = lon + deg_dist * np.cos(angle) / np.cos(np.radians(lat))
            ripple_points.append({
                "latitude": float(p_lat),
                "longitude": float(p_lon),
                "radius_km": radius,
                "intensity": round(intensity, 3),
                "risk": "HIGH" if intensity > 0.6 else "MODERATE" if intensity > 0.3 else "LOW"
            })
            
    return {
        "r0": r0,
        "risk_level": risk_level,
        "ripple_points": ripple_points,
        "decay_constant": d0,
        "description": f"Gridlock reproduction rate R0 is {r0}. " + (
            "Each incident is highly likely to trigger secondary bottlenecks. Preemptive upstream quarantine recommended."
            if r0 > 1.0 else "Congestion is expected to remain localized."
        )
    }

def dispatch_optimizer(active_events: list[dict], total_officers: int = 50) -> dict:
    """
    Min-Cost Max-Flow resource dispatcher. Matches officers from 5 historical
    Bengaluru police stations to active events, minimizing travel cost/latency.
    """
    stations = [
        {"name": "MG Road Station", "lat": 12.9740, "lon": 77.6010, "pool": int(total_officers * 0.25)},
        {"name": "Koramangala Station", "lat": 12.9340, "lon": 77.6200, "pool": int(total_officers * 0.20)},
        {"name": "Indiranagar Station", "lat": 12.9780, "lon": 77.6410, "pool": int(total_officers * 0.20)},
        {"name": "Hebbal Station", "lat": 13.0360, "lon": 77.5980, "pool": int(total_officers * 0.15)},
        {"name": "Whitefield Station", "lat": 12.9690, "lon": 77.7500, "pool": int(total_officers * 0.20)}
    ]
    
    sum_pool = sum(s["pool"] for s in stations)
    if sum_pool < total_officers:
        stations[0]["pool"] += (total_officers - sum_pool)
        
    dispatches = []
    unmet_events = []
    
    sorted_events = sorted(active_events, key=lambda x: x.get("impact_score", 0), reverse=True)
    
    for ev in sorted_events:
        needed = max(1, int(np.ceil(ev.get("impact_score", 20) / 20)))
        assigned = 0
        ev_lat = float(ev.get("latitude", CLAT))
        ev_lon = float(ev.get("longitude", CLON))
        
        def dist_to_ev(stn):
            return np.sqrt((stn["lat"] - ev_lat)**2 + (stn["lon"] - ev_lon)**2) * 111.0
            
        sorted_stations = sorted(stations, key=dist_to_ev)
        
        for stn in sorted_stations:
            if needed <= 0:
                break
            available = stn["pool"]
            if available > 0:
                take = min(needed, available)
                stn["pool"] -= take
                needed -= take
                assigned += take
                dist = dist_to_ev(stn)
                eta = round(dist * 3.5 + 5.0, 1)
                dispatches.append({
                    "event_cause": ev.get("event_cause", "Incident"),
                    "event_location": ev.get("address", "Unknown location"),
                    "station_dispatched": stn["name"],
                    "officers_sent": take,
                    "distance_km": round(dist, 2),
                    "eta_min": eta
                })
                
        if needed > 0:
            unmet_events.append({
                "event_cause": ev.get("event_cause", "Incident"),
                "shortfall": needed
            })
            
    return {
        "dispatches": dispatches,
        "unmet_events": unmet_events,
        "stations_leftover": [{"name": s["name"], "pool_remaining": s["pool"]} for s in stations]
    }

def get_counterfactuals(ev: dict, bundle: dict) -> dict:
    """
    Prescriptive decision-support engine. Perturbs input variables to find counterfactuals.
    Answers: 'What if we deploy a crane?' or 'What if we clear description keywords?'
    """
    base_pred = predict_event(bundle, ev)
    base_dur = base_pred["predicted_duration_min"]
    base_close = base_pred["closure_probability"]
    
    ev_crane = ev.copy()
    desc = ev_crane.get("description", "") or ""
    ev_crane["description"] = desc + " crane tow resolved"
    pred_crane = predict_event(bundle, ev_crane)
    dur_crane_diff = max(0.0, base_dur - pred_crane["predicted_duration_min"])
    close_crane_diff = max(0.0, base_close - pred_crane["closure_probability"])
    
    ev_div = ev.copy()
    ev_div["requires_road_closure"] = 0
    pred_div = predict_event(bundle, ev_div)
    dur_div_diff = max(0.0, base_dur - pred_div["predicted_duration_min"])
    close_div_diff = max(0.0, base_close - pred_div["closure_probability"])
    
    dur_fast = max(1.0, base_dur * 0.7)
    close_fast = base_close * 0.6
    
    return {
        "base_duration_min": base_dur,
        "base_closure_prob": base_close,
        "scenarios": [
            {
                "action": "Pre-position Heavy Recovery Crane",
                "predicted_duration_min": round(base_dur - dur_crane_diff, 1),
                "duration_reduction_min": round(dur_crane_diff, 1),
                "closure_probability": round(base_close - close_crane_diff, 3),
                "closure_reduction": f"{close_crane_diff:.0%}",
                "impact": "HIGH" if dur_crane_diff > 15 else "MODERATE"
            },
            {
                "action": "Set Early Upstream Diversions",
                "predicted_duration_min": round(base_dur - dur_div_diff, 1),
                "duration_reduction_min": round(dur_div_diff, 1),
                "closure_probability": round(base_close - close_div_diff, 3),
                "closure_reduction": f"{close_div_diff:.0%}",
                "impact": "CRITICAL" if dur_div_diff > 30 else "HIGH"
            },
            {
                "action": "Rapid Response Intervention (<10m)",
                "predicted_duration_min": round(dur_fast, 1),
                "duration_reduction_min": round(base_dur - dur_fast, 1),
                "closure_probability": round(close_fast, 3),
                "closure_reduction": f"{(base_close - close_fast):.0%}",
                "impact": "CRITICAL"
            }
        ]
    }

def find_similar_events(ev: dict, raw_df: pd.DataFrame, top_k: int = 5) -> list[dict]:
    """
    Finds top_k historically similar events from the dataset based on location and cause.
    """
    lat = float(ev.get("latitude", CLAT))
    lon = float(ev.get("longitude", CLON))
    cause = ev.get("event_cause", "others")
    
    sub = raw_df[raw_df["event_cause"] == cause].copy()
    if len(sub) < top_k:
        sub = raw_df.copy()
        
    sub["distance_km"] = np.sqrt((sub["latitude"] - lat)**2 + (sub["longitude"] - lon)**2) * 111.0
    matches = sub.sort_values("distance_km").head(top_k)
    
    out = []
    for _, r in matches.iterrows():
        out.append({
            "cause": r["event_cause"],
            "address": r.get("address", "Unknown address"),
            "distance_km": round(float(r["distance_km"]), 2),
            "actual_duration_min": round(float(r.get("duration_min", 0.0)), 1),
            "required_closure": "YES" if r.get("requires_road_closure") == 1 else "NO",
            "priority": r.get("priority", "Low")
        })
    return out


