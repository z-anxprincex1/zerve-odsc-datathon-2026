import joblib, pandas as pd

model     = joblib.load('reports/xgb_model.pkl')
feat_cols = joblib.load('reports/feature_cols.pkl')

personas = {
    'New User':   dict(tenure_days=2,  total_events=6,   ai_gen_count=0,  deployment_count=0, credit_pressure_count=0, has_used_ai=0, has_deployed=0, has_hit_credit_limit=0, events_last_7d=6,  events_last_14d=6,  active_days_30d=2),
    'AI User':    dict(tenure_days=21, total_events=220, ai_gen_count=25, deployment_count=0, credit_pressure_count=2, has_used_ai=1, has_deployed=0, has_hit_credit_limit=0, events_last_7d=45, events_last_14d=80, active_days_30d=18),
    'Power User': dict(tenure_days=35, total_events=480, ai_gen_count=80, deployment_count=6, credit_pressure_count=8, has_used_ai=1, has_deployed=1, has_hit_credit_limit=1, events_last_7d=90, events_last_14d=160, active_days_30d=25),
    'At Risk':    dict(tenure_days=45, total_events=350, ai_gen_count=30, deployment_count=2, credit_pressure_count=3, has_used_ai=1, has_deployed=1, has_hit_credit_limit=0, events_last_7d=2,  events_last_14d=5,  active_days_30d=5),
}

for name, vals in personas.items():
    row = {c: 0.0 for c in feat_cols}
    row.update({k: float(v) for k, v in vals.items() if k in row})
    row['pct_activity_last_7d'] = row.get('events_last_7d', 0) / max(row.get('total_events', 1), 1)
    e7, e14 = row.get('events_last_7d', 0), row.get('events_last_14d', 0)
    row['is_ramping'] = float(int(e7 > e14 / 2) if e14 > 0 else 0)
    df = pd.DataFrame([row])[feat_cols]
    p = model.predict_proba(df)[0][1]
    print(f'{name:20s}: {p*100:.3f}%')
