"""Funnel Sandbox — Zerve Datathon"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Funnel Sandbox", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --bg:#050607;
  --panel:#0b0d10;
  --panel-2:#101318;
  --line:#242932;
  --line-soft:#171b21;
  --text:#f2f5f8;
  --muted:#8b949e;
  --dim:#5f6873;
  --blue:#2d7dd2;
  --green:#44d17a;
  --yellow:#ffd166;
}

html,body,[class*="css"]{font-family:Inter,Segoe UI,Arial,sans-serif}
[data-testid="stAppViewContainer"],[data-testid="stHeader"],section[data-testid="stMain"]{background:var(--bg)}
[data-testid="stDecoration"]{display:none}
.block-container{padding-top:4.25rem;max-width:1220px;padding-bottom:2.25rem}
h1,h2,h3,h4{color:var(--text)!important;letter-spacing:0}
p,label,div,span{color:var(--muted)}
hr{border-color:var(--line)!important;margin:18px 0!important}

.metro-rail{border-left:5px solid var(--blue);padding-left:18px;margin-bottom:18px}
.metro-title{color:var(--text);font-size:42px;font-weight:800;line-height:1;margin:0}
.metro-subtitle{color:var(--muted);font-size:13px;margin-top:8px}
.panel-title{color:var(--text);font-size:13px;font-weight:800;letter-spacing:1.3px;text-transform:uppercase;margin-bottom:12px}
.section-note{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:1.2px;font-weight:700}

[data-testid="stMetric"]{background:var(--panel-2);border:1px solid var(--line);padding:14px 15px;border-radius:0}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:10px!important;text-transform:uppercase;letter-spacing:1.35px;font-weight:800}
[data-testid="stMetricValue"]{color:var(--text)!important;font-size:1.75rem!important;font-weight:800!important}
[data-testid="stMetricDelta"]{font-size:11px!important}

.stButton>button{background:var(--panel-2)!important;border:1px solid var(--line)!important;color:var(--text)!important;
  border-radius:0!important;font-size:12px!important;font-weight:700!important;padding:10px 12px!important;text-align:left!important;
  min-height:40px!important;box-shadow:none!important;transition:border-color .12s ease,background .12s ease}
.stButton>button:hover{border-color:var(--blue)!important;background:#121923!important;color:#fff!important}
.stButton>button:focus{box-shadow:0 0 0 1px var(--blue)!important}
.stSelectbox>div>div{background:var(--panel)!important;border-color:var(--line)!important;border-radius:0!important;color:var(--text)!important}

.signal-row{display:grid;grid-template-columns:1fr auto;gap:16px;align-items:center;padding:10px 0;border-top:1px solid var(--line-soft)}
.signal-label{color:var(--muted);font-size:12px;font-weight:600}
.signal-val{color:var(--text);font-size:13px;font-weight:800}
.stage-badge{display:inline-block;padding:5px 10px;border-radius:0;font-size:10px;
  font-weight:800;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:12px}
.action-pill{display:inline-block;background:#0d1712;border:1px solid #23543a;color:var(--green);
  border-radius:0;font-size:10px;font-weight:800;padding:4px 7px;margin:3px;text-transform:uppercase;letter-spacing:.8px}
.diff-row{display:grid;grid-template-columns:1fr auto;gap:12px;border-top:1px solid var(--line-soft);padding:8px 0}
.diff-pos{color:var(--green);font-size:12px;font-weight:800}
.diff-label{color:var(--muted);font-size:12px;font-weight:600}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent
REPORTS  = HERE / "reports"
DATASETS = HERE.parent / "datasets"

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_RATE  = 0.0184
FEAT_CACHE = REPORTS / "user_features.pkl"

DEPLOY_EVENTS = {"notebook_deployment_usage_tracked","notebook_deployed","app_deployed","deployment_created"}
BLOCK_CREATE  = {"block_create","block_created"}
RUN_BLOCK     = {"run_block","block_run"}
RUN_ALL       = {"run_all_blocks","notebook_run"}
FILE_UPLOAD   = {"file_uploaded","file_created","files_upload"}
SHARE_EVENTS  = {"report_shared","canvas_shared"}

STAGE_COLOR = {
    "New":                  "#5f6368",
    "Exploring":            "#2d7dd2",
    "Created Content":      "#8f6ed5",
    "AI Engaged":           "#00a88f",
    "Workflow Builder":     "#ff8a3d",
    "Credit Active":        "#ff5f57",
    "Consistently Engaged": "#ffd166",
    "At Risk":              "#d1495b",
    "Upgraded":             "#44d17a",
}

TIERS = [
    (BASE_RATE * 1.5, "Low Intent",  "#6f747d"),
    (BASE_RATE * 4,   "Moderate",    "#2d7dd2"),
    (BASE_RATE * 10,  "High Intent", "#ffd166"),
    (2.0,             "Very High",   "#44d17a"),
]

EVENT_ACTIONS = {
    "Run AI generation": {"ai_gen_count":1,"ai_gens_last_7d":1,"ai_gens_last_14d":1,
                          "total_events":1,"events_last_7d":1,"events_last_14d":1,
                          "events_last_30d":1,"ai_tokens_total":3500},
    "Deploy notebook":   {"deployment_count":1,"notebook_deploy_usage":1,"deploys_last_14d":1,
                          "total_events":1,"events_last_7d":1,"events_last_14d":1,"events_last_30d":1},
    "Hit credit limit":  {"credit_pressure_count":1,
                          "total_events":1,"events_last_7d":1,"events_last_14d":1,"events_last_30d":1},
    "Create canvas":     {"canvas_create_count":1,
                          "total_events":1,"events_last_7d":1,"events_last_14d":1,"events_last_30d":1},
    "Run block":         {"run_block_count":1,
                          "total_events":1,"events_last_7d":1,"events_last_14d":1,"events_last_30d":1},
    "Create block":      {"block_create_count":1,
                          "total_events":1,"events_last_7d":1,"events_last_14d":1,"events_last_30d":1},
    "Upload file":       {"files_upload_count":1,
                          "total_events":1,"events_last_7d":1,"events_last_14d":1,"events_last_30d":1},
    "Return next day":   {"active_days":1,"active_days_7d":1,"active_days_14d":1,
                          "active_days_30d":1,"tenure_days":1},
}

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m  = joblib.load(REPORTS / "xgb_model.pkl")
    fc = joblib.load(REPORTS / "feature_cols.pkl")
    return m, fc

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["person_id", "ts"])
    SNAP = df["ts"].max()

    upgrades  = df[df["event"] == "subscription_upgraded"].groupby("person_id")["ts"].min()
    df["snap"]= df["person_id"].map(upgrades).fillna(SNAP)
    safe      = df[df["ts"] < df["snap"]].copy()
    g         = safe.groupby("person_id")

    feats = pd.DataFrame({
        "total_events":          g["event"].count(),
        "active_days":           g["ts"].apply(lambda x: x.dt.date.nunique()),
        "unique_event_types":    g["event"].nunique(),
        "pageview_count":        g["event"].apply(lambda x: (x=="$pageview").sum()),
        "exception_count":       g["event"].apply(lambda x: (x=="$exception").sum()),
        "ai_gen_count":          g["event"].apply(lambda x: (x=="$ai_generation").sum()),
        "run_block_count":       g["event"].apply(lambda x: x.isin(RUN_BLOCK).sum()),
        "run_all_blocks_count":  g["event"].apply(lambda x: x.isin(RUN_ALL).sum()),
        "block_create_count":    g["event"].apply(lambda x: x.isin(BLOCK_CREATE).sum()),
        "canvas_create_count":   g["event"].apply(lambda x: (x=="canvas_create").sum()),
        "files_upload_count":    g["event"].apply(lambda x: x.isin(FILE_UPLOAD).sum()),
        "report_share_count":    g["event"].apply(lambda x: x.isin(SHARE_EVENTS).sum()),
        "deployment_count":      g["event"].apply(lambda x: x.isin(DEPLOY_EVENTS).sum()),
        "notebook_deploy_usage": g["event"].apply(lambda x: (x=="notebook_deployment_usage_tracked").sum()),
        "credit_pressure_count": g["event"].apply(lambda x: (x=="credits_remaining").sum()),
    })

    ai = safe[safe["event"] == "$ai_generation"].copy()
    if "properties.$ai_input_tokens" in ai.columns:
        ai["tkn"]                     = ai["properties.$ai_input_tokens"].fillna(0) + ai["properties.$ai_output_tokens"].fillna(0)
        feats["ai_tokens_total"]      = ai.groupby("person_id")["tkn"].sum().reindex(feats.index, fill_value=0)
        feats["ai_tokens_max_single"] = ai.groupby("person_id")["tkn"].max().reindex(feats.index, fill_value=0)
    else:
        feats["ai_tokens_total"] = feats["ai_tokens_max_single"] = 0

    feats["ai_latency_avg"] = (
        ai.groupby("person_id")["properties.$ai_latency"].mean().reindex(feats.index, fill_value=0)
        if "properties.$ai_latency" in ai.columns else 0
    )

    feats["tenure_days"]        = g["ts"].apply(lambda x: max((x.max()-x.min()).total_seconds()/86400+1,1))
    feats["session_count"]      = g["ts"].apply(lambda x: int((x.sort_values().diff().dt.total_seconds().fillna(9999)>1800).sum()+1))
    feats["avg_inter_event_min"]= g["ts"].apply(lambda x: x.sort_values().diff().dt.total_seconds().dropna().mean()/60 if len(x)>1 else 0)

    for col in ["person_properties.purpose","person_properties.role","person_properties.work_type","person_properties.source"]:
        feats[col] = (
            df.groupby("person_id")[col].first().astype("category").cat.codes.reindex(feats.index, fill_value=-1)
            if col in df.columns else 0
        )

    for days, sfx in [(7,"7d"),(14,"14d"),(30,"30d")]:
        w = safe[safe["ts"] >= SNAP - pd.Timedelta(days=days)]
        feats[f"events_last_{sfx}"]  = w.groupby("person_id")["event"].count().reindex(feats.index, fill_value=0)
        feats[f"active_days_{sfx}"]  = w.groupby("person_id")["ts"].apply(lambda x: x.dt.date.nunique()).reindex(feats.index, fill_value=0)
        if sfx != "30d":
            feats[f"ai_gens_last_{sfx}"] = w.groupby("person_id")["event"].apply(lambda x: (x=="$ai_generation").sum()).reindex(feats.index, fill_value=0)
        if sfx == "14d":
            feats["deploys_last_14d"] = w.groupby("person_id")["event"].apply(lambda x: x.isin(DEPLOY_EVENTS).sum()).reindex(feats.index, fill_value=0)

    feats["events_per_day"]       = feats["total_events"]   / feats["tenure_days"].clip(lower=1)
    feats["ai_pct_events"]        = feats["ai_gen_count"]   / feats["total_events"].clip(lower=1)
    feats["has_used_ai"]          = (feats["ai_gen_count"]          > 0).astype(int)
    feats["has_deployed"]         = (feats["deployment_count"]       > 0).astype(int)
    feats["has_hit_credit_limit"] = (feats["credit_pressure_count"]  > 0).astype(int)
    feats["events_per_session"]   = feats["total_events"]   / feats["session_count"].clip(lower=1)
    feats["pct_activity_last_7d"] = feats["events_last_7d"] / feats["total_events"].clip(lower=1)
    feats["is_ramping"]           = (feats["events_last_7d"] > feats["events_last_14d"] / 2).astype(int)

    ai_first  = safe[safe["event"]=="$ai_generation"].groupby("person_id")["ts"].min()
    first_evt = g["ts"].min()
    feats["days_to_first_ai"] = ((ai_first - first_evt).dt.total_seconds()/86400).reindex(feats.index, fill_value=-1)
    feats["upgraded"]         = feats.index.isin(upgrades.index).astype(int)
    return feats

@st.cache_data(show_spinner=False)
def get_features() -> pd.DataFrame:
    if FEAT_CACHE.exists():
        return pd.read_pickle(FEAT_CACHE)
    with st.spinner("Building user features — first run only (~2 min)…"):
        feats = build_features(pd.read_csv(DATASETS / "zerve_events.csv"))
        feats.to_pickle(FEAT_CACHE)
    return feats

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_tier(p: float):
    for thr, label, color in TIERS:
        if p < thr:
            return label, color
    return "Very High", "#2ecc71"

def funnel_stage(f: dict) -> str:
    if f.get("upgraded", 0):          return "Upgraded"
    if f.get("total_events", 0) < 3:  return "New"
    stage = "Exploring"
    if f.get("canvas_create_count",0)>0 or f.get("block_create_count",0)>0 or f.get("files_upload_count",0)>0:
        stage = "Created Content"
    if f.get("ai_gen_count",0)>0:
        stage = "AI Engaged"
    if f.get("run_block_count",0)>2 or f.get("deployment_count",0)>0 or f.get("run_all_blocks_count",0)>0:
        stage = "Workflow Builder"
    if f.get("credit_pressure_count",0)>0:
        stage = "Credit Active"
    if f.get("active_days",0) >= 7:
        stage = "Consistently Engaged"
    if stage not in ("New","Exploring") and f.get("events_last_14d",0)==0:
        stage = "At Risk"
    return stage

def score(f: dict, model, feat_cols) -> float:
    row = {c: 0.0 for c in feat_cols}
    row.update({k: float(v) for k, v in f.items() if k in row})
    row["has_used_ai"]          = 1.0 if row["ai_gen_count"]         > 0 else 0.0
    row["has_deployed"]         = 1.0 if row["deployment_count"]      > 0 else 0.0
    row["has_hit_credit_limit"] = 1.0 if row["credit_pressure_count"] > 0 else 0.0
    row["ai_pct_events"]        = row["ai_gen_count"]   / max(row["total_events"],  1)
    row["events_per_day"]       = row["total_events"]   / max(row["tenure_days"],   1)
    row["events_per_session"]   = row["total_events"]   / max(row["session_count"], 1)
    row["pct_activity_last_7d"] = row["events_last_7d"] / max(row["total_events"],  1)
    e7, e14 = row["events_last_7d"], row["events_last_14d"]
    row["is_ramping"]           = 1.0 if (e14 > 0 and e7 > e14/2) else 0.0
    return model.predict_proba(pd.DataFrame([row])[feat_cols])[0][1]

def stage_badge(stage: str) -> str:
    c = STAGE_COLOR.get(stage, "#555")
    return f'<span class="stage-badge" style="background:{c}22;color:{c};border:1px solid {c}55">{stage}</span>'

def signal_row(label, val):
    st.markdown(
        f'<div class="signal-row"><span class="signal-label">{label}</span>'
        f'<span class="signal-val">{val}</span></div>',
        unsafe_allow_html=True
    )

def prob_gauge(orig_p: float, sim_p: float):
    fig, ax = plt.subplots(figsize=(5, 0.68))
    fig.patch.set_facecolor("#0b0d10"); ax.set_facecolor("#0b0d10")
    ax.set_xlim(0, 0.25); ax.set_ylim(0, 1); ax.axis("off")
    bounds = [0] + [t[0] for t in TIERS[:-1]] + [0.25]
    for i, color in enumerate([t[2] for t in TIERS]):
        ax.barh(0.55, bounds[i+1]-bounds[i], left=bounds[i], height=0.24, color=color, alpha=0.28)
    for p, col, y, lbl in [(orig_p,"#6f747d",0.32,"baseline"),(sim_p,"#2d7dd2",0.62,"simulated")]:
        px = min(p, 0.249)
        ax.plot([px,px],[y-0.12,y+0.12], color=col, lw=2.5, solid_capstyle="round")
        ax.text(px, y-0.18 if lbl=="baseline" else y+0.18,
                f"{lbl} {p*100:.2f}%", color=col, ha="center",
                va="top" if lbl=="baseline" else "bottom", fontsize=8, fontweight="bold")
    plt.tight_layout(pad=0)
    return fig

# ── App ───────────────────────────────────────────────────────────────────────
model, feat_cols = load_model()
all_feats        = get_features()

non_upg = all_feats[(all_feats["upgraded"]==0) & (all_feats["total_events"]>=5)].copy()
non_upg["_stage"] = non_upg.apply(lambda r: funnel_stage(r.to_dict()), axis=1)

STAGES_SHOW = ["Exploring","Created Content","AI Engaged",
               "Workflow Builder","Credit Active","Consistently Engaged","At Risk"]
options = {}
for s in STAGES_SHOW:
    pool = non_upg[non_upg["_stage"]==s]
    for uid in pool.sample(min(4,len(pool)), random_state=42).index:
        options[f"{s}  ·  {uid[:12]}…"] = uid

# Header
hc1, hc2 = st.columns([2.2, 1], gap="large")
with hc1:
    st.markdown(
        """
<div class="metro-rail">
  <div class="metro-title">Funnel Sandbox</div>
  <div class="metro-subtitle">Pick a real user, add hypothetical product events, and watch upgrade intent move.</div>
</div>
""",
        unsafe_allow_html=True,
    )
with hc2:
    st.markdown('<div class="section-note">User profile</div>', unsafe_allow_html=True)
    selected_label = st.selectbox("User", list(options.keys()), label_visibility="collapsed")

selected_uid = options[selected_label]
real_feats   = all_feats.loc[selected_uid].to_dict()

if st.session_state.get("uid") != selected_uid:
    st.session_state.uid     = selected_uid
    st.session_state.sim     = real_feats.copy()
    st.session_state.actions = []

left, right = st.columns([1,1], gap="large")

# Left — real profile
with left:
    st.markdown('<div class="panel-title">Current Profile</div>', unsafe_allow_html=True)
    orig_p     = score(real_feats, model, feat_cols)
    orig_stage = funnel_stage(real_feats)
    orig_lift  = orig_p / BASE_RATE

    st.markdown(stage_badge(orig_stage), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Upgrade probability", f"{orig_p*100:.2f}%")
    c2.metric("Lift vs avg",         f"{orig_lift:.1f}x")

    st.markdown('<div class="section-note" style="margin-top:16px;margin-bottom:4px">Behavior signals</div>', unsafe_allow_html=True)
    for label, key in [
        ("AI generations",    "ai_gen_count"),
        ("Total events",      "total_events"),
        ("Active days",       "active_days"),
        ("Deployments",       "deployment_count"),
        ("Credit limit hits", "credit_pressure_count"),
        ("Tenure days",       "tenure_days"),
        ("Events last 7d",    "events_last_7d"),
    ]:
        signal_row(label, int(real_feats.get(key, 0)))

# Right — simulation
with right:
    st.markdown('<div class="panel-title">Simulate Next Events</div>', unsafe_allow_html=True)
    sim_feats = st.session_state.sim
    sim_p     = score(sim_feats, model, feat_cols)
    sim_stage = funnel_stage(sim_feats)
    delta_p   = sim_p - orig_p

    st.markdown(stage_badge(sim_stage), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Simulated probability", f"{sim_p*100:.2f}%",  f"{delta_p*100:+.2f}%")
    c2.metric("Lift vs avg",           f"{sim_p/BASE_RATE:.1f}x",
              f"{(sim_p-orig_p)/BASE_RATE:+.1f}x" if delta_p else None)

    st.pyplot(prob_gauge(orig_p, sim_p), use_container_width=True)
    plt.close("all")

    st.markdown('<div class="section-note" style="margin-top:14px;margin-bottom:8px">Event actions</div>', unsafe_allow_html=True)
    btn_cols = st.columns(2)
    for i, (action, deltas) in enumerate(EVENT_ACTIONS.items()):
        if btn_cols[i%2].button(f"+  {action}", key=f"btn_{action}", use_container_width=True):
            for feat, delta in deltas.items():
                st.session_state.sim[feat] = st.session_state.sim.get(feat, 0) + delta
            st.session_state.actions.append(action)
            st.rerun()

    if st.button("Reset to baseline", use_container_width=True):
        st.session_state.sim     = real_feats.copy()
        st.session_state.actions = []
        st.rerun()

    if st.session_state.actions:
        pills = "".join(f'<span class="action-pill">{a}</span>' for a in st.session_state.actions)
        st.markdown(f"<div style='margin-top:8px'>{pills}</div>", unsafe_allow_html=True)

    if delta_p > 0.0001:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-note" style="margin-bottom:4px">Changed signals</div>', unsafe_allow_html=True)
        WATCH = ["ai_gen_count","deployment_count","credit_pressure_count",
                 "total_events","active_days","events_last_7d","block_create_count"]
        for feat in WATCH:
            diff = sim_feats.get(feat,0) - real_feats.get(feat,0)
            if diff > 0:
                st.markdown(
                    f'<div class="diff-row"><span class="diff-label">{feat}</span>'
                    f'<span class="diff-pos">+{int(diff)}</span></div>',
                    unsafe_allow_html=True
                )
