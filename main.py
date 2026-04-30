import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Upgrade Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Metro dark CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0a !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
[data-testid="stSidebar"] {
    background: #0f0f0f !important;
    border-right: 1px solid #1a1a1a;
}
[data-testid="stSidebar"] section { padding-top: 1.5rem; }
h1, h2, h3, h4 { color: #ffffff !important; font-weight: 700; letter-spacing: -0.5px; }
p, div, span, label { color: #a0a0a0; }
.stSelectbox label, .stSlider label {
    color: #555 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}
.stSelectbox > div > div {
    background: #141414 !important;
    border: 1px solid #1e1e1e !important;
    color: #e0e0e0 !important;
}
[data-testid="stMetric"] {
    background: #111;
    border: 1px solid #1e1e1e;
    padding: 20px 24px;
}
[data-testid="stMetricLabel"] {
    color: #555 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: .8px;
}
[data-testid="stMetricValue"] {
    color: #fff !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.9rem !important; }
hr { border-color: #1a1a1a !important; margin: 1rem 0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)

# Metro dark visual override
st.markdown("""
<style>
:root {
    --bg: #0b0d10;
    --panel: #11151a;
    --panel-2: #151b22;
    --line: #25303b;
    --muted: #91a0ad;
    --text: #f3f8fb;
    --accent: #00bcf2;
    --green: #7fba00;
    --yellow: #ffb900;
    --red: #e81123;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background:
        linear-gradient(90deg, rgba(0, 188, 242, 0.045), transparent 34rem),
        var(--bg) !important;
    color: var(--text);
    font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 1220px;
    padding-top: 1.35rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
}

p, div, span, label { color: var(--muted); }
hr {
    border: 0;
    border-top: 1px solid var(--line) !important;
    margin: 1.25rem 0 !important;
}

.metro-brand {
    border-left: 5px solid var(--accent);
    padding-left: 1rem;
    margin-bottom: .4rem;
}
.metro-kicker,
.section-label,
[data-testid="stMetricLabel"],
.stSelectbox label,
.stSlider label {
    color: #7d8c99 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 700 !important;
}
.metro-title {
    color: var(--text);
    font-size: 2.55rem;
    line-height: 1.05;
    font-weight: 300;
    margin-top: .25rem;
}
.metro-subtitle {
    max-width: 760px;
    color: #a9b7c4;
    font-size: 1rem;
    margin-top: .45rem;
}

.metro-tile {
    background: var(--panel);
    border: 1px solid var(--line);
    border-top: 4px solid var(--accent);
    min-height: 150px;
    padding: 1.15rem 1.25rem;
}
.metro-tile.green { border-top-color: var(--green); }
.metro-tile.yellow { border-top-color: var(--yellow); }
.metro-tile.red { border-top-color: var(--red); }
.tile-label {
    color: #8493a0;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}
.tile-value {
    color: var(--text);
    font-size: 2.45rem;
    font-weight: 300;
    line-height: 1.05;
    margin-top: .55rem;
}
.tile-note {
    color: #a7b4bf;
    font-size: .92rem;
    line-height: 1.45;
    margin-top: .65rem;
}
.tile-accent { color: var(--accent); font-weight: 600; }
.intent-pill {
    display: inline-flex;
    align-items: center;
    min-height: 2.4rem;
    padding: .25rem .75rem;
    border: 1px solid currentColor;
    color: inherit;
    font-size: 1.25rem;
    font-weight: 600;
}
.section-label { margin-bottom: .75rem; }

.stSelectbox > div > div {
    background: var(--panel-2) !important;
    border: 1px solid var(--line) !important;
    color: var(--text) !important;
    border-radius: 0 !important;
    min-height: 2.8rem;
}
.stSelectbox [data-baseweb="select"] span { color: var(--text) !important; }

.stSlider {
    background: rgba(17, 21, 26, .72);
    border-left: 3px solid #26313c;
    padding: .7rem .85rem .35rem .85rem;
    margin-bottom: .7rem;
}
.stSlider [data-baseweb="slider"] > div { background: #26313c !important; }
.stSlider [role="slider"] {
    background: var(--accent) !important;
    border: 2px solid #071013 !important;
    box-shadow: none !important;
}
[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-top: 4px solid var(--accent);
    padding: 1rem 1.25rem;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 2.45rem !important;
    font-weight: 300 !important;
}
[data-testid="stMetricDelta"] { font-size: .92rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("reports", "xgb_model.pkl")
FEAT_PATH  = os.path.join("reports", "feature_cols.pkl")

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(FEAT_PATH)

model, feat_cols = load_artifacts()

if model is None:
    st.error("Model not found at `reports/xgb_model.pkl`. Run cell 9a in t1.ipynb first.")
    st.stop()

# ── Persona baselines — values from actual upgrader analysis ──────────────────
# Non-upgrader medians: total_events=17, ai_gen_count=0, unique_events=11
# Upgrader medians:     total_events=199, ai_gen_count=14, unique_events=26
# Upgrades happen fast: median tenure = 1 day, p90 = 24 days
PERSONAS: dict[str, dict] = {
    "Casual (no AI, low activity)": {
        "tenure_days": 3, "total_events": 17, "active_days": 1,
        "unique_event_types": 11, "pageview_count": 12, "exception_count": 1,
        "ai_gen_count": 0, "run_block_count": 3, "run_all_blocks_count": 1,
        "block_create_count": 2, "canvas_create_count": 0,
        "files_upload_count": 0, "report_share_count": 0,
        "deployment_count": 0, "notebook_deploy_usage": 0,
        "credit_pressure_count": 0, "ai_tokens_total": 0,
        "ai_tokens_max_single": 0, "ai_latency_avg": 0,
        "session_count": 3, "avg_inter_event_min": 10,
        "events_last_7d": 17, "ai_gens_last_7d": 0, "active_days_7d": 1,
        "events_last_14d": 17, "ai_gens_last_14d": 0, "active_days_14d": 1,
        "deploys_last_14d": 0, "events_last_30d": 17, "active_days_30d": 1,
        "events_per_day": 5.7, "ai_pct_events": 0.0, "has_used_ai": 0,
        "has_deployed": 0, "has_hit_credit_limit": 0, "days_to_first_ai": 0,
        "events_per_session": 5.7,
    },
    "Explorer (tried AI, low volume)": {
        "tenure_days": 5, "total_events": 60, "active_days": 2,
        "unique_event_types": 18, "pageview_count": 30, "exception_count": 3,
        "ai_gen_count": 5, "run_block_count": 15, "run_all_blocks_count": 3,
        "block_create_count": 8, "canvas_create_count": 1,
        "files_upload_count": 1, "report_share_count": 0,
        "deployment_count": 0, "notebook_deploy_usage": 0,
        "credit_pressure_count": 0, "ai_tokens_total": 20000,
        "ai_tokens_max_single": 5000, "ai_latency_avg": 3.2,
        "session_count": 8, "avg_inter_event_min": 7,
        "events_last_7d": 60, "ai_gens_last_7d": 5, "active_days_7d": 2,
        "events_last_14d": 60, "ai_gens_last_14d": 5, "active_days_14d": 2,
        "deploys_last_14d": 0, "events_last_30d": 60, "active_days_30d": 2,
        "events_per_day": 12.0, "ai_pct_events": 0.08, "has_used_ai": 1,
        "has_deployed": 0, "has_hit_credit_limit": 0, "days_to_first_ai": 1,
        "events_per_session": 7.5,
    },
    "Heavy AI User (upgrader profile)": {
        # Matches actual upgrader medians from data analysis
        "tenure_days": 3, "total_events": 199, "active_days": 2,
        "unique_event_types": 26, "pageview_count": 80, "exception_count": 8,
        "ai_gen_count": 14, "run_block_count": 60, "run_all_blocks_count": 15,
        "block_create_count": 20, "canvas_create_count": 3,
        "files_upload_count": 2, "report_share_count": 0,
        "deployment_count": 0, "notebook_deploy_usage": 0,
        "credit_pressure_count": 3, "ai_tokens_total": 80000,
        "ai_tokens_max_single": 15000, "ai_latency_avg": 3.8,
        "session_count": 20, "avg_inter_event_min": 4,
        "events_last_7d": 199, "ai_gens_last_7d": 14, "active_days_7d": 2,
        "events_last_14d": 199, "ai_gens_last_14d": 14, "active_days_14d": 2,
        "deploys_last_14d": 0, "events_last_30d": 199, "active_days_30d": 2,
        "events_per_day": 66.3, "ai_pct_events": 0.10, "has_used_ai": 1,
        "has_deployed": 0, "has_hit_credit_limit": 1, "days_to_first_ai": 0,
        "events_per_session": 10.0,
    },
    "Power AI User (top upgrader)": {
        # p75+ upgrader: ai_gen_count > 50, heavy credit pressure
        "tenure_days": 6, "total_events": 450, "active_days": 4,
        "unique_event_types": 28, "pageview_count": 160, "exception_count": 15,
        "ai_gen_count": 80, "run_block_count": 140, "run_all_blocks_count": 35,
        "block_create_count": 50, "canvas_create_count": 8,
        "files_upload_count": 6, "report_share_count": 2,
        "deployment_count": 1, "notebook_deploy_usage": 2,
        "credit_pressure_count": 10, "ai_tokens_total": 400000,
        "ai_tokens_max_single": 40000, "ai_latency_avg": 4.5,
        "session_count": 40, "avg_inter_event_min": 3,
        "events_last_7d": 450, "ai_gens_last_7d": 80, "active_days_7d": 4,
        "events_last_14d": 450, "ai_gens_last_14d": 80, "active_days_14d": 4,
        "deploys_last_14d": 1, "events_last_30d": 450, "active_days_30d": 4,
        "events_per_day": 75.0, "ai_pct_events": 0.18, "has_used_ai": 1,
        "has_deployed": 1, "has_hit_credit_limit": 1, "days_to_first_ai": 0,
        "events_per_session": 11.3,
    },
    "At Risk (was active, now quiet)": {
        "tenure_days": 30, "total_events": 199, "active_days": 3,
        "unique_event_types": 20, "pageview_count": 80, "exception_count": 8,
        "ai_gen_count": 14, "run_block_count": 50, "run_all_blocks_count": 12,
        "block_create_count": 18, "canvas_create_count": 3,
        "files_upload_count": 2, "report_share_count": 0,
        "deployment_count": 0, "notebook_deploy_usage": 0,
        "credit_pressure_count": 2, "ai_tokens_total": 70000,
        "ai_tokens_max_single": 12000, "ai_latency_avg": 3.5,
        "session_count": 18, "avg_inter_event_min": 15,
        "events_last_7d": 2, "ai_gens_last_7d": 0, "active_days_7d": 1,
        "events_last_14d": 5, "ai_gens_last_14d": 0, "active_days_14d": 1,
        "deploys_last_14d": 0, "events_last_30d": 30, "active_days_30d": 3,
        "events_per_day": 1.0, "ai_pct_events": 0.07, "has_used_ai": 1,
        "has_deployed": 0, "has_hit_credit_limit": 0, "days_to_first_ai": 2,
        "events_per_session": 11.1,
    },
}

SLIDER_DEFS = [
    ("ai_gen_count",          "AI Generations (total)",    0, 200,  1),
    ("total_events",          "Total events",              0, 800, 10),
    ("unique_event_types",    "Unique event types",        0,  30,  1),
    ("credit_pressure_count", "Credit limit hits",         0,  20,  1),
    ("session_count",         "Sessions",                  0, 100,  1),
    ("canvas_create_count",   "Canvases created",          0,  20,  1),
    ("block_create_count",    "Blocks created",            0, 100,  1),
    ("tenure_days",           "Tenure (days)",             0,  60,  1),
    ("deployment_count",      "Deployments",               0,  20,  1),
    ("ai_tokens_total",       "AI tokens used (total)",    0, 500000, 5000),
]

# Tiers anchored to base rate of 1.84% (actual upgrader rate from data)
BASE_RATE = 0.0184
TIER_CONFIG = [
    (BASE_RATE * 1.5, "Low Intent",   "#e74c3c"),   # < 1.5x base rate
    (BASE_RATE * 4,   "Moderate",     "#e67e22"),   # 1.5–4x
    (BASE_RATE * 10,  "High Intent",  "#f1c40f"),   # 4–10x
    (1.01,            "Very High",    "#2ecc71"),   # > 10x
]

ACTIONS = {
    "Low Intent":  "Feature discovery emails + onboarding nudges",
    "Moderate":    "In-app upgrade prompt at peak engagement",
    "High Intent": "Personalised outreach + time-limited offer",
    "Very High":   "Direct sales contact / priority demo",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_tier(p: float) -> tuple[str, str]:
    for thresh, label, color in TIER_CONFIG:
        if p < thresh:
            return label, color
    return "Very High", "#2ecc71"


def score(vals: dict) -> float:
    row = {c: 0.0 for c in feat_cols}
    row.update({k: float(v) for k, v in vals.items() if k in row})
    # derived features
    total = max(row.get("total_events", 1), 1)
    row["pct_activity_last_7d"] = row.get("events_last_7d", 0) / total
    e7  = row.get("events_last_7d", 0)
    e14 = row.get("events_last_14d", 0)
    row["is_ramping"] = float(int(e7 > e14 / 2) if e14 > 0 else 0)
    # binary flags
    row["has_used_ai"]           = float(row.get("ai_gen_count", 0) > 0)
    row["has_deployed"]          = float(row.get("deployment_count", 0) > 0)
    row["has_hit_credit_limit"]  = float(row.get("credit_pressure_count", 0) > 0)
    df = pd.DataFrame([row])[feat_cols]
    return float(model.predict_proba(df)[0][1])


# ── Session state — slider reset on persona change ─────────────────────────────
def _init_sliders(persona_name: str) -> None:
    for col, *_ in SLIDER_DEFS:
        st.session_state[f"sl_{col}"] = int(round(
            PERSONAS[persona_name].get(col, 0)
        ))

if "ready" not in st.session_state:
    st.session_state.ready = True
    st.session_state.active_persona = list(PERSONAS.keys())[0]
    _init_sliders(st.session_state.active_persona)

def _on_persona_change() -> None:
    name = st.session_state.persona_select
    st.session_state.active_persona = name
    _init_sliders(name)


# ── Compute scores ─────────────────────────────────────────────────────────────
persona_name = st.session_state.active_persona
baseline     = PERSONAS[persona_name]

# ── Header ─────────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([2.4, 1])
with hcol1:
    st.markdown(
        """
        <div class="metro-brand">
            <div class="metro-kicker">Zerve Growth Intelligence</div>
            <div class="metro-title">Upgrade Probability Simulator</div>
            <div class="metro-subtitle">
                Model-driven what-if analysis for product behavior, AI usage,
                credit pressure, and conversion intent.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hcol2:
    st.selectbox(
        "Persona",
        list(PERSONAS.keys()),
        key="persona_select",
        on_change=_on_persona_change,
    )
st.markdown("---")

# ── Sliders in main area — 2 columns ──────────────────────────────────────────
st.markdown(
    '<div class="section-label">Behavioral Signals</div>',
    unsafe_allow_html=True,
)
half = len(SLIDER_DEFS) // 2
scol1, scol2 = st.columns(2)
slider_vals: dict[str, int] = {}

with scol1:
    for col, label, lo, hi, step in SLIDER_DEFS[:half]:
        slider_vals[col] = st.slider(label, lo, hi, step=step, key=f"sl_{col}")

with scol2:
    for col, label, lo, hi, step in SLIDER_DEFS[half:]:
        slider_vals[col] = st.slider(label, lo, hi, step=step, key=f"sl_{col}")

st.markdown("---")

what_if = {**baseline, **{col: float(v) for col, v in slider_vals.items()}}
base_p  = score(baseline)
what_p  = score(what_if)
delta   = what_p - base_p
tier_lbl, tier_col = get_tier(what_p)

# ── Metric row ─────────────────────────────────────────────────────────────────
lift = what_p / BASE_RATE
tile_class = "green" if tier_lbl == "Very High" else "yellow" if tier_lbl == "High Intent" else "red" if tier_lbl == "Low Intent" else ""
delta_sign = "+" if delta >= 0 else ""

c1, c2, c3 = st.columns([1, 1, 1.6])

with c1:
    st.markdown(
        f"""
        <div class="metro-tile">
            <div class="tile-label">Upgrade Probability</div>
            <div class="tile-value">{what_p * 100:.2f}%</div>
            <div class="tile-note">
                <span class="tile-accent">{delta_sign}{delta * 100:.2f}pp</span>
                vs selected persona baseline
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metro-tile {tile_class}">
            <div class="tile-label">Upgrade Tier</div>
            <div style="color:{tier_col}; margin-top:.75rem;">
                <span class="intent-pill">{tier_lbl}</span>
            </div>
            <div class="tile-note">
                <span class="tile-accent">{lift:.1f}x</span>
                the average user conversion rate
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metro-tile">
            <div class="tile-label">Recommended Action</div>
            <div class="tile-note" style="font-size:1.08rem;color:#d9e4ec;margin-top:.9rem;">
                {ACTIONS[tier_lbl]}
            </div>
            <div class="tile-note">
                Baseline: {base_p * 100:.2f}% | What-if: {what_p * 100:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Chart ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Probability Comparison</div>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 2.35))
fig.patch.set_facecolor("#0b0d10")
ax.set_facecolor("#0b0d10")

x_max = max(what_p * 100 * 1.5, base_p * 100 * 1.5, 1.0)
bars = ax.barh(
    ["Baseline", "What-If"],
    [base_p * 100, what_p * 100],
    color=["#25303b", tier_col],
    height=0.38,
    edgecolor="#0b0d10",
)
for bar, val in zip(bars, [base_p, what_p]):
    ax.text(
        bar.get_width() + x_max * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val * 100:.2f}%",
        va="center", color="#f3f8fb", fontsize=12, fontweight="600",
    )

ax.set_xlim(0, x_max)
ax.axvline(base_p * 100, color="#51606e", lw=1, linestyle="--", alpha=.8)
ax.spines[["top", "right", "bottom", "left"]].set_color("#25303b")
ax.tick_params(colors="#91a0ad", length=0)
ax.set_xlabel("Upgrade Probability (%)", color="#91a0ad", fontsize=10)
ax.grid(axis="x", color="#1b232b", linewidth=.8)
ax.set_axisbelow(True)
for lbl in ax.get_yticklabels():
    lbl.set_color("#c8d3dc")
    lbl.set_fontweight("600")

plt.tight_layout(pad=0.5)
st.pyplot(fig, use_container_width=True)
plt.close(fig)
