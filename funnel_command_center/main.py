"""Funnel Command Center - Zerve Datathon demo app."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
DATA_PATH = ROOT / "datasets" / "zerve_events.csv"
REPORTS = APP_DIR / "reports"
FEATURE_CACHE = APP_DIR / "command_center_user_features.pkl"

BASE_RATE = 0.0184

DEPLOY_EVENTS = {
    "notebook_deployment_usage_tracked",
    "notebook_deployed",
    "app_deployed",
    "deployment_created",
}
BLOCK_CREATE = {"block_create", "block_created"}
RUN_BLOCK = {"run_block", "block_run"}
RUN_ALL = {"run_all_blocks", "notebook_run"}
FILE_UPLOAD = {"file_uploaded", "file_created", "files_upload"}
SHARE_EVENTS = {"report_shared", "canvas_shared"}

STAGE_ORDER = [
    "New",
    "Exploring",
    "Created Content",
    "AI Engaged",
    "Workflow Builder",
    "Credit Active",
    "Consistently Engaged",
    "At Risk",
    "Upgraded",
]

STAGE_COLORS = {
    "New": "#5f6368",
    "Exploring": "#2d7dd2",
    "Created Content": "#8f6ed5",
    "AI Engaged": "#00a88f",
    "Workflow Builder": "#ff8a3d",
    "Credit Active": "#ff5f57",
    "Consistently Engaged": "#ffd166",
    "At Risk": "#d1495b",
    "Upgraded": "#44d17a",
}

TIER_ORDER = ["Low Intent", "Moderate", "High Intent", "Very High"]
TIER_COLORS = {
    "Low Intent": "#6f747d",
    "Moderate": "#2d7dd2",
    "High Intent": "#ffd166",
    "Very High": "#44d17a",
}


st.set_page_config(
    page_title="Funnel Command Center",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #050607;
  --panel: #0b0d10;
  --panel2: #101318;
  --line: #242932;
  --text: #f2f5f8;
  --muted: #8b949e;
  --blue: #2d7dd2;
  --green: #44d17a;
  --yellow: #ffd166;
  --red: #ff5f57;
}

html, body, [class*="css"] {
  font-family: Inter, Segoe UI, Arial, sans-serif;
}

[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
section[data-testid="stSidebar"] {
  background: var(--bg);
}

section[data-testid="stSidebar"] {
  border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] * {
  color: var(--text);
}

.block-container {
  max-width: 1420px;
  padding-top: 4.25rem;
  padding-bottom: 2rem;
}

[data-testid="stDecoration"] {
  display: none;
}

h1, h2, h3, h4 {
  color: var(--text) !important;
  letter-spacing: 0;
}

p, label, span, div {
  color: var(--muted);
}

.metro-title {
  color: var(--text);
  font-size: 42px;
  font-weight: 800;
  line-height: 1.02;
  margin: 0;
}

.metro-subtitle {
  color: var(--muted);
  font-size: 14px;
  margin-top: 8px;
}

.rail {
  border-left: 5px solid var(--blue);
  padding-left: 18px;
  margin-bottom: 18px;
}

.kpi, .panel, .mini-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 0;
}

.kpi {
  padding: 17px 18px 15px 18px;
  min-height: 116px;
}

.kpi-label {
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  font-weight: 700;
}

.kpi-value {
  color: var(--text);
  font-size: 30px;
  font-weight: 800;
  margin-top: 9px;
}

.kpi-note {
  color: var(--muted);
  font-size: 12px;
  margin-top: 4px;
}

.panel {
  padding: 18px;
}

.section-label {
  color: var(--text);
  font-size: 15px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  margin-bottom: 12px;
}

.stage-row, .drop-row, .action-row {
  display: grid;
  align-items: center;
  gap: 12px;
  border-top: 1px solid #171b21;
  padding: 10px 0;
}

.stage-row {
  grid-template-columns: 156px 1fr 74px 68px;
}

.drop-row {
  grid-template-columns: 210px 95px 1fr;
}

.action-row {
  grid-template-columns: 190px 1fr;
}

.bar-track {
  height: 12px;
  background: #151920;
}

.bar-fill {
  height: 12px;
}

.stage-name, .drop-name, .action-title {
  color: var(--text);
  font-size: 13px;
  font-weight: 700;
}

.stage-count, .stage-pct, .drop-pct {
  color: var(--text);
  font-size: 13px;
  font-weight: 700;
  text-align: right;
}

.diagnosis {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}

.tag {
  display: inline-block;
  padding: 4px 7px;
  border: 1px solid var(--line);
  background: #11151b;
  color: var(--text);
  font-size: 11px;
  font-weight: 700;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
  background: #0b0d10;
  border: 1px solid var(--line);
  border-bottom: none;
  border-radius: 0;
  padding: 11px 18px;
  color: var(--muted);
  font-weight: 700;
}

.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  border-top: 3px solid var(--blue);
}

[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 14px 16px;
}

[data-testid="stMetricLabel"] {
  color: var(--muted) !important;
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 1.4px;
}

[data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-size: 28px !important;
  font-weight: 800 !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
}

.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
  background: #0b0d10;
  border-radius: 0;
}

.stSlider [data-baseweb="slider"] {
  margin-top: 8px;
}

hr {
  border-color: var(--line);
  margin: 16px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


def compact_number(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def clean_segment(series: pd.Series) -> pd.Series:
    out = series.fillna("Unknown").astype(str).str.strip()
    return out.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})


def funnel_stage(row: pd.Series | dict) -> str:
    get = row.get
    if get("upgraded", 0):
        return "Upgraded"
    if get("total_events", 0) < 3:
        return "New"

    stage = "Exploring"
    if get("canvas_create_count", 0) > 0 or get("block_create_count", 0) > 0 or get("files_upload_count", 0) > 0:
        stage = "Created Content"
    if get("ai_gen_count", 0) > 0:
        stage = "AI Engaged"
    if get("run_block_count", 0) > 2 or get("deployment_count", 0) > 0 or get("run_all_blocks_count", 0) > 0:
        stage = "Workflow Builder"
    if get("credit_pressure_count", 0) > 0:
        stage = "Credit Active"
    if get("active_days", 0) >= 7:
        stage = "Consistently Engaged"
    if stage not in ("New", "Exploring") and get("events_last_14d", 0) == 0:
        stage = "At Risk"
    return stage


def risk_tier(probability: float) -> str:
    if probability < BASE_RATE * 1.5:
        return "Low Intent"
    if probability < BASE_RATE * 4:
        return "Moderate"
    if probability < BASE_RATE * 10:
        return "High Intent"
    return "Very High"


@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load(REPORTS / "xgb_model.pkl")
    feature_cols = joblib.load(REPORTS / "feature_cols.pkl")
    return model, feature_cols


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["person_id", "ts"])
    snapshot = df["ts"].max()

    upgrades = df[df["event"] == "subscription_upgraded"].groupby("person_id")["ts"].min()
    snap_map = pd.to_datetime(df["person_id"].map(upgrades), utc=True).fillna(snapshot)
    df["snap"] = snap_map
    safe = df[df["ts"] < df["snap"]].copy()
    grouped = safe.groupby("person_id")

    feats = pd.DataFrame(
        {
            "total_events": grouped["event"].count(),
            "active_days": grouped["ts"].apply(lambda x: x.dt.date.nunique()),
            "unique_event_types": grouped["event"].nunique(),
            "pageview_count": grouped["event"].apply(lambda x: (x == "$pageview").sum()),
            "exception_count": grouped["event"].apply(lambda x: (x == "$exception").sum()),
            "ai_gen_count": grouped["event"].apply(lambda x: (x == "$ai_generation").sum()),
            "run_block_count": grouped["event"].apply(lambda x: x.isin(RUN_BLOCK).sum()),
            "run_all_blocks_count": grouped["event"].apply(lambda x: x.isin(RUN_ALL).sum()),
            "block_create_count": grouped["event"].apply(lambda x: x.isin(BLOCK_CREATE).sum()),
            "canvas_create_count": grouped["event"].apply(lambda x: (x == "canvas_create").sum()),
            "files_upload_count": grouped["event"].apply(lambda x: x.isin(FILE_UPLOAD).sum()),
            "report_share_count": grouped["event"].apply(lambda x: x.isin(SHARE_EVENTS).sum()),
            "deployment_count": grouped["event"].apply(lambda x: x.isin(DEPLOY_EVENTS).sum()),
            "notebook_deploy_usage": grouped["event"].apply(lambda x: (x == "notebook_deployment_usage_tracked").sum()),
            "credit_pressure_count": grouped["event"].apply(lambda x: (x == "credits_remaining").sum()),
        }
    )

    ai = safe[safe["event"] == "$ai_generation"].copy()
    if {"properties.$ai_input_tokens", "properties.$ai_output_tokens"}.issubset(ai.columns):
        ai["tkn"] = ai["properties.$ai_input_tokens"].fillna(0) + ai["properties.$ai_output_tokens"].fillna(0)
        feats["ai_tokens_total"] = ai.groupby("person_id")["tkn"].sum().reindex(feats.index, fill_value=0)
        feats["ai_tokens_max_single"] = ai.groupby("person_id")["tkn"].max().reindex(feats.index, fill_value=0)
    else:
        feats["ai_tokens_total"] = 0
        feats["ai_tokens_max_single"] = 0

    if "properties.$ai_latency" in ai.columns:
        feats["ai_latency_avg"] = ai.groupby("person_id")["properties.$ai_latency"].mean().reindex(feats.index, fill_value=0)
    else:
        feats["ai_latency_avg"] = 0

    feats["tenure_days"] = grouped["ts"].apply(lambda x: max((x.max() - x.min()).total_seconds() / 86400 + 1, 1))
    feats["session_count"] = grouped["ts"].apply(lambda x: int((x.sort_values().diff().dt.total_seconds().fillna(9999) > 1800).sum() + 1))
    feats["avg_inter_event_min"] = grouped["ts"].apply(
        lambda x: x.sort_values().diff().dt.total_seconds().dropna().mean() / 60 if len(x) > 1 else 0
    )

    for col in [
        "person_properties.purpose",
        "person_properties.role",
        "person_properties.work_type",
        "person_properties.source",
    ]:
        if col in df.columns:
            first_values = df.groupby("person_id")[col].first().reindex(feats.index)
            feats[f"{col}_label"] = clean_segment(first_values)
            feats[col] = first_values.fillna("Unknown").astype("category").cat.codes
        else:
            feats[f"{col}_label"] = "Unknown"
            feats[col] = 0

    for days, suffix in [(7, "7d"), (14, "14d"), (30, "30d")]:
        window = safe[safe["ts"] >= snapshot - pd.Timedelta(days=days)]
        feats[f"events_last_{suffix}"] = window.groupby("person_id")["event"].count().reindex(feats.index, fill_value=0)
        feats[f"active_days_{suffix}"] = window.groupby("person_id")["ts"].apply(lambda x: x.dt.date.nunique()).reindex(feats.index, fill_value=0)
        if suffix != "30d":
            feats[f"ai_gens_last_{suffix}"] = window.groupby("person_id")["event"].apply(lambda x: (x == "$ai_generation").sum()).reindex(
                feats.index, fill_value=0
            )
        if suffix == "14d":
            feats["deploys_last_14d"] = window.groupby("person_id")["event"].apply(lambda x: x.isin(DEPLOY_EVENTS).sum()).reindex(
                feats.index, fill_value=0
            )

    first_event = grouped["ts"].min()
    first_ai = safe[safe["event"] == "$ai_generation"].groupby("person_id")["ts"].min()
    feats["days_to_first_ai"] = ((first_ai - first_event).dt.total_seconds() / 86400).reindex(feats.index, fill_value=-1)

    feats["events_per_day"] = feats["total_events"] / feats["tenure_days"].clip(lower=1)
    feats["ai_pct_events"] = feats["ai_gen_count"] / feats["total_events"].clip(lower=1)
    feats["has_used_ai"] = (feats["ai_gen_count"] > 0).astype(int)
    feats["has_deployed"] = (feats["deployment_count"] > 0).astype(int)
    feats["has_hit_credit_limit"] = (feats["credit_pressure_count"] > 0).astype(int)
    feats["events_per_session"] = feats["total_events"] / feats["session_count"].clip(lower=1)
    feats["pct_activity_last_7d"] = feats["events_last_7d"] / feats["total_events"].clip(lower=1)
    feats["is_ramping"] = (feats["events_last_7d"] > feats["events_last_14d"] / 2).astype(int)
    feats["upgraded"] = feats.index.isin(upgrades.index).astype(int)
    feats["stage"] = feats.apply(funnel_stage, axis=1)
    feats["activity_level"] = pd.cut(
        feats["total_events"],
        bins=[-1, 5, 30, 150, np.inf],
        labels=["Low", "Medium", "High", "Power"],
    ).astype(str)
    feats["ai_usage"] = pd.cut(
        feats["ai_gen_count"],
        bins=[-1, 0, 3, 15, np.inf],
        labels=["None", "Light", "Active", "Power"],
    ).astype(str)
    feats["deployment_usage"] = np.where(feats["deployment_count"] > 0, "Deployed", "No deployment")
    return feats.reset_index().rename(columns={"index": "person_id"})


@st.cache_data(show_spinner=False)
def get_user_features() -> pd.DataFrame:
    if FEATURE_CACHE.exists():
        return pd.read_pickle(FEATURE_CACHE)

    usecols = [
        "person_id",
        "timestamp",
        "event",
        "person_properties.purpose",
        "person_properties.role",
        "person_properties.work_type",
        "person_properties.source",
        "properties.$ai_input_tokens",
        "properties.$ai_output_tokens",
        "properties.$ai_latency",
    ]
    with st.spinner("Building command center dataset. First run can take a minute."):
        df = pd.read_csv(DATA_PATH, usecols=lambda c: c in usecols)
        features = build_features(df)
        model, feature_cols = load_model()
        scoring_frame = features.copy()
        for col in feature_cols:
            if col not in scoring_frame:
                scoring_frame[col] = 0
        features["upgrade_probability"] = model.predict_proba(scoring_frame[feature_cols])[:, 1]
        features["risk_tier"] = features["upgrade_probability"].apply(risk_tier)
        features.to_pickle(FEATURE_CACHE)
    return features


def sidebar_filters(features: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Filters")
    purpose = st.sidebar.multiselect(
        "Purpose",
        sorted(features["person_properties.purpose_label"].unique()),
        default=[],
    )
    role = st.sidebar.multiselect(
        "Role",
        sorted(features["person_properties.role_label"].unique()),
        default=[],
    )
    source = st.sidebar.multiselect(
        "Source",
        sorted(features["person_properties.source_label"].unique()),
        default=[],
    )
    activity = st.sidebar.multiselect(
        "Activity level",
        ["Low", "Medium", "High", "Power"],
        default=[],
    )
    ai_usage = st.sidebar.multiselect(
        "AI usage",
        ["None", "Light", "Active", "Power"],
        default=[],
    )
    deployment = st.sidebar.multiselect(
        "Deployment",
        ["No deployment", "Deployed"],
        default=[],
    )
    min_probability = st.sidebar.slider("Minimum upgrade probability", 0.0, 50.0, 0.0, 0.5)

    filtered = features.copy()
    if purpose:
        filtered = filtered[filtered["person_properties.purpose_label"].isin(purpose)]
    if role:
        filtered = filtered[filtered["person_properties.role_label"].isin(role)]
    if source:
        filtered = filtered[filtered["person_properties.source_label"].isin(source)]
    if activity:
        filtered = filtered[filtered["activity_level"].isin(activity)]
    if ai_usage:
        filtered = filtered[filtered["ai_usage"].isin(ai_usage)]
    if deployment:
        filtered = filtered[filtered["deployment_usage"].isin(deployment)]
    if min_probability > 0:
        filtered = filtered[filtered["upgrade_probability"] >= min_probability / 100]
    return filtered


def kpi(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
<div class="kpi">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-note">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_stage_breakdown(df: pd.DataFrame):
    counts = df["stage"].value_counts().reindex(STAGE_ORDER, fill_value=0)
    total = max(len(df), 1)
    max_count = max(counts.max(), 1)
    rows = ['<div class="panel"><div class="section-label">Funnel Stage Breakdown</div>']
    for stage, count in counts.items():
        width = count / max_count * 100
        color = STAGE_COLORS[stage]
        rows.append(
            f"""
<div class="stage-row">
  <div class="stage-name">{stage}</div>
  <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{color};"></div></div>
  <div class="stage-count">{compact_number(count)}</div>
  <div class="stage-pct">{count / total * 100:.1f}%</div>
</div>
"""
        )
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_risk_tiers(df: pd.DataFrame):
    counts = df["risk_tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
    total = max(len(df), 1)
    max_count = max(counts.max(), 1)
    rows = ['<div class="panel"><div class="section-label">Risk Tiers</div>']
    for tier, count in counts.items():
        subset = df[df["risk_tier"] == tier]
        avg_prob = subset["upgrade_probability"].mean() if len(subset) else 0
        width = count / max_count * 100
        color = TIER_COLORS[tier]
        rows.append(
            f"""
<div class="stage-row">
  <div class="stage-name">{tier}</div>
  <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{color};"></div></div>
  <div class="stage-count">{compact_number(count)}</div>
  <div class="stage-pct">{avg_prob * 100:.1f}%</div>
</div>
"""
        )
    rows.append(f'<div class="diagnosis" style="margin-top:10px">Coverage: {compact_number(total)} users in current segment.</div></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_dropoffs(df: pd.DataFrame):
    ordered = [
        "New",
        "Exploring",
        "Created Content",
        "AI Engaged",
        "Workflow Builder",
        "Credit Active",
        "Consistently Engaged",
        "Upgraded",
    ]
    rank = {stage: i for i, stage in enumerate(ordered)}
    ranked = df[df["stage"].isin(rank)].copy()
    ranked["_stage_rank"] = ranked["stage"].map(rank)
    rows = ['<div class="panel"><div class="section-label">Top Drop-Off Points</div>']
    transitions = []
    for current, nxt in zip(ordered, ordered[1:]):
        current_count = int((ranked["_stage_rank"] >= rank[current]).sum())
        next_count = int((ranked["_stage_rank"] >= rank[nxt]).sum())
        if current_count > 0:
            drop = max(current_count - next_count, 0)
            transitions.append((current, nxt, drop / current_count, drop))

    transitions = sorted(transitions, key=lambda x: x[2], reverse=True)[:5]
    diagnoses = {
        "New": "Improve first-run guidance and reduce empty-state friction.",
        "Exploring": "Move browsing users into templates, canvas creation, or guided project starts.",
        "Created Content": "Surface contextual AI prompts inside active work.",
        "AI Engaged": "Encourage repeat AI workflows and deeper notebook actions.",
        "Workflow Builder": "Make deployment and sharing paths more visible.",
        "Credit Active": "Clarify plan value at the moment of credit pressure.",
        "Consistently Engaged": "Prioritize lifecycle messaging for active but unpaid teams.",
    }
    for current, nxt, rate, drop in transitions:
        rows.append(
            f"""
<div class="drop-row">
  <div class="drop-name">{current} to {nxt}</div>
  <div class="drop-pct">{rate * 100:.1f}%</div>
  <div class="diagnosis">{diagnoses.get(current, "Review stage behavior for intervention opportunities.")}</div>
</div>
"""
        )
    if not transitions:
        rows.append('<div class="diagnosis">No drop-off points available for the current filter.</div>')
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_segment_chart(df: pd.DataFrame, column: str, title: str):
    grouped = df.groupby(column, observed=True).agg(
        users=("person_id", "count"),
        upgrade_rate=("upgraded", "mean"),
        avg_probability=("upgrade_probability", "mean"),
    )
    grouped = grouped.sort_values("users", ascending=False).head(12).sort_values("users")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.patch.set_facecolor("#050607")
    ax.set_facecolor("#050607")
    ax.barh(grouped.index.astype(str), grouped["users"], color="#2d7dd2", alpha=0.86)
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.set_title(title, color="#f2f5f8", loc="left", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Users", color="#8b949e", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#242932")
    ax.grid(axis="x", color="#171b21", linewidth=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def recommended_action(row: pd.Series) -> str:
    if row["risk_tier"] == "Very High" and row["credit_pressure_count"] > 0:
        return "Trigger upgrade offer"
    if row["ai_gen_count"] > 0 and row["deployment_count"] == 0:
        return "Prompt deployment"
    if row["stage"] == "Created Content":
        return "Nudge AI workflow"
    if row["stage"] == "At Risk":
        return "Send reactivation"
    if row["ai_gen_count"] == 0:
        return "Show AI starter"
    return "Lifecycle nurture"


def render_actions():
    actions = [
        ("New and Exploring", "Guide users into a concrete starter template before they drift."),
        ("Created Content", "Place AI prompts directly where users are building."),
        ("AI Engaged", "Encourage repeat generations and workflow depth."),
        ("Credit Active", "Make upgrade value explicit at the credit-pressure moment."),
        ("At Risk", "Recover previously engaged users with recent-work reminders."),
    ]
    rows = ['<div class="panel"><div class="section-label">Product Actions</div>']
    for title, body in actions:
        rows.append(
            f"""
<div class="action-row">
  <div class="action-title">{title}</div>
  <div class="diagnosis">{body}</div>
</div>
"""
        )
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


features = get_user_features()
filtered = sidebar_filters(features)

st.markdown(
    """
<div class="rail">
  <div class="metro-title">Funnel Command Center</div>
  <div class="metro-subtitle">Executive view of activation, upgrade intent, and product-led growth opportunities.</div>
</div>
""",
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No users match the current filters.")
    st.stop()

total_users = len(filtered)
upgraded_users = int(filtered["upgraded"].sum())
upgrade_rate = upgraded_users / max(total_users, 1)
high_intent = int(((filtered["risk_tier"].isin(["High Intent", "Very High"])) & (filtered["upgraded"] == 0)).sum())
at_risk = int((filtered["stage"] == "At Risk").sum())
median_days_to_ai = filtered.loc[filtered["days_to_first_ai"] >= 0, "days_to_first_ai"].median()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi("Users", compact_number(total_users), "Current segment")
with k2:
    kpi("Upgrade Rate", pct(upgrade_rate), f"{compact_number(upgraded_users)} upgraded")
with k3:
    kpi("High Intent Free", compact_number(high_intent), "Scored above 7.4%")
with k4:
    kpi("At Risk", compact_number(at_risk), "Engaged then quiet")
with k5:
    kpi("Median First AI", f"{median_days_to_ai:.1f}d" if pd.notna(median_days_to_ai) else "n/a", "Among AI users")

st.markdown("<br>", unsafe_allow_html=True)

overview_tab, segment_tab, opportunity_tab, diagnosis_tab = st.tabs(
    ["Overview", "Segments", "Opportunities", "Diagnosis"]
)

with overview_tab:
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        render_stage_breakdown(filtered)
    with right:
        render_risk_tiers(filtered)
    st.markdown("<br>", unsafe_allow_html=True)
    render_dropoffs(filtered)

with segment_tab:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_segment_chart(filtered, "person_properties.purpose_label", "Users by Purpose")
    with c2:
        render_segment_chart(filtered, "person_properties.role_label", "Users by Role")
    c3, c4 = st.columns(2, gap="large")
    with c3:
        render_segment_chart(filtered, "person_properties.source_label", "Users by Source")
    with c4:
        render_segment_chart(filtered, "activity_level", "Users by Activity Level")

with opportunity_tab:
    free_users = filtered[filtered["upgraded"] == 0].copy()
    free_users["recommended_action"] = free_users.apply(recommended_action, axis=1)
    opportunities = free_users.sort_values(
        ["upgrade_probability", "events_last_7d", "ai_gen_count"],
        ascending=[False, False, False],
    ).head(40)
    display = opportunities[
        [
            "person_id",
            "stage",
            "risk_tier",
            "upgrade_probability",
            "ai_gen_count",
            "deployment_count",
            "credit_pressure_count",
            "events_last_7d",
            "recommended_action",
        ]
    ].rename(
        columns={
            "person_id": "User",
            "stage": "Stage",
            "risk_tier": "Risk Tier",
            "upgrade_probability": "Upgrade Probability",
            "ai_gen_count": "AI Generations",
            "deployment_count": "Deployments",
            "credit_pressure_count": "Credit Pressure",
            "events_last_7d": "Events Last 7d",
            "recommended_action": "Recommended Action",
        }
    )

    st.markdown('<div class="section-label">Top Opportunities</div>', unsafe_allow_html=True)
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Upgrade Probability": st.column_config.ProgressColumn(
                "Upgrade Probability",
                min_value=0,
                max_value=max(0.25, float(display["Upgrade Probability"].max()) if len(display) else 0.25),
                format="%.2f",
            )
        },
    )

with diagnosis_tab:
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        render_dropoffs(filtered)
    with c2:
        render_actions()

    st.markdown("<br>", unsafe_allow_html=True)
    image_path = REPORTS / "fig2_feature_importance.png"
    if image_path.exists():
        st.markdown('<div class="section-label">Model Signal Reference</div>', unsafe_allow_html=True)
        st.image(str(image_path), use_container_width=True)
