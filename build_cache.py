"""Pre-build user feature cache for the Funnel Sandbox app."""
import pandas as pd
import numpy as np
from pathlib import Path

DEPLOY_EVENTS = {"notebook_deployment_usage_tracked","notebook_deployed","app_deployed","deployment_created"}
BLOCK_CREATE  = {"block_create","block_created"}
RUN_BLOCK     = {"run_block","block_run"}
RUN_ALL       = {"run_all_blocks","notebook_run"}
FILE_UPLOAD   = {"file_uploaded","file_created","files_upload"}
SHARE_EVENTS  = {"report_shared","canvas_shared"}

print("Loading CSV...")
df = pd.read_csv("datasets/zerve_events.csv", low_memory=False)
df["ts"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
df = df.sort_values(["person_id", "ts"])
SNAP = df["ts"].max()

upgrades   = df[df["event"] == "subscription_upgraded"].groupby("person_id")["ts"].min()
df["snap"] = df["person_id"].map(upgrades).fillna(SNAP)
safe       = df[df["ts"] < df["snap"]].copy()
g          = safe.groupby("person_id")
print(f"Users: {g.ngroups:,}  Safe events: {len(safe):,}")

print("Computing base counts...")
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
inp, out = "properties.$ai_input_tokens", "properties.$ai_output_tokens"
if inp in ai.columns and out in ai.columns:
    ai["tkn"]                     = ai[inp].fillna(0) + ai[out].fillna(0)
    feats["ai_tokens_total"]      = ai.groupby("person_id")["tkn"].sum().reindex(feats.index, fill_value=0)
    feats["ai_tokens_max_single"] = ai.groupby("person_id")["tkn"].max().reindex(feats.index, fill_value=0)
else:
    feats["ai_tokens_total"] = feats["ai_tokens_max_single"] = 0

lat_col = "properties.$ai_latency"
feats["ai_latency_avg"] = (
    ai.groupby("person_id")[lat_col].mean().reindex(feats.index, fill_value=0)
    if lat_col in ai.columns else 0
)

print("Computing session / tenure features...")
feats["tenure_days"]         = g["ts"].apply(lambda x: max((x.max()-x.min()).total_seconds()/86400+1, 1))
feats["session_count"]       = g["ts"].apply(lambda x: int((x.sort_values().diff().dt.total_seconds().fillna(9999)>1800).sum()+1))
feats["avg_inter_event_min"] = g["ts"].apply(lambda x: x.sort_values().diff().dt.total_seconds().dropna().mean()/60 if len(x)>1 else 0)

for col in ["person_properties.purpose","person_properties.role","person_properties.work_type","person_properties.source"]:
    if col in df.columns:
        first = df.groupby("person_id")[col].first()
        feats[col] = first.astype("category").cat.codes.reindex(feats.index, fill_value=-1)
    else:
        feats[col] = 0

print("Computing time-window features...")
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
feats["has_used_ai"]          = (feats["ai_gen_count"]         > 0).astype(int)
feats["has_deployed"]         = (feats["deployment_count"]      > 0).astype(int)
feats["has_hit_credit_limit"] = (feats["credit_pressure_count"] > 0).astype(int)
feats["events_per_session"]   = feats["total_events"]   / feats["session_count"].clip(lower=1)
feats["pct_activity_last_7d"] = feats["events_last_7d"] / feats["total_events"].clip(lower=1)
feats["is_ramping"]           = (feats["events_last_7d"] > feats["events_last_14d"]/2).astype(int)

ai_first  = safe[safe["event"]=="$ai_generation"].groupby("person_id")["ts"].min()
first_evt = g["ts"].min()
feats["days_to_first_ai"] = ((ai_first - first_evt).dt.total_seconds()/86400).reindex(feats.index, fill_value=-1)
feats["upgraded"]         = feats.index.isin(upgrades.index).astype(int)

feats.to_pickle("reports/user_features.pkl")
print(f"\nSaved: {len(feats):,} users  {feats.shape[1]} cols  -> reports/user_features.pkl")
print(f"Upgraded: {feats.upgraded.sum()}  Non-upgraded: {(feats.upgraded==0).sum()}")
