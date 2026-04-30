import pandas as pd
import numpy as np

print("Loading events...")
df = pd.read_csv('datasets/zerve_events.csv')
df['ts'] = pd.to_datetime(df['timestamp'], utc=True)
df = df.sort_values('ts')

# ── Find upgraders ─────────────────────────────────────────────────────────────
upgrade_events = df[df['event'] == 'subscription_upgraded']
upgraders      = set(upgrade_events['person_id'].unique())
print(f"\nTotal users:    {df['person_id'].nunique():,}")
print(f"Upgraders:      {len(upgraders):,}  ({len(upgraders)/df['person_id'].nunique()*100:.2f}%)")

# ── For each upgrader: get their FIRST upgrade timestamp ──────────────────────
first_upgrade = upgrade_events.groupby('person_id')['ts'].min().rename('upgrade_ts')

# ── Compute pre-upgrade behavior for upgraders ─────────────────────────────────
# Only events strictly before their upgrade
pre_upgrade = (
    df[df['person_id'].isin(upgraders)]
    .merge(first_upgrade, on='person_id')
    .query('ts < upgrade_ts')
)

# ── Compute behavior for non-upgraders ────────────────────────────────────────
non_upgraders_df = df[~df['person_id'].isin(upgraders)]

DEPLOY_EVENTS = {'notebook_deployment_usage_tracked', 'notebook_deployed',
                 'app_deployed', 'deployment_created'}
AI_EVENT      = '$ai_generation'

def compute_features(grp_df):
    g = grp_df.groupby('person_id')

    feats = pd.DataFrame({
        'total_events':      g['event'].count(),
        'active_days':       g['ts'].apply(lambda x: x.dt.date.nunique()),
        'tenure_days':       g['ts'].apply(lambda x: (x.max() - x.min()).days + 1),
        'ai_gen_count':      g['event'].apply(lambda x: (x == AI_EVENT).sum()),
        'deploy_count':      g['event'].apply(lambda x: x.isin(DEPLOY_EVENTS).sum()),
        'canvas_creates':    g['event'].apply(lambda x: (x == 'canvas_create').sum()),
        'block_creates':     g['event'].apply(lambda x: (x.isin({'block_create','block_created'})).sum()),
        'unique_events':     g['event'].nunique(),
    })

    # Recent activity windows
    now = grp_df['ts'].max()
    w7  = grp_df[grp_df['ts'] >= now - pd.Timedelta(days=7)]
    w14 = grp_df[grp_df['ts'] >= now - pd.Timedelta(days=14)]

    feats['events_last_7d']     = w7.groupby('person_id')['event'].count().reindex(feats.index, fill_value=0)
    feats['ai_gens_last_7d']    = w7.groupby('person_id')['event'].apply(lambda x: (x == AI_EVENT).sum()).reindex(feats.index, fill_value=0)
    feats['events_last_14d']    = w14.groupby('person_id')['event'].count().reindex(feats.index, fill_value=0)
    feats['deploys_last_14d']   = w14.groupby('person_id')['event'].apply(lambda x: x.isin(DEPLOY_EVENTS).sum()).reindex(feats.index, fill_value=0)

    feats['has_used_ai']        = (feats['ai_gen_count'] > 0).astype(int)
    feats['has_deployed']       = (feats['deploy_count'] > 0).astype(int)
    feats['pct_ai']             = feats['ai_gen_count'] / feats['total_events'].clip(lower=1)

    return feats

print("\nComputing upgrader features (pre-upgrade behavior)...")
up_feats  = compute_features(pre_upgrade)

print("Computing non-upgrader features...")
non_feats = compute_features(non_upgraders_df)

# ── Comparison table ───────────────────────────────────────────────────────────
cols = ['total_events','active_days','tenure_days','ai_gen_count','deploy_count',
        'canvas_creates','block_creates','unique_events',
        'events_last_7d','ai_gens_last_7d','events_last_14d','deploys_last_14d',
        'has_used_ai','has_deployed','pct_ai']

print("\n" + "="*70)
print(f"{'Feature':<25} {'Upgraders (median)':>20} {'Non-upgraders (med)':>20}")
print("="*70)
for col in cols:
    u_med = up_feats[col].median()
    n_med = non_feats[col].median()
    ratio = f"({u_med/max(n_med,0.001):.1f}x)" if n_med > 0 else ""
    print(f"{col:<25} {u_med:>20.1f} {n_med:>20.1f}  {ratio}")

print("\n" + "="*70)
print("Upgrader %ile breakdown (what fraction of upgraders had each behavior):")
print(f"  Has used AI:     {up_feats['has_used_ai'].mean()*100:.1f}%")
print(f"  Has deployed:    {up_feats['has_deployed'].mean()*100:.1f}%")
print(f"  AI gens > 10:    {(up_feats['ai_gen_count'] > 10).mean()*100:.1f}%")
print(f"  AI gens > 50:    {(up_feats['ai_gen_count'] > 50).mean()*100:.1f}%")
print(f"  Deploys > 0:     {(up_feats['deploy_count'] > 0).mean()*100:.1f}%")
print(f"  Active days > 7: {(up_feats['active_days'] > 7).mean()*100:.1f}%")

print("\nUpgrader tenure distribution:")
for p in [25, 50, 75, 90]:
    print(f"  p{p}: {np.percentile(up_feats['tenure_days'].dropna(), p):.0f} days")
