"""Funnel analysis flow diagram – clean rewrite with edge-to-edge arrows."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── palette ────────────────────────────────────────────────────────────────────
BG     = "#0d1117"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
DIM    = "#3d444d"
BLUE   = "#2f81f7"
GREEN  = "#3fb950"
YELLOW = "#d29922"
ORANGE = "#db6d28"
PURPLE = "#8957e5"
TEAL   = "#26a6a4"
RED    = "#f85149"

W, H = 20, 14
fig = plt.figure(figsize=(W, H), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis("off"); ax.set_facecolor(BG)


# ── helpers ────────────────────────────────────────────────────────────────────
class Node:
    """Box with known center; exposes exact edge midpoints."""
    def __init__(self, cx, cy, w, h):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h

    @property
    def L(self): return (self.cx - self.w / 2, self.cy)
    @property
    def R(self): return (self.cx + self.w / 2, self.cy)
    @property
    def T(self): return (self.cx, self.cy + self.h / 2)
    @property
    def B(self): return (self.cx, self.cy - self.h / 2)


def draw(n, color, label, sub="", lw=1.5):
    ax.add_patch(mpatches.FancyBboxPatch(
        (n.cx - n.w / 2, n.cy - n.h / 2), n.w, n.h,
        boxstyle="round,pad=0.08", linewidth=lw,
        edgecolor=color, facecolor=color + "22", zorder=3
    ))
    yo = 0.12 if sub else 0
    ax.text(n.cx, n.cy + yo, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=TEXT, zorder=4,
            linespacing=1.3)
    if sub:
        ax.text(n.cx, n.cy - 0.24, sub, ha="center", va="center",
                fontsize=7, color=MUTED, zorder=4, linespacing=1.3)


def arr(src, dst, color=DIM, lw=1.3, rad=0.0, style="-|>"):
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=5)


def txt(x, y, s, size=8, color=MUTED, bold=False, ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size,
            fontweight="bold" if bold else "normal", color=color, zorder=6)


def hline(y, x0=0.3, x1=19.7, color=DIM):
    ax.plot([x0, x1], [y, y], color=color, lw=0.7, zorder=2)


def sbg(y0, y1, color):
    ax.add_patch(plt.Rectangle(
        (0.3, y0), 19.4, y1 - y0,
        linewidth=0.8, edgecolor=color + "33",
        facecolor=color + "09", zorder=1
    ))


# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.add_patch(plt.Rectangle((0.3, 13.0), 0.08, 0.85, color=BLUE, zorder=6))
txt(0.65, 13.52, "Funnel Analysis — How It Works",
    size=16, color=TEXT, bold=True, ha="left")
txt(0.65, 13.1,
    "Zerve ODSC Datathon  |  End-to-end pipeline: raw events -> upgrade probability",
    size=8.5, color=MUTED, ha="left")
hline(12.9, color=DIM)


# ══════════════════════════════════════════════════════════════════════════════
# 1  DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
sbg(11.1, 12.85, BLUE)
txt(10, 12.65, "1  DATA PIPELINE", size=8.5, color=BLUE, bold=True)

NW, NH, NY = 2.35, 0.82, 12.0
pipe_spec = [
    (1.55,  BLUE,  "Raw Events",        "3.5M rows  83 cols"),
    (4.35,  MUTED, "Parse Timestamps",  "UTC -> tz-naive"),
    (7.15,  MUTED, "Snapshot Time",     "per-user cutoff"),
    (9.95,  MUTED, "Filter Safe",       "ts < snapshot"),
    (12.75, BLUE,  "Feature Eng.",      "43 features"),
    (15.55, GREEN, "User Table",        "17,541 users"),
]
pipe_nodes = []
for cx, c, lbl, sub in pipe_spec:
    n = Node(cx, NY, NW, NH)
    draw(n, c, lbl, sub)
    pipe_nodes.append((n, c))

for (na, ca), (nb, _) in zip(pipe_nodes, pipe_nodes[1:]):
    arr(na.R, nb.L, color=ca, lw=1.6)

hline(10.95)


# ══════════════════════════════════════════════════════════════════════════════
# 2  FEATURE FAMILIES
# ══════════════════════════════════════════════════════════════════════════════
sbg(9.4, 10.9, PURPLE)
txt(10, 10.72,
    "2  FEATURE FAMILIES  (leakage-safe — all computed before snapshot)",
    size=8.5, color=PURPLE, bold=True)

feat_spec = [
    (1.55,  PURPLE, "Activity",      "total_events\nactive_days\ntenure_days"),
    (4.35,  TEAL,   "AI Usage",      "ai_gen_count\nai_tokens\nai_latency"),
    (7.15,  ORANGE, "Product",       "canvas_creates\nblock_creates\ndeployments"),
    (9.95,  RED,    "Credit",        "credit_pressure\nhas_hit_limit"),
    (12.75, BLUE,   "Time Windows",  "events_7/14/30d\nai_gens_7/14d\ndeploys_14d"),
    (15.55, YELLOW, "Derived",       "events_per_day\nai_pct_events\nis_ramping"),
]
for cx, c, lbl, sub in feat_spec:
    draw(Node(cx, 10.05, 2.35, 1.0), c, lbl, sub)

hline(9.25)


# ══════════════════════════════════════════════════════════════════════════════
# 3  FUNNEL STAGES
# ══════════════════════════════════════════════════════════════════════════════
sbg(6.5, 9.2, TEAL)
txt(10, 9.02,
    "3  DETERMINISTIC FUNNEL STAGES  (highest achieved stage wins)",
    size=8.5, color=TEAL, bold=True)

SW, SH, SY = 1.8, 1.1, 8.2
stage_spec = [
    (1.3,  DIM,    "New",                "< 3 events"),
    (3.4,  BLUE,   "Exploring",          "pageviews\n/ clicks"),
    (5.5,  PURPLE, "Created\nContent",   "canvas / block\n/ file"),
    (7.6,  TEAL,   "AI Engaged",         "ai_gen > 0"),
    (9.7,  ORANGE, "Workflow\nBuilder",  "deploys or\nrun_block > 2"),
    (11.8, RED,    "Credit\nActive",     "credit\npressure > 0"),
    (13.9, YELLOW, "Consistently\nEngaged", "active_days >= 7"),
    (16.7, GREEN,  "Upgraded",           "subscription\nupgraded"),
]
stage_nodes = []
for cx, c, lbl, rule in stage_spec:
    n = Node(cx, SY, SW, SH)
    draw(n, c, lbl, rule)
    stage_nodes.append((n, c))

for (na, ca), (nb, _) in zip(stage_nodes, stage_nodes[1:]):
    arr(na.R, nb.L, color=ca, lw=1.2, rad=0.0)

# At Risk branch — off Consistently Engaged (index 6)
ce, _ = stage_nodes[6]
ar = Node(11.8, 7.3, 1.9, 0.8)
draw(ar, RED, "At Risk", "silent > 14d")
arr(ce.B, ar.R, color=RED, lw=1.3, rad=-0.35)
txt(13.2, 7.7, "no events\nlast 14d", size=6.5, color=RED)

hline(6.65)


# ══════════════════════════════════════════════════════════════════════════════
# 4  ML SCORING & OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
sbg(0.5, 6.6, GREEN)
txt(10, 6.42, "4  ML SCORING & OUTPUT", size=8.5, color=GREEN, bold=True)

# Left column: features -> model -> probability
feat_in = Node(1.3, 5.25, 1.4, 0.72)
draw(feat_in, BLUE, "43\nfeatures", lw=1.2)

xgb = Node(3.9, 5.25, 2.6, 0.9)
draw(xgb, BLUE, "XGBoost Classifier", "5-fold  scale_pos_weight", lw=2)
arr(feat_in.R, xgb.L, color=BLUE, lw=1.6)

prob = Node(7.2, 5.25, 1.9, 0.82)
draw(prob, GREEN, "P(upgrade)", "0 - 100%", lw=1.8)
arr(xgb.R, prob.L, color=BLUE, lw=1.6)

# SHAP and Lift — hang below model and probability
shap = Node(3.9, 3.4, 2.6, 0.82)
draw(shap, PURPLE, "SHAP Explainability", "feature attribution")
arr(xgb.B, shap.T, color=PURPLE, lw=1.2)

lift = Node(7.2, 3.4, 1.9, 0.82)
draw(lift, YELLOW, "Lift Metric", "prob / base_rate")
arr(prob.B, lift.T, color=YELLOW, lw=1.2)

txt(5.5, 4.2, "base rate = 1.84%", size=7, color=MUTED)

# Tiers — fan out from P(upgrade) right edge
tier_spec = [
    ( 9.8, 5.6, "#6b7280", "Low Intent",  "< 1.5x base"),
    (12.3, 5.6, BLUE,      "Moderate",    "< 4x base"),
    (14.8, 5.6, YELLOW,    "High Intent", "< 10x base"),
    (17.3, 5.6, GREEN,     "Very High",   ">= 10x base"),
]
rads = [-0.35, -0.18, 0.0, 0.18]
tier_nodes = []
for (cx, cy, c, lbl, thr), rad in zip(tier_spec, rads):
    n = Node(cx, cy, 2.0, 0.78)
    draw(n, c, lbl, thr)
    arr(prob.R, n.B, color=DIM, lw=0.9, rad=rad)
    tier_nodes.append((n, c))

# Actions below each tier
action_spec = [
    ( 9.8, "Show AI starter"),
    (12.3, "Nudge AI workflow"),
    (14.8, "Prompt deployment"),
    (17.3, "Trigger upgrade"),
]
for (n, c), (cx, lbl) in zip(tier_nodes, action_spec):
    an = Node(cx, 4.05, 1.9, 0.65)
    draw(an, c, lbl, lw=1.2)
    arr(n.B, an.T, color=c, lw=1.0)

# ── footer ─────────────────────────────────────────────────────────────────────
hline(0.88)
txt(10, 0.65,
    "BASE RATE = 1.84%  |  323 upgraders / 17,541 users  |  "
    "XGBoost on 43 leakage-safe features  |  5-fold stratified CV",
    size=7.5, color=DIM)
txt(10, 0.38,
    "Funnel is deterministic: every user is in exactly one stage  |  "
    "At Risk overrides if silent for 14+ days",
    size=7.5, color=DIM)

# ── save ───────────────────────────────────────────────────────────────────────
out = Path("reports/funnel_analysis_flow.png")
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out}  ({out.stat().st_size // 1024} KB)")
plt.close()
