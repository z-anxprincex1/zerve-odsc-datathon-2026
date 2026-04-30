from pathlib import Path

import pandas as pd


DATA_PATH = Path("datasets/zerve_events.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

MD_PATH = REPORT_DIR / "event_type_upgrade_report.md"
CSV_PATH = REPORT_DIR / "event_type_upgrade_report.csv"


def categorize_event(event: str) -> str:
    e = event.lower()
    if event == "subscription_upgraded":
        return "Target outcome"
    if any(k in e for k in ["upgrade", "billing", "subscription", "plan", "promo", "offer", "add_credits", "purchased", "recharge", "charge"]):
        return "Commercial / upgrade intent"
    if any(k in e for k in ["credits", "credit"]):
        return "Credit usage / limits"
    if e.startswith("agent_tool_call") or e.startswith("agent_") or "ask_ai" in e or "ai_" in e or e == "$ai_generation":
        return "AI / agent behavior"
    if any(k in e for k in ["canvas", "block", "edge", "layer", "file", "folder", "asset", "requirements", "source_control", "notebook_import", "notebook_report"]):
        return "Creation / workspace behavior"
    if any(k in e for k in ["deployment", "deploy", "hosted_apps", "api_", "scheduled_job", "persistent_executor", "app_"]):
        return "Deployment / app publishing"
    if any(k in e for k in ["onboarding", "quickstart", "sign_up", "sign_in", "new_user", "$identify", "$groupidentify", "$create_alias", "$set"]):
        return "Lifecycle / onboarding"
    if any(k in e for k in ["referral", "share", "community", "invite", "resource_shared", "link_copied"]):
        return "Sharing / referral"
    if any(k in e for k in ["$pageview", "$pageleave", "$web_vitals", "$autocapture", "$rageclick", "fullscreen", "button_clicked", "link_clicked", "toast"]):
        return "Navigation / UI telemetry"
    if any(k in e for k in ["feedback", "issue", "contact_form", "$exception", "error"]):
        return "Error / feedback"
    return "Other / rare behavior"


def describe_event(event: str, category: str) -> str:
    e = event.lower()
    if event == "subscription_upgraded":
        return "The target conversion event: user upgraded to a paid subscription."
    if category == "Commercial / upgrade intent":
        return "Commercial, billing, plan, offer, or upgrade-flow event indicating pricing intent or subscription movement."
    if category == "Credit usage / limits":
        return "Credit consumption, credit balance threshold, exceeded-credit, or credit-award event."
    if event == "$ai_generation":
        return "AI generation event with model, provider, token, and latency metadata."
    if category == "AI / agent behavior":
        return "AI assistant, agent, message, tool-call, suggestion, or AI workflow interaction."
    if category == "Creation / workspace behavior":
        return "User creates, edits, runs, imports, connects, or manages workspace content such as canvases, blocks, files, folders, layers, or source control."
    if category == "Deployment / app publishing":
        return "Deployment, hosted app, API, scheduled job, or publishing event suggesting movement from exploration to production use."
    if category == "Lifecycle / onboarding":
        return "Signup, signin, onboarding, quickstart, identity, or account lifecycle event."
    if category == "Sharing / referral":
        return "Sharing, referral, invite, or community-distribution event."
    if category == "Navigation / UI telemetry":
        return "Pageview, pageleave, web vital, click, fullscreen, toast, or other UI telemetry event."
    if category == "Error / feedback":
        return "Exception, error-assist, feedback, issue, or support/contact event."
    if "cancel" in e or "downgrade" in e:
        return "Negative subscription lifecycle event indicating cancellation or downgrade."
    return "Rare or miscellaneous event; inspect surrounding rows before using as a feature."


def leakage_risk(event: str, category: str) -> str:
    e = event.lower()
    if event == "subscription_upgraded":
        return "Certain leakage: this is the target and must never be used as a feature."
    if any(k in e for k in ["upgrade", "subscription", "billing", "plan", "promo", "offer", "purchased", "recharge", "charge", "add_credits"]):
        return "High leakage risk: may reveal upgrade funnel, payment intent, subscription state, or downstream commercial action."
    if any(k in e for k in ["credits_exceeded", "credits_below", "credits_remaining", "addon_credits_used", "credits_used"]):
        return "Medium-high leakage risk: valuable usage signal, but may be very close to monetization or post-upgrade behavior."
    if any(k in e for k in ["downgrade", "cancel"]):
        return "High leakage for subscription modeling: reflects downstream subscription lifecycle."
    if category in ["AI / agent behavior", "Creation / workspace behavior", "Deployment / app publishing"]:
        return "Low-medium leakage if restricted to pre-upgrade observation window; strong behavioral signal."
    if category in ["Lifecycle / onboarding", "Navigation / UI telemetry", "Sharing / referral", "Error / feedback"]:
        return "Low leakage if timestamp-filtered before upgrade."
    return "Unknown/low; validate event timing and meaning before modeling."


def intuition(event: str, category: str) -> str:
    e = event.lower()
    if event == "subscription_upgraded":
        return "Use only to define the label and first-upgrade timestamp."
    if category == "Commercial / upgrade intent":
        return "Often directly related to conversion. Great for funnel analysis, but risky for predictive modeling unless the business question explicitly allows near-checkout intent signals."
    if category == "Credit usage / limits":
        return "Can indicate heavy usage or friction from running out of credits. Strong candidate for upgrade propensity, but needs careful time-windowing."
    if category == "AI / agent behavior":
        return "Shows product value and depth of engagement. Good pre-upgrade signal when aggregated over early usage."
    if category == "Creation / workspace behavior":
        return "Shows users building real artifacts. Usually one of the best non-leaky predictors of eventual upgrade."
    if category == "Deployment / app publishing":
        return "Suggests production intent and higher willingness to pay. Very important if observed before upgrade."
    if category == "Lifecycle / onboarding":
        return "Useful for funnel entry, activation timing, and cohort segmentation, but usually weak alone."
    if category == "Sharing / referral":
        return "May indicate collaboration or advocacy. Useful for segmentation and later-stage funnel movement."
    if category == "Navigation / UI telemetry":
        return "Useful for session/activity measurement, but less meaningful than creation, AI, deployment, or credit behavior."
    if category == "Error / feedback":
        return "Can indicate friction. Useful for churn/at-risk analysis and may explain failed conversion."
    return "Potentially useful only after inspecting event context and frequency."


def score_event(event: str, category: str, count: int) -> int:
    e = event.lower()
    score = 10

    if event == "subscription_upgraded":
        return 100

    category_scores = {
        "Commercial / upgrade intent": 95,
        "Credit usage / limits": 82,
        "Deployment / app publishing": 78,
        "Creation / workspace behavior": 72,
        "AI / agent behavior": 68,
        "Sharing / referral": 55,
        "Lifecycle / onboarding": 48,
        "Error / feedback": 42,
        "Navigation / UI telemetry": 32,
        "Other / rare behavior": 25,
    }
    score = category_scores.get(category, score)

    high_signal_terms = [
        "clicked_upgrade",
        "upgrade_subscription",
        "promo_code_redeemed",
        "claim_free_offer",
        "credits_exceeded",
        "credits_below",
        "addon_credits_purchased",
        "add_credits",
        "deployment_deployed",
        "api_deploy",
        "hosted_apps_deploy",
        "source_control",
        "run_block",
        "run_all_blocks",
        "$ai_generation",
        "agent_message",
        "agent_tool_call",
        "canvas_create",
        "block_create",
        "files_upload",
    ]
    if any(term in e for term in high_signal_terms):
        score += 8

    weak_terms = ["$set", "$identify", "$groupidentify", "$create_alias", "$web_vitals", "toast", "fullscreen"]
    if any(term in e for term in weak_terms):
        score -= 10

    rare_negative = ["delete", "undeploy", "unpublish", "downgrade", "cancel"]
    if any(term in e for term in rare_negative):
        score -= 8

    if count >= 100_000:
        score += 4
    elif count >= 10_000:
        score += 3
    elif count >= 1_000:
        score += 1
    elif count < 20:
        score -= 6

    return max(1, min(100, score))


def modeling_recommendation(event: str, category: str) -> str:
    e = event.lower()
    if event == "subscription_upgraded":
        return "Label only. Exclude from features."
    if category == "Commercial / upgrade intent":
        return "Use for funnel diagnostics. Exclude from baseline predictive model or isolate in a separate late-intent model."
    if category == "Credit usage / limits":
        return "Use with strict pre-upgrade windows. Consider counts, first occurrence, and threshold flags."
    if category in ["AI / agent behavior", "Creation / workspace behavior", "Deployment / app publishing"]:
        return "Strong feature candidate when counted before upgrade."
    if category in ["Lifecycle / onboarding", "Sharing / referral"]:
        return "Use for cohort/funnel features and timing features."
    if category == "Navigation / UI telemetry":
        return "Use as aggregate activity/session features, not as isolated one-hot events unless important."
    if category == "Error / feedback":
        return "Use as friction/quality signals; inspect relation to conversion."
    return "Use only after manual validation."


events = pd.read_csv(DATA_PATH, usecols=["event"])["event"].value_counts().rename_axis("event").reset_index(name="count")
events["pct_rows"] = (events["count"] / events["count"].sum() * 100).round(4)
events["category"] = events["event"].map(categorize_event)
events["description"] = events.apply(lambda r: describe_event(r["event"], r["category"]), axis=1)
events["intuition"] = events.apply(lambda r: intuition(r["event"], r["category"]), axis=1)
events["leakage_risk"] = events.apply(lambda r: leakage_risk(r["event"], r["category"]), axis=1)
events["modeling_recommendation"] = events.apply(lambda r: modeling_recommendation(r["event"], r["category"]), axis=1)
events["importance_score"] = events.apply(lambda r: score_event(r["event"], r["category"], int(r["count"])), axis=1)
events["importance_rank"] = events["importance_score"].rank(method="first", ascending=False).astype(int)
events = events.sort_values(["importance_score", "count"], ascending=[False, False])

events.to_csv(CSV_PATH, index=False)

category_summary = (
    events.groupby("category")
    .agg(event_types=("event", "count"), total_rows=("count", "sum"), avg_importance=("importance_score", "mean"))
    .sort_values("avg_importance", ascending=False)
)
category_summary["pct_rows"] = (category_summary["total_rows"] / events["count"].sum() * 100).round(2)
category_summary["avg_importance"] = category_summary["avg_importance"].round(1)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = str(row[col]).replace("\n", " ").replace("|", "\\|")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

with MD_PATH.open("w", encoding="utf-8") as f:
    f.write("# Zerve Event Type Report for Subscription Upgrade Prediction\n\n")
    f.write("This report reviews every distinct value under `df['event']` and evaluates how useful it may be for predicting `subscription_upgraded`.\n\n")
    f.write("## Challenge Framing\n\n")
    f.write("- Target event: `subscription_upgraded`.\n")
    f.write("- Goal: estimate upgrade likelihood using only information known before upgrade.\n")
    f.write("- Main risk: target leakage from upgrade, billing, subscription, offer, promo, or post-upgrade events.\n")
    f.write("- Best feature style: pre-upgrade user-level aggregates, event counts, timing features, and funnel-stage transitions.\n\n")

    f.write("## Event Category Summary\n\n")
    f.write(markdown_table(category_summary.reset_index()))
    f.write("\n\n")

    f.write("## How to Read the Detailed Table\n\n")
    f.write("- `importance_score` is a heuristic product/modeling priority score from 1 to 100.\n")
    f.write("- A high score does not always mean safe to use. Some high-score events are high-leakage and should be label-only or funnel-diagnostic only.\n")
    f.write("- Use `leakage_risk` and `modeling_recommendation` before deciding whether an event belongs in model features.\n\n")

    f.write("## Detailed Event-Type Review\n\n")
    detailed_cols = [
        "importance_rank",
        "event",
        "count",
        "pct_rows",
        "category",
        "description",
        "intuition",
        "leakage_risk",
        "modeling_recommendation",
        "importance_score",
    ]
    f.write(markdown_table(events[detailed_cols]))
    f.write("\n\n")

    f.write("## Ranked Event Types by Importance for Subscription Upgrade Conversion\n\n")
    f.write("This ranking combines likely product signal, proximity to conversion, event frequency, and expected usefulness for funnel/modeling work. Treat high-leakage events as important for understanding the funnel, not automatically safe model features.\n\n")
    rank_cols = ["importance_rank", "event", "importance_score", "category", "leakage_risk", "modeling_recommendation"]
    f.write(markdown_table(events[rank_cols]))
    f.write("\n\n")

    f.write("## Recommended Modeling Use\n\n")
    f.write("1. Use `subscription_upgraded` only to define the label and first-upgrade timestamp.\n")
    f.write("2. Build a strict observation window before upgrade, such as first 24 hours, first 3 days, first 7 days, or first N events.\n")
    f.write("3. Exclude all events at or after each user's first upgrade timestamp.\n")
    f.write("4. Start with safer behavioral families: creation/workspace behavior, AI/agent behavior, deployment behavior, and early lifecycle/onboarding.\n")
    f.write("5. Treat commercial and credit events as separate experiments because they may be extremely predictive but leakage-prone.\n")

print(MD_PATH.resolve())
print(CSV_PATH.resolve())
