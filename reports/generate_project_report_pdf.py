from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "zerve_upgrade_prediction_project_report.pdf"

BLUE = "#1565c0"
DARK = "#1f2933"
MUTED = "#53606d"
LIGHT = "#eef3f8"
GREEN = "#2e7d32"
ORANGE = "#ef6c00"
RED = "#c62828"


def add_header(fig, title: str, subtitle: str | None = None) -> None:
    fig.text(0.06, 0.94, title, fontsize=18, fontweight="bold", color=DARK)
    fig.add_artist(
        plt.Line2D([0.06, 0.94], [0.915, 0.915], color=BLUE, linewidth=2)
    )
    if subtitle:
        fig.text(0.06, 0.89, subtitle, fontsize=10.5, color=MUTED)


def add_wrapped_text(
    fig,
    text: str,
    x: float,
    y: float,
    width: int = 92,
    fontsize: float = 10.5,
    line_height: float = 0.032,
    color: str = DARK,
    weight: str = "normal",
) -> float:
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            y -= line_height * 0.55
            continue
        wrapped = textwrap.wrap(paragraph, width=width) or [""]
        for line in wrapped:
            fig.text(x, y, line, fontsize=fontsize, color=color, fontweight=weight)
            y -= line_height
    return y


def text_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]]) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    add_header(fig, title)
    y = 0.86
    for heading, body in sections:
        fig.text(0.06, y, heading, fontsize=12.5, fontweight="bold", color=BLUE)
        y -= 0.038
        y = add_wrapped_text(fig, body, 0.08, y, width=92)
        y -= 0.025
        if y < 0.12:
            break
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def table_page(
    pdf: PdfPages,
    title: str,
    subtitle: str,
    columns: list[str],
    rows: list[list[str]],
    col_widths: list[float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor="white")
    ax.axis("off")
    add_header(fig, title, subtitle)
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colLoc="left",
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1, 1.8)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d7dee6")
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f5f8fb")
            cell.set_text_props(color=DARK)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def figure_page(
    pdf: PdfPages,
    title: str,
    image_name: str,
    caption: str,
    explanation: str,
    inference: str,
) -> None:
    path = REPORTS / image_name
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    add_header(fig, title, caption)
    if not path.exists():
        fig.text(0.08, 0.5, f"Missing figure: {image_name}", fontsize=14, color=RED)
    else:
        img = mpimg.imread(path)
        ax = fig.add_axes([0.06, 0.28, 0.88, 0.56])
        ax.imshow(img)
        ax.axis("off")

    fig.patches.append(
        plt.Rectangle(
            (0.06, 0.06),
            0.88,
            0.17,
            transform=fig.transFigure,
            facecolor="#f5f8fb",
            edgecolor="#d7dee6",
            linewidth=1,
        )
    )
    fig.text(0.08, 0.195, "What this output shows", fontsize=10.5, fontweight="bold", color=BLUE)
    add_wrapped_text(fig, explanation, 0.08, 0.17, width=112, fontsize=9.2, line_height=0.024)
    fig.text(0.08, 0.118, "What can be inferred", fontsize=10.5, fontweight="bold", color=GREEN)
    add_wrapped_text(fig, inference, 0.08, 0.094, width=112, fontsize=9.2, line_height=0.024)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def cover_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.patches.extend(
        [
            plt.Rectangle((0, 0.78), 1, 0.22, transform=fig.transFigure, color=DARK),
            plt.Rectangle((0, 0.76), 1, 0.02, transform=fig.transFigure, color=BLUE),
        ]
    )
    fig.text(
        0.07,
        0.9,
        "Zerve Upgrade Prediction",
        fontsize=28,
        fontweight="bold",
        color="white",
    )
    fig.text(
        0.07,
        0.84,
        "Project Report from Notebook Analysis",
        fontsize=16,
        color="#dbe7f3",
    )
    fig.text(0.07, 0.68, "Scope", fontsize=14, fontweight="bold", color=BLUE)
    body = (
        "This report summarizes the notebook work for the Zerve telemetry challenge: "
        "data understanding, leakage-safe feature engineering, upgrade prediction, "
        "funnel design, behavioral insights, and visual evidence. It intentionally "
        "excludes the interactive application layer."
    )
    add_wrapped_text(fig, body, 0.07, 0.63, width=82, fontsize=11.5)
    bullets = [
        "Dataset: 3,509,628 telemetry rows across 83 columns.",
        "Target: subscription_upgraded, modeled at the user level.",
        "Final feature table: 16,496 users and 43 predictive features.",
        "Model: XGBoost with stratified 5-fold cross-validation.",
        "Funnel: deterministic, mutually exclusive user stages.",
    ]
    y = 0.48
    for bullet in bullets:
        fig.text(0.09, y, f"- {bullet}", fontsize=11, color=DARK)
        y -= 0.04
    fig.text(0.07, 0.12, f"Generated file: {OUTPUT.name}", fontsize=9.5, color=MUTED)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        cover_page(pdf)
        text_page(
            pdf,
            "Executive Summary",
            [
                (
                    "Objective",
                    "Predict which users are likely to upgrade using only pre-upgrade "
                    "behavior, then translate model outputs into practical growth and "
                    "product actions.",
                ),
                (
                    "Core Method",
                    "The analysis builds a per-user snapshot time: first upgrade moment "
                    "for upgraders and last observed event for non-upgraders. Features "
                    "are computed only before that snapshot, with commercial and "
                    "checkout-adjacent events excluded from the modeling pipeline.",
                ),
                (
                    "Model Result",
                    "The XGBoost model achieved 0.7828 +/- 0.0156 ROC-AUC across "
                    "5 stratified folds, with 0.0928 +/- 0.0187 PR-AUC. The PR-AUC is "
                    "the more realistic secondary metric because the target class is "
                    "rare.",
                ),
                (
                    "Product Result",
                    "Users are bucketed into Low, Moderate, High, and Very High intent "
                    "tiers. The tiers attach model scores to recommended actions such "
                    "as onboarding nudges, contextual prompts, personalized outreach, "
                    "and priority sales contact.",
                ),
                (
                    "Funnel Result",
                    "A deterministic funnel assigns every user to exactly one current "
                    "stage: New, Exploring, Builder, AI User, Power User, Upgraded, or "
                    "At Risk. The funnel uses observable event-count thresholds rather "
                    "than model predictions.",
                ),
            ],
        )
        text_page(
            pdf,
            "Data Understanding and Leakage Controls",
            [
                (
                    "Dataset",
                    "The notebook analyzes zerve_events.csv, containing 3,509,628 rows "
                    "and 83 columns of product telemetry. The raw event stream includes "
                    "227 distinct event types across pageviews, AI usage, canvas and "
                    "block activity, deployments, credits, subscription actions, and "
                    "person-level signup fields.",
                ),
                (
                    "Missingness",
                    "Sixty of the 83 columns contain null values. This is expected "
                    "because many fields are event-specific: AI token and latency fields "
                    "populate only for AI generation events, while browser and GeoIP "
                    "fields populate only for browser-sourced events.",
                ),
                (
                    "Leakage Prevention",
                    "The notebook excludes direct target events and high-risk commercial "
                    "signals from feature generation. It also cuts each user's history "
                    "at the snapshot time so the model cannot learn from post-upgrade "
                    "behavior.",
                ),
                (
                    "Feature Table",
                    "After leakage filtering, 2,890,904 safe events remain. The final "
                    "modeling table contains 16,496 users, 43 predictive features, and "
                    "one label.",
                ),
            ],
        )
        table_page(
            pdf,
            "Modeling Features",
            "Representative feature families created before each user snapshot.",
            ["Feature Family", "Examples", "Why It Matters"],
            [
                [
                    "Engagement volume",
                    "total_events, active_days, session_count",
                    "Captures depth and consistency of product usage.",
                ],
                [
                    "AI usage",
                    "ai_gen_count, ai_tokens_total, ai_latency_avg",
                    "Measures activation around the product's high-value AI workflow.",
                ],
                [
                    "Recent activity",
                    "events_last_7d, events_last_14d, pct_activity_last_7d",
                    "Separates active prospects from stale users.",
                ],
                [
                    "Creation behavior",
                    "canvas_create_count, block_create_count, files_upload_count",
                    "Signals movement from browsing to building.",
                ],
                [
                    "Production intent",
                    "deployment_count, notebook_deploy_usage",
                    "Captures users moving toward operational use.",
                ],
                [
                    "Credit pressure",
                    "credit_pressure_count, has_hit_credit_limit",
                    "Identifies quota moments that can motivate upgrade.",
                ],
            ],
            [0.18, 0.34, 0.42],
        )
        figure_page(
            pdf,
            "Model Performance",
            "fig1_model_performance.png",
            "ROC and precision-recall performance for the leakage-safe XGBoost model.",
            "The ROC curve measures how well the classifier separates upgraders from non-upgraders across thresholds. The precision-recall curve focuses on positive-class retrieval, which is important because upgrades are rare.",
            "The model has meaningful rank-ordering power, but the low base rate means teams should use it for prioritization and tiering rather than treating each probability as a guaranteed outcome.",
        )
        figure_page(
            pdf,
            "Feature Importance",
            "fig2_feature_importance.png",
            "Top model features emphasize AI usage, recent activity, credit pressure, and engagement depth.",
            "This chart ranks the features XGBoost used most heavily when splitting users into higher and lower upgrade-propensity groups.",
            "AI token volume, recent events, credit pressure, total activity, and breadth of event types are among the strongest signals, suggesting upgrade intent is tied to both value discovery and quota pressure.",
        )
        figure_page(
            pdf,
            "SHAP Interpretability",
            "fig9a_shap_beeswarm.png",
            "Per-user SHAP effects show how feature values push upgrade probability up or down.",
            "Each dot is a user-level contribution for one feature. Position shows whether the feature pushed the prediction higher or lower; color shows whether that user had a high or low value for the feature.",
            "High AI usage, recent activity, and credit-pressure related behavior generally push predictions upward, while low engagement patterns push users toward lower upgrade probability.",
        )
        figure_page(
            pdf,
            "Global SHAP Ranking",
            "fig9b_shap_bar.png",
            "Mean absolute SHAP values provide a global feature importance ranking.",
            "The bar chart aggregates absolute SHAP effects across users to show which features most consistently influence model predictions.",
            "The same behavioral themes dominate globally: AI usage, recent engagement, credit pressure, and creation/run activity. This supports a product strategy centered on activation depth.",
        )
        figure_page(
            pdf,
            "Actionable Risk Tiers",
            "fig10_risk_tiers.png",
            "Probability buckets convert raw model scores into growth team actions.",
            "Users are grouped into Low, Moderate, High, and Very High intent tiers, with user counts, upgrader counts, conversion rates, and recommended actions.",
            "The Very High tier has the strongest conversion concentration, so it should receive scarce high-touch resources. Lower tiers are better suited to scalable nudges and onboarding.",
        )
        table_page(
            pdf,
            "Funnel Definition",
            "Deterministic stages based only on observable behavior.",
            ["Stage", "Rule"],
            [
                ["New", "Any event but no meaningful product interaction."],
                ["Exploring", "At least one pageview."],
                ["Builder", "Created at least one canvas or block."],
                ["AI User", "Used AI at least once."],
                ["Power User", "Deployed something or ran blocks at least five times."],
                ["Upgraded", "Has a subscription_upgraded event."],
                ["At Risk", "Was Builder or higher and inactive for 14+ days."],
            ],
            [0.22, 0.72],
        )
        figure_page(
            pdf,
            "Funnel Distribution",
            "fig3_funnel_chart.png",
            "Current-stage distribution across the user base.",
            "The funnel chart assigns every user to a deterministic stage based on observed product behavior, from New through Upgraded and At Risk.",
            "A large share of users remain New, Exploring, or At Risk, which points to two opportunities: improve early activation and re-engage previously meaningful users before they decay.",
        )
        figure_page(
            pdf,
            "Stage Transition Matrix",
            "fig11_transition_matrix.png",
            "Thirty-day movement validates the stage rules and progression logic.",
            "The matrix compares user stage thirty days before the dataset end with current stage, showing how users progressed, stayed still, upgraded, or became at risk.",
            "Most users do not move quickly through the funnel, so conversion work should focus on accelerating key transitions: Exploring to Builder, Builder to AI User, and AI User to Power User.",
        )
        figure_page(
            pdf,
            "Time to Stage",
            "fig12_time_to_stage.png",
            "Median time from first event to each funnel milestone.",
            "This output shows the median number of days from first observed event to major milestones such as first pageview, first build action, first AI generation, deployment, and upgrade.",
            "Important milestones often happen very early. That implies onboarding has a narrow window to create value, and the first session or first few days are critical.",
        )
        text_page(
            pdf,
            "Behavioral Insights",
            [
                (
                    "1. AI power users drive upgrades",
                    "Upgraders averaged 86.51 AI generations versus 25.89 for "
                    "non-upgraders, a 3.3x lift. Credit pressure also showed a 3.5x "
                    "lift, supporting contextual upgrade nudges near AI usage limits.",
                ),
                (
                    "2. Deployment intent is a strong signal",
                    "Users with deployment behavior upgraded at 9.80% versus 1.84% "
                    "for non-deployers, a 5.3x lift. This suggests first deployment "
                    "is a high-intent lifecycle moment.",
                ),
                (
                    "3. Early activation matters",
                    "AI users upgraded at 4.011% overall versus 1.088% for non-AI "
                    "users. The notebook recommends onboarding that gets users to "
                    "their first AI generation quickly.",
                ),
            ],
        )
        figure_page(
            pdf,
            "AI Usage and Credit Pressure",
            "fig4_insight1_ai_usage.png",
            "Comparison of AI and credit-pressure behavior by upgrade outcome.",
            "The plots compare AI generation behavior and credit-pressure behavior between users who upgraded and those who did not.",
            "Upgraders show much heavier AI usage and more credit pressure, suggesting that users who repeatedly experience AI value and resource limits are more likely to pay.",
        )
        figure_page(
            pdf,
            "Daily Event Volume",
            "fig5_daily_trend.png",
            "Event-volume trend used to inspect activity patterns and data gaps.",
            "This time-series view shows aggregate event volume across the observation period and helps reveal spikes, troughs, and possible data collection gaps.",
            "Activity is not perfectly uniform over time, so future validation should consider temporal holdouts to ensure model performance generalizes beyond the observed activity pattern.",
        )
        figure_page(
            pdf,
            "Retention Cohorts",
            "fig6_retention_heatmap.png",
            "Cohort retention view across day 0, 1, 3, 7, 14, and 30.",
            "Users are grouped by signup cohort, then measured by the percentage who return on later lifecycle days.",
            "Retention appears to decline quickly after initial usage, so activation quality and early habit formation are likely important levers for downstream conversion.",
        )
        figure_page(
            pdf,
            "Persona Segmentation",
            "fig7_segmentation.png",
            "Upgrade rate and user distribution by signup persona fields.",
            "The segmentation charts compare upgrade behavior across available signup attributes such as purpose, role, source, or work type.",
            "Some personas and acquisition contexts appear more commercially qualified than others, meaning messaging and onboarding can be tailored by signup intent.",
        )
        figure_page(
            pdf,
            "Events Per User Distribution",
            "fig8_distribution.png",
            "Distributional comparison of activity depth for upgraders and non-upgraders.",
            "The distribution compares total event counts for upgraders and non-upgraders, showing how activity depth differs across outcomes.",
            "Upgraders tend to be more active, but activity alone is not enough; the model benefits from combining volume with AI usage, recency, deployment, and credit-pressure features.",
        )
        text_page(
            pdf,
            "Conclusion",
            [
                (
                    "What worked",
                    "The project combines methodological rigor with operational "
                    "usefulness: leakage-safe snapshots, class-imbalance-aware "
                    "evaluation, interpretable model outputs, action tiers, and a "
                    "deterministic funnel that can be implemented directly.",
                ),
                (
                    "Recommended next steps",
                    "Validate calibration on a future holdout period, monitor tier "
                    "conversion rates after launch, A/B test AI quota messaging, and "
                    "instrument first-deployment upgrade prompts.",
                ),
                (
                    "Submission readiness",
                    "The notebook covers the predictive modeling challenge, funnel "
                    "challenge, and behavioral insight requirements without relying on "
                    "post-upgrade leakage.",
                ),
            ],
        )
    print(OUTPUT)


if __name__ == "__main__":
    main()
