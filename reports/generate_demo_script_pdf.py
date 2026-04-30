from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "zerve_demo_recording_script.pdf"

BLUE = "#1565c0"
DARK = "#1f2933"
MUTED = "#53606d"
LIGHT = "#f5f8fb"
GREEN = "#2e7d32"
ORANGE = "#ef6c00"


def add_header(fig, title: str, subtitle: str | None = None) -> None:
    fig.text(0.055, 0.94, title, fontsize=18, fontweight="bold", color=DARK)
    fig.add_artist(
        plt.Line2D([0.055, 0.945], [0.915, 0.915], color=BLUE, linewidth=2)
    )
    if subtitle:
        fig.text(0.055, 0.887, subtitle, fontsize=10, color=MUTED)


def wrapped(
    fig,
    text: str,
    x: float,
    y: float,
    width: int = 108,
    fontsize: float = 9.2,
    line_height: float = 0.024,
    color: str = DARK,
    weight: str = "normal",
) -> float:
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            y -= line_height * 0.5
            continue
        for line in textwrap.wrap(paragraph, width=width) or [""]:
            fig.text(x, y, line, fontsize=fontsize, color=color, fontweight=weight)
            y -= line_height
    return y


def cover(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.patches.extend(
        [
            plt.Rectangle((0, 0.78), 1, 0.22, transform=fig.transFigure, color=DARK),
            plt.Rectangle((0, 0.76), 1, 0.02, transform=fig.transFigure, color=BLUE),
        ]
    )
    fig.text(0.07, 0.9, "Demo Recording Script", fontsize=30, fontweight="bold", color="white")
    fig.text(0.07, 0.84, "Zerve Upgrade Prediction Notebook", fontsize=16, color="#dbe7f3")
    fig.text(0.07, 0.68, "How to use this PDF", fontsize=14, fontweight="bold", color=BLUE)
    body = (
        "Use this as presenter notes while recording. Each segment tells you which "
        "notebook cells to open, which output or figure to point at, what to say, "
        "and what conclusion the audience should take away. The flow excludes the "
        "Streamlit app and focuses only on the notebook project."
    )
    wrapped(fig, body, 0.07, 0.63, width=82, fontsize=11)
    bullets = [
        "Suggested length: 6 to 9 minutes.",
        "Start from the top of t1.ipynb and scroll only to the referenced cells.",
        "Spend the most time on leakage safety, model results, SHAP, funnel, and insights.",
        "Do not explain every line of code. Explain the decision and the output.",
    ]
    y = 0.49
    for bullet in bullets:
        fig.text(0.09, y, f"- {bullet}", fontsize=10.8, color=DARK)
        y -= 0.04
    fig.text(0.07, 0.12, f"Generated file: {OUTPUT.name}", fontsize=9.5, color=MUTED)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def overview_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    add_header(fig, "Demo Flow Overview", "Use this as the recording order.")
    rows = [
        ("0:00-0:45", "Cells 0-5", "Dataset and target", "Show shape: 3,509,628 rows x 83 columns."),
        ("0:45-1:30", "Cells 6-24", "EDA and leakage scan", "Explain missingness and high-risk event filtering."),
        ("1:30-2:30", "Cells 26-31", "Feature engineering", "Show snapshot cutoff and final 16,496 x 44 feature table."),
        ("2:30-3:45", "Cells 32-36", "Upgrade model", "Show ROC-AUC, PR-AUC, and feature importance."),
        ("3:45-4:45", "Cells 37-40", "Interpretability and tiers", "Show SHAP and risk-tier outputs."),
        ("4:45-5:55", "Cells 41-46", "Funnel", "Show deterministic stages, funnel chart, transitions, time-to-stage."),
        ("5:55-7:20", "Cells 47-54", "Behavioral insights", "Explain AI power users, deployment intent, early activation."),
        ("7:20-8:30", "Cells 55-63", "Supporting visuals and wrap", "Show trend, retention, segmentation, distribution, checklist."),
    ]
    table = plt.table(
        cellText=[list(r) for r in rows],
        colLabels=["Time", "Cells", "Segment", "Output to show"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.12, 0.13, 0.24, 0.46],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    table.scale(1, 1.9)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d7dee6")
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else LIGHT)
            cell.set_text_props(color=DARK)
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def segment_page(
    pdf: PdfPages,
    number: int,
    title: str,
    cells: str,
    output: str,
    talk_track: str,
    inference: str,
    transition: str,
) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    add_header(fig, f"{number}. {title}", f"Notebook cells: {cells}")

    fig.patches.append(
        plt.Rectangle((0.06, 0.79), 0.88, 0.08, transform=fig.transFigure, facecolor=LIGHT, edgecolor="#d7dee6")
    )
    fig.text(0.08, 0.84, "Output to point at", fontsize=11.5, fontweight="bold", color=BLUE)
    wrapped(fig, output, 0.08, 0.815, width=86, fontsize=10, line_height=0.026)

    y = 0.735
    fig.text(0.06, y, "What to say", fontsize=12.5, fontweight="bold", color=BLUE)
    y -= 0.04
    y = wrapped(fig, talk_track, 0.08, y, width=86, fontsize=10.2, line_height=0.03)

    y -= 0.02
    fig.text(0.06, y, "What can be inferred", fontsize=12.5, fontweight="bold", color=GREEN)
    y -= 0.04
    y = wrapped(fig, inference, 0.08, y, width=86, fontsize=10.2, line_height=0.03)

    y -= 0.02
    fig.text(0.06, y, "Transition line", fontsize=12.5, fontweight="bold", color=ORANGE)
    y -= 0.04
    wrapped(fig, transition, 0.08, y, width=86, fontsize=10.2, line_height=0.03)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def appendix_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    add_header(fig, "Quick Reference: Figures and Cells", "Use this when you need to jump during recording.")
    refs = [
        ("Fig 1", "Cell 35", "Model performance: ROC and precision-recall curves."),
        ("Fig 2", "Cell 36", "Feature importance: top XGBoost features."),
        ("Fig 9a/b", "Cell 38", "SHAP beeswarm and global SHAP bar chart."),
        ("Fig 10", "Cell 40", "Actionable upgrade risk tiers."),
        ("Fig 3", "Cell 43", "Funnel distribution."),
        ("Fig 11", "Cell 45", "Thirty-day stage transition matrix."),
        ("Fig 12", "Cell 46", "Median time-to-stage progression."),
        ("Fig 4", "Cell 50", "AI usage and upgrade outcome comparison."),
        ("Fig 5", "Cell 56", "Daily event-volume trend."),
        ("Fig 6", "Cell 58", "Retention cohort heatmap."),
        ("Fig 7", "Cell 60", "Persona segmentation."),
        ("Fig 8", "Cell 62", "Events-per-user distribution."),
    ]
    y = 0.84
    for fig_id, cell, desc in refs:
        fig.text(0.08, y, fig_id, fontsize=10.5, fontweight="bold", color=BLUE)
        fig.text(0.20, y, cell, fontsize=10.5, fontweight="bold", color=DARK)
        wrapped(fig, desc, 0.34, y, width=54, fontsize=10.2, line_height=0.026)
        y -= 0.055
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    segments = [
        (
            "Project Setup and Dataset",
            "0-5",
            "Cell 4 output: (3509628, 83). Cell 5 output: column list with person_id, timestamp, event, person properties, AI fields, credits, deployment and product activity fields.",
            "I start by loading the Zerve telemetry dataset. The key point is that this is event-level product data, not one row per user. The target event is subscription_upgraded, so the challenge is to turn this raw event stream into a user-level prediction problem.",
            "The dataset is large enough to build behavioral features, but the target is rare. That means accuracy is not a useful metric, and the pipeline needs to be careful about leakage.",
            "Next I check data quality and identify which event types are safe to use before modeling.",
        ),
        (
            "EDA and Leakage Risk",
            "6-24",
            "Cells 8-11: null coverage summary. Cell 22: top upgrade/commercial event types. Cell 24: leakage-risk categories.",
            "This section shows that many columns are sparse because they only apply to certain event types. That is expected for telemetry. More importantly, I separate normal behavioral signals from commercial or checkout-adjacent signals that could leak the answer.",
            "Columns and events close to billing, checkout, subscription, credits, or the upgrade target should not be used naively. The model must predict upgrade intent before the user upgrades, not recognize upgrade behavior after it has happened.",
            "With leakage risks identified, I move into the most important part: building the user snapshot and feature table.",
        ),
        (
            "Leakage-Safe Feature Engineering",
            "26-31",
            "Cell 27: total users, upgraders, conversion rate. Cell 28: safe events retained and dropped. Cell 31: final feature table, 16,496 users x 44 columns.",
            "Here I define a snapshot time for every user. For upgraders, the snapshot is their first subscription_upgraded event. For non-upgraders, it is their last observed event. Every feature is computed only from events before that snapshot.",
            "This makes the prediction setup realistic. The final table has 43 predictive features plus the label, including activity volume, recent activity, AI usage, credit pressure, deployment, sessions, and persona attributes.",
            "Now that the feature table is safe, I can train and evaluate the upgrade prediction model.",
        ),
        (
            "Upgrade Prediction Model",
            "32-36",
            "Cell 33: X shape and positive rate. Cell 34: fold metrics and CV summary. Cell 35: Fig 1 model performance. Cell 36: Fig 2 feature importance.",
            "The model is XGBoost with stratified 5-fold cross-validation. I use ROC-AUC and PR-AUC because the class is imbalanced. The key result is around 0.78 ROC-AUC, which shows meaningful separation between likely upgraders and non-upgraders.",
            "The feature importance output shows that upgrade intent is most associated with AI token usage, recent events, credit pressure, total activity, AI generation counts, and product-building behavior.",
            "A model score is useful, but for a product team we also need to explain why users score high and how to act on the score.",
        ),
        (
            "Interpretability and Risk Tiers",
            "37-40",
            "Cell 38: Fig 9a SHAP beeswarm and Fig 9b SHAP bar. Cell 40: risk-tier table and Fig 10.",
            "SHAP explains the model at both the user level and global level. The beeswarm shows whether high or low values push individual predictions up or down. The bar chart shows the strongest overall drivers.",
            "The risk-tier output turns probabilities into product actions. Very High users have the strongest conversion concentration and deserve high-touch outreach, while Low and Moderate users are better served with scalable nudges.",
            "After modeling individual upgrade probability, I switch to the second challenge: a deterministic funnel that explains where users are in the product journey.",
        ),
        (
            "Deterministic User Funnel",
            "41-46",
            "Cell 42: funnel distribution table. Cell 43: Fig 3 funnel chart. Cell 45: Fig 11 transition matrix. Cell 46: Fig 12 time-to-stage.",
            "The funnel assigns every user to exactly one stage based on observable rules: New, Exploring, Builder, AI User, Power User, Upgraded, or At Risk. This is not a model; it is deterministic logic that an engineer could implement.",
            "A large share of users are New, Exploring, or At Risk. The transition matrix shows how users move over 30 days, and the time-to-stage chart shows that important milestones often happen very early.",
            "With the model and funnel in place, I use them to produce product insights and recommended actions.",
        ),
        (
            "Behavioral Insight 1: AI Usage",
            "47-50",
            "Cell 49: mean and median comparison table. Cell 50: Fig 4 AI usage vs upgrade outcome.",
            "The first insight is that AI power users are much more likely to upgrade. Upgraders average far more AI generations and more credit-pressure events than non-upgraders.",
            "This suggests that AI usage creates perceived value, and credit limits create a natural upgrade moment. A good action is a contextual upgrade nudge inside the AI workflow when usage or token pressure is high.",
            "The next insight looks at an even stronger production-intent signal: deployment.",
        ),
        (
            "Behavioral Insight 2: Deployment Intent",
            "51-52",
            "Cell 52: upgrade rate by deployment and AI usage flags.",
            "Deployment is a strong signal because it means the user is moving from exploration into production or sharing. The notebook shows deployers upgrade at a much higher rate than non-deployers.",
            "The inference is that deployment should trigger a different kind of message than generic AI usage. The pitch should focus on reliability, production capacity, and collaboration.",
            "The final behavioral insight looks at how quickly users reach value after signup.",
        ),
        (
            "Behavioral Insight 3: Early Activation",
            "53-54",
            "Cell 54: upgrade rate by time-to-first-AI window, plus AI vs non-AI upgrade rate.",
            "This section compares users based on when they first use AI. The bigger point is that users who reach meaningful product value are more likely to convert than users who never use AI.",
            "The product implication is that onboarding should push users toward a first AI generation quickly, ideally in the first session or first few days.",
            "I finish the demo by showing supporting visuals and the submission checklist.",
        ),
        (
            "Supporting Visuals and Wrap-Up",
            "55-63",
            "Cell 56: Fig 5 daily trend. Cell 58: Fig 6 retention. Cell 60: Fig 7 segmentation. Cell 62: Fig 8 distribution. Cell 63: checklist.",
            "These final figures support the overall story: product activity changes over time, retention drops by cohort, some personas convert better than others, and upgraders tend to be more active.",
            "The final checklist ties the work back to the challenge requirements: leakage-safe prediction, interpretable model, actionable tiers, deterministic funnel, transition logic, and behavioral insights.",
            "My closing line: the project turns raw event telemetry into a practical growth system: score users, explain the score, place them in a funnel stage, and recommend the next action.",
        ),
    ]

    with PdfPages(OUTPUT) as pdf:
        cover(pdf)
        overview_page(pdf)
        for i, args in enumerate(segments, start=1):
            segment_page(pdf, i, *args)
        appendix_page(pdf)
    print(OUTPUT)


if __name__ == "__main__":
    main()
