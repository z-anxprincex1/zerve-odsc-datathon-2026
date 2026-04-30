# Zerve ODSC Datathon — Upgrade Prediction & Funnel Analysis

A 24-hour datathon analysing 3.5 million product events from the Zerve AI platform to predict user upgrades and model the user journey.

---

## Challenges

### Challenge 1 — Predict Upgrades

Build a machine-learning model that estimates the probability a free-tier user will upgrade to a paid plan, using only behaviour observed **before** the upgrade event.

**Approach:**
- Defined a per-user snapshot time (first upgrade timestamp for upgraders, global max for non-upgraders)
- Filtered all features to strictly pre-snapshot events — no leakage
- Engineered 43 behavioural features across activity, AI usage, deployments, credit pressure, and time windows
- Trained XGBoost with 5-fold stratified cross-validation (`scale_pos_weight` for class imbalance)
- Explained predictions with SHAP values

**Results:**
- Base upgrade rate: **1.84%** (227 upgraders out of ~12,300 users)
- Top predictors: `ai_gen_count`, `total_events`, `credit_pressure_count`, `unique_event_types`, `tenure_days`
- Upgraders generate **11.7× more events** than non-upgraders (median 199 vs 17)
- Median upgrader converts after just **1 day** of tenure — value must land fast

---

### Challenge 2 — Deterministic User Funnel

Classify every user into exactly one ordered stage at any point in time, based purely on observed behaviour.

**Funnel stages (in order):**

| Stage | Definition |
|---|---|
| New | Signed up, no meaningful product action |
| Exploring | Has pageviews, clicks, or navigation events |
| Created Content | Has created a canvas, block, file, or workspace object |
| AI Engaged | Has triggered at least one AI generation |
| Workflow Builder | Uses tools, integrations, code blocks, or multi-step workflows |
| Credit Active | Has consumed credits or add-on credits |
| Consistently Engaged | Active across multiple days past a threshold |
| Upgraded | Has a `subscription_upgraded` event |
| At Risk | Previously engaged but no meaningful activity in the recent window |

Each user is assigned their **highest achieved stage** as of any given timestamp, making the funnel fully deterministic and time-aware.

---

## Why is the Conversion Rate Only 1.84%?

At first glance, 227 upgraders out of ~12,300 users looks low. It isn't anomalous — here's why.

**1. This is normal for PLG SaaS**
Free-to-paid conversion rates in product-led growth models typically sit between 2–5% for mature products and lower for early-stage platforms. Zerve is competing in the AI data tooling space where users have many free alternatives, so a sub-2% rate is expected, not alarming.

**2. The free tier is genuinely useful**
Users get free credits and can run AI generations, build canvases, and deploy notebooks without paying. A generous free tier by design depresses conversion — users only need to upgrade when they hit limits. This is intentional PLG strategy: land broadly, convert the power users.

**3. Credit pressure is the actual trigger — and it's rare**
The `credit_pressure_count` feature (credit limit events) is a top upgrade predictor. Most casual users never exhaust their free credits, so the upgrade trigger never fires. Only heavy AI users burn through them quickly.

**4. Upgraders convert in the first 1–3 days**
The median upgrader's tenure is 1 day. This means the upgrade decision is made very early — almost immediately after experiencing the core value. Users who don't convert quickly tend to churn or stay indefinitely on free tier. There is no slow-burn conversion path evident in the data.

**5. Many users are explorers, not buyers**
The `person_properties.purpose` field includes Education and Personal Projects as common values. A meaningful share of signups are students or hobbyists who have no intent to pay — they inflate the denominator without ever being real conversion candidates.

**6. The dataset window may not capture delayed conversions**
Users who signed up near the end of the observation period haven't had time to convert yet. The true eventual conversion rate is likely higher than the snapshot shows.

**The takeaway:** the 1.84% figure reflects a healthy free tier doing its job. The real question — which the model answers — is *which* of the remaining 98% are most worth activating.

---

## Key Behavioural Insights

**Insight 1 — AI Power Users Drive Upgrades**
Users who ran 10+ AI generations were dramatically more likely to upgrade. AI usage is the single strongest pre-upgrade signal in the dataset.

**Insight 2 — Deployment Intent is a Strong Secondary Signal**
A disproportionate share of upgraders had deployment-related events (`notebook_deployed`, `app_deployed`) before converting — even though only a minority of users deploy at all.

**Insight 3 — Early Engagement Predicts Long-Term Conversion**
Users who triggered their first AI generation within 3 days of signup converted at significantly higher rates. The activation window is narrow.

---

## Bonus — What-If Upgrade Probability Simulator

An interactive Streamlit app (`main.py`) that lets you adjust user behaviour sliders and see the model's upgrade probability in real time. Built for demo and product intuition.

```bash
conda activate zerve-odsc
streamlit run main.py
```

Persona presets included: Casual, Explorer, Heavy AI User (upgrader profile), Power AI User, At Risk.

---

## Repository Structure

```
.
├── t1.ipynb                        # Main analysis notebook
├── main.py                         # Streamlit What-If Simulator
├── datasets/
│   ├── zerve_events.csv            # Raw event log (3.5M rows, 83 columns)
│   └── data_dictionary.csv         # Column descriptions, types, null rates
└── reports/
    ├── xgb_model.pkl               # Trained XGBoost model
    ├── feature_cols.pkl            # Feature column list (43 features)
    └── event_type_upgrade_report.csv
```

---

## Setup

```bash
conda activate zerve-odsc
pip install streamlit   # if not already installed
```

Run the notebook end-to-end to regenerate the model, then launch the app.

---

## Dataset

| Property | Value |
|---|---|
| Rows | 3,509,628 |
| Columns | 83 |
| Grain | One row per product event |
| Core fields | `person_id`, `timestamp`, `event` |
| Target event | `subscription_upgraded` |
| Upgrade rate | 1.84% of users |

Key data notes:
- `properties.$ai_latency` is in **seconds**, not milliseconds
- `properties.credits_remaining` is only populated when the balance hits zero; all non-null values are `0.0`
- Many null columns are structural — a field is missing because it does not apply to that event type, not because data is absent
