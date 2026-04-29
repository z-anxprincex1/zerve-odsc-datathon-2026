# Zerve ODSC Datathon: Upgrade Prediction and Funnel Analysis

This repository contains analysis for the Zerve ODSC datathon event dataset.

The challenge has two connected goals:

1. Build a model that predicts whether a user will upgrade.
2. Define a deterministic, time-aware user funnel where every user can be classified at any point in time.

The upgrade event is:

```python
df["event"] == "subscription_upgraded"
```

The most important constraint is leakage prevention: upgrade likelihood must be estimated using only information that was known before the user upgraded.

## Project Contents

- `t1.ipynb`: Main analysis notebook.
- `datasets/zerve_events.csv`: Raw Zerve event dataset.
- `datasets/data_dictionary.csv`: Feature dictionary with column descriptions, data types, null percentages, and sample values.
- `reports/`: Local generated notes/reports. This directory is ignored by Git.

## Dataset Overview

The primary dataset has:

- `3,509,628` rows
- `83` columns
- Event-level grain: each row represents a product analytics event.

The three complete core fields are:

- `person_id`
- `timestamp`
- `event`

These define who performed an action, when it happened, and what event occurred.

## Work Completed So Far

Current exploration includes:

- Loaded the Zerve events CSV into pandas.
- Checked dataset shape and column names.
- Created a custom null-summary dataframe with columns shown as rows.
- Grouped columns by identical missing-value counts.
- Added column-name lists to grouped missingness output.
- Used the data dictionary to better interpret feature groups and avoid relying only on column-name guesses.
- Identified `subscription_upgraded` as the challenge target event.
- Reframed the modeling plan around time-aware upgrade prediction.

## Early Data Understanding

The dataset should be treated as an event log rather than a flat user table.

Many null values appear to be structural, meaning a field is missing because it does not apply to that event type. For example:

- AI metadata appears on AI-generation events.
- Pageview scroll fields appear on pageview-related events.
- Credit fields appear on credit/billing-related events.
- Browser, device, and geography fields come from event context enrichment.

This means high-null columns should not be automatically dropped. Their presence or absence can become useful behavioral signals.

## Important Data Dictionary Notes

- `properties.$ai_latency` is measured in seconds, not milliseconds.
- AI fields such as token counts, model, provider, and latency are only populated for `$ai_generation` events.
- `properties.credits_remaining` is only populated when the balance reaches zero; all non-null values are `0.0`.
- `properties.amount` is documented as always `$25.00` in this dataset.
- Credit and subscription-related columns are potentially leakage-sensitive for monetization or retention prediction.

## Challenge 1: Predict Upgrades

The modeling goal is to estimate the probability that a user will upgrade in the future.

The prediction setup should be time-aware:

- Observation window: behavior known before the prediction point.
- Prediction window: whether the user upgrades later.
- Target event: `subscription_upgraded`.

Example setup:

```text
Use the first 7 days of user behavior to predict whether the user upgrades after that window.
```

Alternative observation windows to test:

- First 24 hours
- First 3 days
- First 7 days
- First N events

The best setup is not necessarily the one with the highest raw accuracy. It should be valid, realistic, explainable, and useful for product decisions.

## Feature Engineering Direction

Potential feature families:

- User activity: total events, unique events, active days, first event, last event.
- AI engagement: AI event count, total input tokens, total output tokens, average AI latency, AI provider/model usage.
- Product engagement: canvas activity, workspace activity, block/file/tool interactions.
- Credit behavior: credits used, zero-balance events, add-on credit activity.
- Context: browser, device type, OS, country, timezone, UTM fields.
- Presence flags: whether sparse event-specific fields were populated.

Example presence flags:

- `has_ai_generation`
- `has_ai_tool_usage`
- `hit_zero_credit_balance`
- `has_pageview_engagement_metrics`
- `has_device_context`
- `has_geo_context`

Additional upgrade-prediction features:

- `events_before_prediction`
- `unique_events_before_prediction`
- `active_days_before_prediction`
- `time_to_first_ai_event`
- `time_to_first_canvas_or_content_event`
- `ai_events_before_prediction`
- `total_ai_input_tokens_before_prediction`
- `total_ai_output_tokens_before_prediction`
- `avg_ai_latency_before_prediction`
- `tool_usage_events_before_prediction`
- `content_creation_events_before_prediction`
- `credit_usage_events_before_prediction`
- `utm_source`
- `person_properties.role`
- `person_properties.purpose`
- `person_properties.work_type`

## Challenge 2: Build a Funnel

The funnel must classify every user into exactly one stage at any point in time.

Good funnel stages should be:

- Specific and observable
- Based on real user behavior
- Deterministic
- Complete, so no user is uncategorized
- Time-aware, so users can move between stages as events happen

Proposed funnel stages:

1. `New`: User exists but has not performed a meaningful product action.
2. `Exploring`: User has performed non-trivial events such as pageviews, clicks, or navigation.
3. `Created Content`: User has created or interacted with a canvas, file, block, workspace, or similar content object.
4. `AI Engaged`: User has triggered AI generation or has AI token/model/provider fields populated.
5. `Workflow Builder`: User uses tools, code, blocks, requirements, integrations, or file/workspace workflows.
6. `Credit Active`: User uses credits or add-on credits.
7. `Consistently Engaged`: User returns across multiple days or crosses a meaningful engagement threshold.
8. `Upgraded`: User has the `subscription_upgraded` event.
9. `At Risk`: Previously engaged user with no meaningful recent activity after a defined inactivity window.

The final funnel should include exact transition rules, thresholds, and time windows.

## Modeling Plan

A strong next direction is to build a leakage-safe upgrade model:

1. Parse timestamps and sort events by `person_id` and `timestamp`.
2. Identify each user's first `subscription_upgraded` timestamp.
3. Define an observation window before upgrade or before a prediction cutoff.
4. Exclude the `subscription_upgraded` event itself from feature creation.
5. Exclude any rows that occur after a user's first upgrade.
6. Aggregate event-level data to `person_id` or `workspace_id`.
7. Train a baseline explainable model.
8. Evaluate on future data or a time-based split.
9. Interpret feature importance and convert findings into product recommendations.

Baseline model candidates:

- Logistic regression
- Random forest
- Gradient boosting, if available

Evaluation metrics:

- ROC-AUC
- Precision
- Recall
- F1 score
- Confusion matrix

Interpretability outputs:

- Feature importance
- Coefficients for linear models
- Segment-level upgrade rates
- Funnel-stage conversion rates

## Leakage Considerations

Fields related to credits, subscriptions, and downstream usage should be handled carefully.

For example, if predicting whether a user upgrades, avoid using fields that are only known after upgrade or payment activity occurs.

High-risk leakage fields include:

- `properties.subscription_type`
- `properties.amount`
- `properties.total_credits`
- `properties.total_addon_credits`
- `properties.credits_remaining`
- `subscription_upgraded` event rows
- Any events or properties occurring after the user's upgrade timestamp

The preferred approach is a time-based split:

- Feature window: early user behavior before upgrade.
- Prediction window: future upgrade outcome.

## Suggested Analysis Flow

1. Load `zerve_events.csv` and `data_dictionary.csv`.
2. Parse `timestamp` with `pd.to_datetime()`.
3. Inspect event counts and upgrade-event frequency.
4. Calculate number of unique upgrading users.
5. Define first upgrade timestamp per user.
6. Create a leakage-safe feature window.
7. Aggregate event-level data to `person_id` or `workspace_id`.
8. Engineer behavioral features.
9. Define deterministic funnel stages.
10. Train a baseline model.
11. Evaluate on future data.
12. Explain drivers of upgrade likelihood.
13. Recommend product actions.

## Next Steps

- Validate grouped-null feature families by checking which `event` values populate each group.
- Count total `subscription_upgraded` events.
- Count unique users who upgraded.
- Define the first upgrade timestamp per user.
- Build leakage-safe observation windows.
- Build user-level aggregates from event-level rows.
- Define funnel stages and stage-transition rules.
- Create the upgrade target.
- Train a baseline explainable model.
- Convert model outputs into product recommendations.
