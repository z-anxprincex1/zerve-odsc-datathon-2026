# Zerve Data Dictionary Notes

Source: `datasets/data_dictionary.csv`

## Dataset Grain

- `person_id`: pseudonymous UUID v5 per user. It is not a raw email.
- `timestamp`: event occurrence timestamp in UTC. Parse with `pd.to_datetime()`.
- `event`: event type/name. Sample distribution includes `credits_used`, `$ai_generation`, `$exception`, `addon_credits_used`, and `notebook_deployment_usage_tracked`.
- The dataset is best interpreted as an event log, not a flat user table.

## Category Counts

- Custom Event Properties: 39 columns
- Other Properties: 17 columns
- Event Device & Browser Context: 7 columns
- AI/ML Properties: 7 columns
- Person Custom Properties: 5 columns
- Event Geography: 3 columns
- Event Screen & Viewport: 2 columns
- Core Identifiers: 1 column
- Timestamps: 1 column
- Event Type: 1 column

## Important Definitions

- `properties.$ai_latency`: AI response latency in seconds, not milliseconds. Median is documented as 3.52s and p95 as 34.9s. Only for `$ai_generation` events.
- `properties.$ai_input_tokens`: number of tokens in the AI model input/prompt.
- `properties.$ai_output_tokens`: number of tokens in the AI model output/completion.
- `properties.$ai_model`: AI model identifier.
- `properties.$ai_provider`: AI provider.
- `properties.$ai_tool_call_count` and `properties.$ai_tools_called`: AI tool/agent behavior fields.
- `properties.credits_remaining`: only populated when balance reaches zero. All non-null values are `0.0`; null means balance is positive or not applicable. This is leakage-sensitive.
- `properties.amount`: transaction amount in USD. In this dataset, documented as always `$25.00`.
- Browser/device columns describe client context.
- GeoIP columns describe inferred geography from IP address.
- Previous pageview columns describe page scroll/content exposure on the previous pageview.

## Modeling / Leakage Notes

- Credit and subscription fields can easily leak downstream monetization outcomes.
- AI fields are only populated on AI-related events, so missingness is a meaningful behavioral signal.
- High-null product-event fields should usually become presence flags or user-level aggregates, not be dropped immediately.
- Use early time windows, such as first 24 hours or first N events, before predicting later activation or engagement.

## Feature Engineering Ideas

- `has_ai_generation`: `properties.$ai_model.notna()`
- `has_ai_tool_usage`: `properties.$ai_tool_call_count.notna()`
- `hit_zero_credit_balance`: `properties.credits_remaining.notna()`
- `has_pageview_engagement_metrics`: `properties.$prev_pageview_max_scroll.notna()`
- `has_device_context`: `properties.$browser.notna()`
- `has_geo_context`: `properties.$geoip_country_name.notna()`
- User-level aggregates: event count, unique event count, active days, first event, last event, AI event count, total input/output tokens, average AI latency, tool usage count, credit usage count.
