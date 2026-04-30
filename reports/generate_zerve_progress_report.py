from pathlib import Path
from textwrap import wrap


OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)
PDF_PATH = OUT_DIR / "zerve_dataset_progress_report.pdf"

TOTAL_ROWS = 3_509_628
TOTAL_COLS = 83


def pct_missing(count):
    return round((count / TOTAL_ROWS) * 100, 2)


grouped_nulls = [
    (
        3_477_639,
        8,
        "Previous pageview engagement",
        [
            "properties.$prev_pageview_last_content",
            "properties.$prev_pageview_last_content_percentage",
            "properties.$prev_pageview_last_scroll",
            "properties.$prev_pageview_last_scroll_percentage",
            "properties.$prev_pageview_max_content",
            "properties.$prev_pageview_max_content_percentage",
            "properties.$prev_pageview_max_scroll",
            "properties.$prev_pageview_max_scroll_percentage",
        ],
        "These fields likely come from the same pageview tracking subsystem. They measure prior page scroll/content exposure and are only populated for a small subset of pageview-like events.",
    ),
    (
        2_535_841,
        6,
        "Browser/device metadata",
        [
            "properties.$browser",
            "properties.$browser_language",
            "properties.$device_type",
            "properties.$screen_height",
            "properties.$screen_width",
            "properties.$lib_rate_limit_remaining_tokens",
        ],
        "These fields appear together because they are client-side tracking metadata. They can support segmentation but are weaker than product behavior for activation modeling.",
    ),
    (
        2_959_810,
        5,
        "AI usage metadata",
        [
            "properties.$ai_input_tokens",
            "properties.$ai_latency",
            "properties.$ai_model",
            "properties.$ai_output_tokens",
            "properties.$ai_provider",
        ],
        "These columns likely identify AI interaction events. Missingness is meaningful: absence probably means the row is not an AI event rather than bad data.",
    ),
    (
        0,
        3,
        "Core event identifiers",
        ["person_id", "timestamp", "event"],
        "These are complete and should form the base grain of analysis: who performed what event and when.",
    ),
    (
        1_986_064,
        3,
        "Geo enrichment",
        [
            "properties.$geoip_continent_name",
            "properties.$geoip_country_name",
            "properties.$geoip_time_zone",
        ],
        "These are inferred location fields. They appear together and can be used for regional segmentation, with an Unknown category for missing values.",
    ),
    (
        3_507_488,
        2,
        "Credit balance snapshot",
        ["properties.credits_remaining", "properties.total_credits"],
        "These are highly sparse billing/credit fields. They may be powerful but are leakage-sensitive if predicting monetization or credit usage.",
    ),
    (
        3_296_653,
        2,
        "AI tool usage",
        ["properties.$ai_tool_call_count", "properties.$ai_tools_called"],
        "These likely represent agent/tool behavior inside AI workflows. Presence can become a useful product-engagement flag.",
    ),
    (
        3_509_624,
        2,
        "Feature/subscription signals",
        ["properties.feature_tag", "properties.subscription_type"],
        "These are almost entirely missing and likely tied to rare subscription or feature-gating events. Treat carefully because they may encode downstream outcomes.",
    ),
]


class SimplePDF:
    def __init__(self):
        self.pages = []
        self.lines = []
        self.y = 760
        self.left = 54
        self.width_chars = 96

    @staticmethod
    def esc(text):
        return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def new_page(self, title=None):
        if self.lines:
            self.pages.append("\n".join(self.lines))
        self.lines = []
        self.y = 760
        if title:
            self.text(title, 20, bold=True)
            self.rule()
            self.gap(14)

    def text(self, text, size=10, x=None, bold=False):
        font = "F2" if bold else "F1"
        x = self.left if x is None else x
        self.lines.append(f"BT /{font} {size} Tf {x} {self.y} Td ({self.esc(text)}) Tj ET")
        self.y -= size + 5

    def paragraph(self, text, size=10, indent=0, width=None):
        width = width or self.width_chars
        for line in wrap(text, width=width):
            self.text(line, size=size, x=self.left + indent)
        self.gap(5)

    def bullets(self, items, size=10):
        for item in items:
            wrapped = wrap(item, width=92)
            self.text("- " + wrapped[0], size=size, x=self.left + 14)
            for line in wrapped[1:]:
                self.text("  " + line, size=size, x=self.left + 14)
            self.gap(2)
        self.gap(4)

    def section(self, title):
        if self.y < 110:
            self.new_page()
        self.gap(6)
        self.text(title, 13, bold=True)

    def gap(self, amount):
        self.y -= amount

    def rule(self):
        y = self.y + 4
        self.lines.append(f"0.15 w 54 {y} m 558 {y} l S")

    def table(self, headers, rows, widths):
        if self.y < 160:
            self.new_page()
        self.text(" | ".join(headers), size=9, bold=True)
        self.rule()
        self.gap(5)
        for row in rows:
            chunks = []
            for value, width in zip(row, widths):
                chunks.append(str(value)[:width].ljust(width))
            self.text(" | ".join(chunks), size=8)
        self.gap(8)

    def save(self, path):
        if self.lines:
            self.pages.append("\n".join(self.lines))

        objects = []
        objects.append("<< /Type /Catalog /Pages 2 0 R >>")
        kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(self.pages)))
        objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(self.pages)} >>")

        for i, content in enumerate(self.pages):
            page_obj = 3 + i * 2
            content_obj = page_obj + 1
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> "
                f"/F2 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> >> >> "
                f"/Contents {content_obj} 0 R >>"
            )
            stream = content.encode("latin-1", "replace")
            objects.append(f"<< /Length {len(stream)} >>\nstream\n{content}\nendstream")

        pdf = ["%PDF-1.4\n"]
        offsets = [0]
        for idx, obj in enumerate(objects, start=1):
            offsets.append(sum(len(part.encode("latin-1", "replace")) for part in pdf))
            pdf.append(f"{idx} 0 obj\n{obj}\nendobj\n")

        xref_offset = sum(len(part.encode("latin-1", "replace")) for part in pdf)
        pdf.append(f"xref\n0 {len(objects) + 1}\n")
        pdf.append("0000000000 65535 f \n")
        for off in offsets[1:]:
            pdf.append(f"{off:010d} 00000 n \n")
        pdf.append(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        )
        path.write_bytes("".join(pdf).encode("latin-1", "replace"))


pdf = SimplePDF()
pdf.new_page("Zerve ODSC Datathon: Dataset Progress Report")
pdf.paragraph(
    "This report summarizes the data understanding work completed so far on the Zerve events dataset. "
    "It focuses on schema discovery, missing-value structure, grouped-null interpretation, and implications "
    "for leakage-safe funnel and modeling work."
)
pdf.section("Dataset Snapshot")
pdf.bullets(
    [
        f"Original shape observed: {TOTAL_ROWS:,} rows x {TOTAL_COLS} columns.",
        "Each row appears to represent one product analytics event tied to a person, timestamp, and event name.",
        "The schema includes person properties, browser/device metadata, geolocation fields, AI usage metadata, product activity fields, billing/credit fields, marketing attribution, page engagement fields, and web-vitals metrics.",
        "A dtype warning appeared during CSV load because several property columns contain mixed types. This is expected in event exports, where sparse properties can vary by event type.",
    ]
)
pdf.section("Current Interpretation")
pdf.paragraph(
    "The dataset should be treated as an event log, not as a flat user table. Many null values are likely structural: "
    "a property is missing because it does not apply to that event. Missingness itself can become a useful signal, "
    "especially after aggregating events to the person or workspace level."
)

pdf.new_page("Schema Understanding")
pdf.section("Major Column Families Found")
pdf.bullets(
    [
        "Core identifiers: person_id, timestamp, event. These have no missing values and define the event grain.",
        "Person properties: cloud provider, purpose, role, source, and work type. These describe user profile or onboarding context.",
        "Browser/device/geography: browser, language, OS, device type, screen size, country, continent, and time zone.",
        "AI interaction properties: input tokens, output tokens, latency, model, provider, tool call count, and tools called.",
        "Product activity properties: canvas_id, workspace_id, block types, file type/extension, requirements status, tool name, button/link names, and object counts.",
        "Credit/subscription properties: credits used/received/remaining, total credits, add-on credits, amount, subscription type, and seats.",
        "Marketing attribution: UTM source, medium, campaign, content, and term.",
        "Page engagement/performance: previous pageview scroll/content metrics and web-vitals values such as CLS, FCP, INP, and LCP.",
    ]
)
pdf.section("Important Data Understanding Point")
pdf.paragraph(
    "The same event row will not contain every property. For example, an AI event may contain model/token/latency fields, "
    "while a pageview may contain scroll fields, and a billing event may contain credit fields. This makes grouped missingness "
    "a useful way to infer subsystems and event-specific feature families."
)

pdf.new_page("Missing-Value Analysis")
pdf.section("Method Used")
pdf.paragraph(
    "A custom null-summary dataframe was created by iterating through df.columns, counting nulls per column, and transposing "
    "the result so original columns appeared as rows. The analysis then grouped columns by identical missing-value counts."
)
pdf.section("Key Output: Repeated Missing Counts")
table_rows = [
    [f"{count:,}", f"{pct_missing(count)}%", num_cols, group]
    for count, num_cols, group, _columns, _inference in grouped_nulls
]
pdf.table(["Null Count", "Missing %", "# Cols", "Likely Group"], table_rows, [14, 10, 6, 38])

pdf.new_page("Grouped Missingness Inferences")
for count, num_cols, group, columns, inference in grouped_nulls[:4]:
    pdf.section(group)
    pdf.paragraph(f"Columns: {', '.join(columns)}", size=8, width=110)
    pdf.paragraph(f"Inference: {inference}", size=10)

pdf.new_page("Grouped Missingness Inferences Continued")
for count, num_cols, group, columns, inference in grouped_nulls[4:]:
    pdf.section(group)
    pdf.paragraph(f"Columns: {', '.join(columns)}", size=8, width=110)
    pdf.paragraph(f"Inference: {inference}", size=10)

pdf.new_page("Implications for Feature Engineering")
pdf.section("What This Means")
pdf.bullets(
    [
        "Do not blindly drop high-null columns. In an event log, sparse fields often identify meaningful event types.",
        "Create presence flags for important sparse groups, such as has_ai_usage, has_ai_tool_call, has_credit_snapshot, has_pageview_engagement, and has_subscription_signal.",
        "Aggregate row-level events to person_id or workspace_id before modeling. Useful features include total events, unique event count, active days, first event, last event, AI event count, total tokens, average latency, canvas/file/block activity, and credit-related activity.",
        "Use grouped missingness to build feature families. Columns with identical null counts likely originate from the same product subsystem or event instrumentation path.",
        "Handle categorical missing values as Unknown or Not Applicable, depending on whether missingness means unavailable user metadata or not relevant for the event.",
    ]
)
pdf.section("Useful Presence-Flag Examples")
pdf.bullets(
    [
        "has_ai_usage = properties.$ai_model is not null",
        "has_ai_tool_usage = properties.$ai_tool_call_count is not null",
        "has_credit_snapshot = properties.credits_remaining is not null",
        "has_pageview_scroll_data = properties.$prev_pageview_max_scroll is not null",
        "has_subscription_signal = properties.subscription_type is not null",
    ]
)

pdf.new_page("Leakage and Modeling Plan")
pdf.section("Leakage Considerations")
pdf.paragraph(
    "Several strong-looking fields may be downstream of the thing we eventually want to predict. For example, credit balance, "
    "subscription type, total credits, or feature gating fields should not be used to predict monetization if they are only known "
    "after monetization happens."
)
pdf.section("Recommended Prediction Setup")
pdf.bullets(
    [
        "Use a time-windowed setup: first 24 hours or first N events as features, then predict later activation or engagement.",
        "Define activation behaviorally, for example: a user creates or uses a canvas, interacts with AI, uses tools/blocks, uploads files, or returns across sessions.",
        "Keep the model explainable. Logistic regression, random forest, or gradient boosting can work, but feature importance and actionable interpretation matter more than raw score.",
        "Evaluate with ROC-AUC, precision/recall, F1, and confusion matrix, but connect the results back to product decisions.",
    ]
)
pdf.section("Next Analytical Step")
pdf.paragraph(
    "For each grouped-null family, check which event values populate the group. This will validate whether the inferred groups map "
    "to pageview events, AI events, credit events, or subscription/feature events. That validation directly supports the rubric "
    "categories for data understanding, funnel design, transition logic, and leakage handling."
)

pdf.save(PDF_PATH)
print(PDF_PATH.resolve())
