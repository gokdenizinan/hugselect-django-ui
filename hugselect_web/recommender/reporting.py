from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPORT_TITLE = "HugSelect Analysis Report"
PRIORITY_LABELS = {
    "must": "Must Have",
    "should": "Should Have",
    "could": "Could Have",
    "wont": "Won't Have",
}


def _display_value(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value if item not in (None, "")]
        return ", ".join(items) or None
    return str(value)


def _feature_label(feature_key):
    if not feature_key:
        return "Requirement"
    return str(feature_key).replace("_", " ").strip().title()


def build_analysis_report_data(
    *,
    search_state,
    comparison_state=None,
    decision_stress_state=None,
    generated_at=None,
):
    """Normalize saved analysis state without recalculating any result."""
    search_state = search_state or {}
    comparison_state = comparison_state or {}
    decision_stress_state = decision_stress_state or {}
    generated_at = generated_at or datetime.now().astimezone()

    explicit_by_priority = {
        priority: []
        for priority in PRIORITY_LABELS
    }
    for requirement in search_state.get("explicit_requirements", []) or []:
        priority = requirement.get("priority")
        if priority not in explicit_by_priority:
            continue
        explicit_by_priority[priority].append({
            "feature": _feature_label(requirement.get("feature_key")),
            "value": _display_value(requirement.get("value")),
        })

    recommended_models = []
    for rank, model in enumerate(search_state.get("results", []) or [], start=1):
        recommended_models.append({
            "rank": rank,
            "model_id": _display_value(model.get("model_id")),
            "author": _display_value(model.get("author")),
            "task": _display_value(model.get("pipeline_tag")),
            "license": _display_value(model.get("license")),
            "library": _display_value(model.get("library_name")),
            "base_models": _display_value(model.get("basemodels")),
            "score": model.get("score"),
            "availability": _display_value(
                (model.get("availability") or {}).get("status")
            ),
            "url": _display_value(model.get("url")),
        })

    return {
        "generated_at": generated_at,
        "overview": {
            "query": _display_value(search_state.get("query")),
            "search_mode": _display_value(search_state.get("search_mode")),
            "search_scope": _display_value(
                search_state.get("search_scope") or "all"
            ),
            "base_model_family": _display_value(
                search_state.get("base_model_family")
            ),
        },
        "explicit_requirements": explicit_by_priority,
        "recommended_models": recommended_models,
        "comparison": comparison_state,
        "decision_stress": decision_stress_state,
    }


def _styles():
    sample = getSampleStyleSheet()
    ink = colors.HexColor("#171914")
    muted = colors.HexColor("#5E635A")
    blue = colors.HexColor("#1747D1")

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=ink,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=muted,
            spaceAfter=7 * mm,
        ),
        "section": ParagraphStyle(
            "ReportSection",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=ink,
            spaceBefore=6 * mm,
            spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "ReportSubsection",
            parent=sample["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=blue,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=ink,
            splitLongWords=True,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=muted,
            splitLongWords=True,
            wordWrap="CJK",
        ),
        "label": ParagraphStyle(
            "ReportLabel",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=10,
            textColor=muted,
        ),
        "table_header": ParagraphStyle(
            "ReportTableHeader",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=10,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
    }


def _paragraph(value, style):
    text = _display_value(value)
    if text is None:
        text = "Not available"
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def _metadata_table(rows, styles, widths=(42 * mm, 128 * mm)):
    data = [
        [
            _paragraph(label, styles["label"]),
            _paragraph(value, styles["body"]),
        ]
        for label, value in rows
        if value not in (None, "")
    ]
    if not data:
        return Paragraph("No additional metadata is available.", styles["small"])
    table = Table(data, colWidths=list(widths), hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ECEDE8")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#AEB2A8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5D6CD")),
    ]))
    return table


def _section_heading(story, title, styles):
    story.append(Paragraph(escape(title), styles["section"]))
    rule = HRFlowable(
        width="100%",
        thickness=0.7,
        color=colors.HexColor("#171914"),
        spaceAfter=3 * mm,
    )
    rule.keepWithNext = True
    story.append(rule)


def _score_text(score, search_mode):
    if score is None:
        return None
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return str(score)
    if search_mode == "feature-based":
        return f"{numeric_score:.1f}% feature match"
    return f"{numeric_score:.3f} relevance"


def _add_recommended_models(story, report_data, styles):
    models = report_data["recommended_models"]
    if not models:
        return
    _section_heading(story, "Recommended models", styles)
    search_mode = report_data["overview"].get("search_mode")

    for model in models:
        heading = f"{model['rank']:02d}  {model.get('model_id') or 'Unknown model'}"
        story.append(Paragraph(escape(heading), styles["subsection"]))
        story.append(_metadata_table([
            ("Author", model.get("author")),
            ("Task", model.get("task")),
            ("License", model.get("license")),
            ("Library", model.get("library")),
            ("Base model(s)", model.get("base_models")),
            ("Score", _score_text(model.get("score"), search_mode)),
            ("Availability", model.get("availability")),
            ("Hugging Face URL", model.get("url")),
        ], styles))
        story.append(Spacer(1, 3 * mm))


def _add_comparison(story, comparison, styles):
    models = comparison.get("models", []) or []
    if not models:
        return
    _section_heading(story, "Active comparison", styles)
    search_mode = comparison.get("search_mode")
    score_label = (
        "Feature match"
        if search_mode == "feature-based"
        else "Relevance"
    )

    for model in models:
        story.append(Paragraph(
            escape(str(model.get("model_id") or "Unknown model")),
            styles["subsection"],
        ))
        story.append(_metadata_table([
            ("Author", model.get("author")),
            ("Task", model.get("pipeline_tag")),
            ("License", model.get("license")),
            ("Library", model.get("library_name")),
            ("Base model(s)", model.get("basemodels")),
            (score_label, _score_text(
                model.get("comparison_score"),
                search_mode,
            )),
        ], styles))
        story.append(Spacer(1, 2 * mm))

    coverage_rows = comparison.get("coverage_rows", []) or []
    if coverage_rows:
        story.append(Paragraph("Requirement coverage", styles["subsection"]))
        model_ids = [str(model.get("model_id") or "") for model in models]
        header = [
            _paragraph("Requirement", styles["table_header"]),
            _paragraph("Value", styles["table_header"]),
            *[
                _paragraph(model_id, styles["table_header"])
                for model_id in model_ids
            ],
        ]
        rows = [header]
        for coverage in coverage_rows:
            status_by_model = {
                status.get("model_id"): status.get("matched")
                for status in coverage.get("model_statuses", []) or []
            }
            rows.append([
                _paragraph(
                    _feature_label(coverage.get("feature_key")),
                    styles["body"],
                ),
                _paragraph(coverage.get("user_value"), styles["body"]),
                *[
                    _paragraph(
                        "Matched" if status_by_model.get(model_id) else "Not matched",
                        styles["small"],
                    )
                    for model_id in model_ids
                ],
            ])
        available_width = 170 * mm
        first_widths = [37 * mm, 35 * mm]
        status_width = (available_width - sum(first_widths)) / max(1, len(model_ids))
        table = Table(
            rows,
            colWidths=first_widths + [status_width] * len(model_ids),
            repeatRows=1,
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#30342D")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AEB2A8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)

    summary = comparison.get("decision_summary") or {}
    if summary:
        leaders = _display_value(summary.get("leaders"))
        recommendation = summary.get("tie_breaker_model") or leaders
        story.append(Spacer(1, 3 * mm))
        story.append(_metadata_table([
            ("Recommendation", recommendation),
            ("Best feature-match score", _score_text(
                summary.get("best_score"),
                "feature-based",
            )),
            ("Tie", "Yes" if summary.get("is_tie") else "No"),
            ("Tie-break result", summary.get("tie_breaker_model")),
        ], styles))


def _add_decision_stress(story, state, styles):
    summary = state.get("stress_summary") or {}
    result = state.get("stress_result") or {}
    scenarios = result.get("scenarios", []) or []
    if not summary and not scenarios:
        return

    _section_heading(story, "Decision Stress analysis", styles)
    story.append(_metadata_table([
        ("Stability", summary.get("stability_label")),
        ("Consistency", (
            f"{summary.get('stability_percentage'):.1f}%"
            if isinstance(summary.get("stability_percentage"), (int, float))
            else None
        )),
        ("Most consistent winner(s)", _display_value(summary.get("leaders"))),
        ("Tested scenarios", summary.get("scenario_count")),
    ], styles))

    if scenarios:
        story.append(Paragraph("Scenario outcomes", styles["subsection"]))
        rows = [[
            _paragraph("Scenario", styles["table_header"]),
            _paragraph("Winner(s)", styles["table_header"]),
            _paragraph("Important exclusions", styles["table_header"]),
        ]]
        for scenario in scenarios:
            exclusions = []
            for model_result in scenario.get("model_results", []) or []:
                if not model_result.get("strict_exclusion"):
                    continue
                missed = [
                    f"{_feature_label(item.get('feature_key'))}: "
                    f"{_display_value(item.get('user_value')) or 'unknown'}"
                    for item in model_result.get("missed_essentials", []) or []
                ]
                detail = "; ".join(missed) or "strict requirement not met"
                exclusions.append(
                    f"{model_result.get('model_id')}: {detail}"
                )
            winners = (
                "No eligible model"
                if scenario.get("all_models_excluded")
                else _display_value(scenario.get("winners"))
            )
            rows.append([
                _paragraph(scenario.get("label"), styles["body"]),
                _paragraph(winners, styles["body"]),
                _paragraph(_display_value(exclusions), styles["small"]),
            ])
        table = Table(
            rows,
            colWidths=[50 * mm, 48 * mm, 72 * mm],
            repeatRows=1,
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#30342D")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AEB2A8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)


def generate_analysis_report(
    *,
    search_state,
    comparison_state=None,
    decision_stress_state=None,
    generated_at=None,
):
    """Generate a PDF from saved session analysis and return its bytes."""
    report_data = build_analysis_report_data(
        search_state=search_state,
        comparison_state=comparison_state,
        decision_stress_state=decision_stress_state,
        generated_at=generated_at,
    )
    styles = _styles()
    output = BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        title=REPORT_TITLE,
        author="HugSelect",
        subject="Search, comparison, and decision robustness analysis",
        pageCompression=1,
    )
    story = [
        Paragraph(REPORT_TITLE, styles["title"]),
        Paragraph(
            "Evidence-led Hugging Face model recommendation and decision support",
            styles["subtitle"],
        ),
    ]

    generated_at = report_data["generated_at"]
    generated_text = generated_at.strftime("%d %B %Y, %H:%M %Z")
    overview = report_data["overview"]
    scope = overview.get("search_scope")
    if scope == "family" and overview.get("base_model_family"):
        scope = f"Base-model family: {overview['base_model_family'].title()}"
    elif scope:
        scope = "All models" if scope == "all" else scope

    _section_heading(story, "Analysis overview", styles)
    story.append(_metadata_table([
        ("Generated", generated_text),
        ("Original query", overview.get("query")),
        ("Search mode", overview.get("search_mode")),
        ("Search scope", scope),
        ("Selected family", (
            overview.get("base_model_family").title()
            if overview.get("base_model_family")
            else None
        )),
    ], styles))

    explicit_requirements = report_data["explicit_requirements"]
    if any(explicit_requirements.values()):
        _section_heading(story, "Explicit MoSCoW requirements", styles)
        for priority, label in PRIORITY_LABELS.items():
            requirements = explicit_requirements.get(priority, [])
            if not requirements:
                continue
            story.append(Paragraph(label, styles["subsection"]))
            for requirement in requirements:
                text = (
                    f"<b>{escape(requirement['feature'])}:</b> "
                    f"{escape(requirement.get('value') or 'Not available')}"
                )
                story.append(Paragraph(text, styles["body"]))
            story.append(Spacer(1, 1.5 * mm))

    _add_recommended_models(story, report_data, styles)
    _add_comparison(story, report_data["comparison"], styles)
    _add_decision_stress(story, report_data["decision_stress"], styles)

    def draw_page(canvas, document):
        canvas.saveState()
        width, _ = A4
        canvas.setStrokeColor(colors.HexColor("#AEB2A8"))
        canvas.setLineWidth(0.4)
        canvas.line(20 * mm, 14 * mm, width - 20 * mm, 14 * mm)
        canvas.setFillColor(colors.HexColor("#5E635A"))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(20 * mm, 9 * mm, "HUGSELECT / ANALYSIS REPORT")
        canvas.drawRightString(
            width - 20 * mm,
            9 * mm,
            f"Page {document.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return output.getvalue()
