# pdf_generator.py (Product A: Genesis Architecture Diagnostic — McKinsey Executive Edition)
import io
import html
from datetime import datetime
from typing import Dict, Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
    KeepTogether,
)


def _esc(text: Any) -> str:
    """Safely escapes text for ReportLab XML Paragraphs without double-escaping."""
    if text is None:
        return ""
    unescaped = html.unescape(str(text))
    return html.escape(unescaped).replace("\n", "<br/>")


def _draw_mckinsey_canvas(canvas, doc, client_ref_id: str, date_str: str):
    """Draws McKinsey-style top rule and bottom executive footer on every page."""
    canvas.saveState()
    page_width, page_height = letter

    # Top thin architectural accent rule
    canvas.setStrokeColor(colors.HexColor("#051C2C"))
    canvas.setLineWidth(2.5)
    canvas.line(40, page_height - 24, page_width - 40, page_height - 24)

    # Bottom footer separator rule
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.setLineWidth(0.6)
    canvas.line(40, 38, page_width - 40, 38)

    # Footer metadata
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(
        40,
        26,
        f"PickModelAI Architecture Advisory  |  Vault Ref: {client_ref_id}  |  "
        f"Estimates reflect regional cloud rates (e.g., eu-central-1 / us-east-1) excl. DevOps labor.",
    )
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(colors.HexColor("#051C2C"))
    canvas.drawRightString(page_width - 40, 26, f"Page {doc.page}")
    canvas.restoreState()


def generate_blueprint_pdf(
    blueprint_data: Dict[str, Any],
    client_ref_id: str = "PMAI-BETA-0001",
) -> bytes:
    """
    Generates a 2-page McKinsey-style executive Platypus PDF for Product A
    (Genesis Architecture Diagnostic). Compatible with the upgraded FreeTierBlueprint schema.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=48,
        title=f"PickModelAI Executive Diagnostic - {client_ref_id}",
    )

    styles = getSampleStyleSheet()

    # McKinsey-inspired Executive Color Palette
    mck_navy = colors.HexColor("#051C2C")       # Deep authoritative midnight navy
    mck_blue = colors.HexColor("#005EB8")       # Executive accent blue
    mck_cyan = colors.HexColor("#00A3E0")       # Secondary highlight cyan
    slate_dark = colors.HexColor("#1E293B")     # Primary body charcoal
    slate_muted = colors.HexColor("#64748B")    # Muted captions & labels
    bg_tint = colors.HexColor("#F8FAFC")        # Subtle alternating table row tint
    bg_header_tint = colors.HexColor("#F1F5F9") # KPI & Callout background
    rule_light = colors.HexColor("#E2E8F0")     # Hairline table dividers
    rule_med = colors.HexColor("#94A3B8")       # Structural sub-borders
    warn_amber = colors.HexColor("#D97706")     # Risk callout accent bar
    warn_bg = colors.HexColor("#FFFBEB")        # Risk callout soft fill

    # Typography Styles
    kicker_style = ParagraphStyle(
        "Kicker",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=mck_cyan,
    )
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15.5,
        leading=19,
        textColor=colors.white,
    )
    hdr_meta_style = ParagraphStyle(
        "HdrMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#E2E8F0"),
        alignment=TA_RIGHT,
    )
    exhibit_label_style = ParagraphStyle(
        "ExhibitLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=mck_blue,
        spaceBefore=8,
        spaceAfter=1,
    )
    section_hdr = ParagraphStyle(
        "SectionHdr",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=mck_navy,
        spaceBefore=0,
        spaceAfter=5,
    )
    lead_summary_style = ParagraphStyle(
        "LeadSummary",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.4,
        leading=14,
        textColor=slate_dark,
        alignment=TA_JUSTIFY,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.8,
        textColor=slate_dark,
        alignment=TA_LEFT,
    )
    th_style = ParagraphStyle(
        "ThStyle",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.white,
    )
    dim_style = ParagraphStyle(
        "DimStyle",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=mck_navy,
    )

    story = []
    date_str = datetime.now().strftime("%B %d, %Y")

    project_title = _esc(blueprint_data.get("project_short_title", "AI Architecture Blueprint"))
    grade_raw = str(blueprint_data.get("feasibility_grade", "Grade A (Fast & Affordable to Build)"))
    grade = _esc(grade_raw)
    modality = _esc(blueprint_data.get("detected_modality", "TEXT_REASONING"))
    summary = _esc(blueprint_data.get("user_idea_summary", ""))
    workflow_steps = blueprint_data.get("workflow_steps", [])
    next_steps = _esc(blueprint_data.get("next_steps_recommendation", ""))

    paid = blueprint_data.get("paid_stack", {})
    oss = blueprint_data.get("open_source_stack", {})
    paid_cost_num = int(paid.get("monthly_cost_number_usd", 0))
    oss_cost_num = int(oss.get("monthly_cost_number_usd", 350))

    # Split Grade Badge & Subtitle for cleaner KPI visual hierarchy
    if "(" in grade_raw and ")" in grade_raw:
        grade_main = _esc(grade_raw.split("(")[0].strip())
        grade_sub = _esc(grade_raw.split("(")[1].replace(")", "").strip())
    else:
        grade_main = grade
        grade_sub = "Verified Architecture"

    # =========================================================================
    # PAGE 1: EXECUTIVE BRIEFING & ARCHITECTURE TRADE-OFF MATRIX
    # =========================================================================
    header_table = Table(
        [
            [
                Paragraph(
                    f"<font size=7.5 color='#38BDF8'><b>PICKMODELAI ARCHITECTURE PRACTICE  |  GENESIS DIAGNOSTIC</b></font><br/>"
                    f"{project_title.upper()}",
                    title_style,
                ),
                Paragraph(
                    f"<font size=7.2 color='#94A3B8'><b>CLIENT VAULT REFERENCE</b></font><br/>"
                    f"<font size=10 color='#FFFFFF'><b>{_esc(client_ref_id)}</b></font><br/>"
                    f"<font size=7.8 color='#CBD5E1'>{date_str}</font>",
                    hdr_meta_style,
                ),
            ]
        ],
        colWidths=[382, 150],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), mck_navy),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 11),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 8))

    # Executive KPI Ribbon (Top & Bottom Heavy Rules, Clean Vertical Dividers)
    kpi_table = Table(
        [
            [
                Paragraph(
                    f"<font size=7 color='#64748B'><b>PRIMARY MODALITY</b></font><br/>"
                    f"<font size=10 color='#051C2C'><b>{modality}</b></font><br/>"
                    f"<font size=7 color='#64748B'>Deterministic Guardrails</font>",
                    body_style,
                ),
                Paragraph(
                    f"<font size=7 color='#64748B'><b>FEASIBILITY INDEX</b></font><br/>"
                    f"<font size=10 color='#005EB8'><b>{grade_main}</b></font><br/>"
                    f"<font size=7 color='#64748B'>{grade_sub}</font>",
                    body_style,
                ),
                Paragraph(
                    f"<font size=7 color='#64748B'><b>OPTION 1: MANAGED API</b></font><br/>"
                    f"<font size=10 color='#051C2C'><b>${paid_cost_num:,} / mo</b></font><br/>"
                    f"<font size=7 color='#0F766E'>Target Prod Run-Rate</font>",
                    body_style,
                ),
                Paragraph(
                    f"<font size=7 color='#64748B'><b>OPTION 2: PRIVATE VPC</b></font><br/>"
                    f"<font size=10 color='#051C2C'><b>${oss_cost_num:,} / mo</b></font><br/>"
                    f"<font size=7 color='#B45309'>Infra Only (Excl. MLOps)</font>",
                    body_style,
                ),
            ]
        ],
        colWidths=[133, 133, 133, 133],
    )
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg_header_tint),
                ("LINEABOVE", (0, 0), (-1, 0), 1.5, mck_navy),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, rule_med),
                ("LINEBEFORE", (1, 0), (-1, -1), 0.5, rule_light),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 6))

    # Section 1: Executive Thesis & Target SLA Scope (Left Accent Bar Callout)
    story.append(Paragraph("EXHIBIT 1  |  STRATEGIC MANDATE & SLA TARGETS", exhibit_label_style))
    story.append(Paragraph("1. Executive Scope, Operational Bottlenecks & 6-Month Value Thesis", section_hdr))
    summary_box = Table([[Paragraph(summary, lead_summary_style)]], colWidths=[532])
    summary_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg_tint),
                ("LINEBEFORE", (0, 0), (0, -1), 3.0, mck_blue),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 11),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(summary_box)
    story.append(Spacer(1, 6))

    # Section 2: End-to-End Deterministic Architecture Pipeline
    story.append(Paragraph("EXHIBIT 2  |  DETERMINISTIC EXECUTION PIPELINE", exhibit_label_style))
    story.append(Paragraph("2. End-to-End System Data Flow & Escalation Routing", section_hdr))

    phase_labels = ["INGESTION & SYNC", "PRIVACY & CONTEXT", "INFERENCE & SCORING", "FALLBACK & HANDOFF"]
    step_cells = []
    for idx, step_text in enumerate(workflow_steps[:4], 1):
        p_label = phase_labels[idx - 1] if idx <= 4 else "EXECUTION"
        step_cells.append(
            Paragraph(
                f"<font size=7 color='#005EB8'><b>STAGE 0{idx} — {p_label}</b></font><br/>"
                f"<font size=8.4 color='#1E293B'>{_esc(step_text)}</font>",
                body_style,
            )
        )
    if step_cells:
        flow_table = Table([step_cells], colWidths=[133] * len(step_cells))
        flow_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), bg_tint),
                    ("LINEABOVE", (0, 0), (-1, 0), 1.5, mck_blue),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.75, rule_med),
                    ("LINEBEFORE", (1, 0), (-1, -1), 0.5, rule_light),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(flow_table)
    story.append(Spacer(1, 6))

    # Section 3: McKinsey Dual-Track Architecture & TCO Comparison Matrix
    story.append(Paragraph("EXHIBIT 3  |  ARCHITECTURAL & TCO TRADE-OFF MATRIX", exhibit_label_style))
    story.append(
        Paragraph(
            "3. Option 1 (Managed Cloud APIs) vs. Option 2 (Self-Hosted Private VPC)",
            section_hdr,
        )
    )

    stack_table = Table(
        [
            [
                Paragraph("ARCHITECTURAL DIMENSION", th_style),
                Paragraph("OPTION 1: MANAGED ENTERPRISE APIs<br/><font size=7.2 color='#BAE6FD'>Recommended for Lean Teams & Fast Time-to-Value</font>", th_style),
                Paragraph("OPTION 2: SELF-HOSTED PRIVATE VPC<br/><font size=7.2 color='#BAE6FD'>Dedicated GPU Compute & Full VPC Isolation</font>", th_style),
            ],
            [
                Paragraph("Model & Inference Stack", dim_style),
                Paragraph(_esc(paid.get("model_stack", "")), body_style),
                Paragraph(_esc(oss.get("model_stack", "")), body_style),
            ],
            [
                Paragraph("Vector Knowledge Store", dim_style),
                Paragraph(_esc(paid.get("vector_db", "")), body_style),
                Paragraph(_esc(oss.get("vector_db", "")), body_style),
            ],
            [
                Paragraph("Strategic & Team Fit", dim_style),
                Paragraph(_esc(paid.get("why_this_stack", "")), body_style),
                Paragraph(_esc(oss.get("why_this_stack", "")), body_style),
            ],
            [
                Paragraph("Monthly TCO & Run-Rate", dim_style),
                Paragraph(_esc(paid.get("estimated_monthly_cost", "")), body_style),
                Paragraph(_esc(oss.get("estimated_monthly_cost", "")), body_style),
            ],
            [
                Paragraph("Unit-Cost & Scaling Lever", dim_style),
                Paragraph(_esc(paid.get("scaling_path", "")), body_style),
                Paragraph(_esc(oss.get("scaling_path", "")), body_style),
            ],
        ],
        colWidths=[116, 208, 208],
    )
    stack_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), mck_navy),
                ("BACKGROUND", (0, 1), (-1, 1), colors.white),
                ("BACKGROUND", (0, 2), (-1, 2), bg_tint),
                ("BACKGROUND", (0, 3), (-1, 3), colors.white),
                ("BACKGROUND", (0, 4), (-1, 4), bg_tint),
                ("BACKGROUND", (0, 5), (-1, 5), colors.white),
                ("LINEABOVE", (0, 0), (-1, 0), 1.5, mck_navy),
                ("LINEBELOW", (0, 0), (-1, 0), 1.2, mck_blue),
                ("LINEBELOW", (0, 1), (-1, -2), 0.5, rule_light),
                ("LINEBELOW", (0, -1), (-1, -1), 1.5, mck_navy),
                ("LINEBEFORE", (1, 0), (-1, -1), 0.5, rule_light),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(stack_table)

    # =========================================================================
    # PAGE 2: PRODUCTION GOVERNANCE, RISK CONTROLS & EXECUTION SUITE
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("EXHIBIT 4  |  PRODUCTION GOVERNANCE & RISK MITIGATION", exhibit_label_style))
    story.append(
        Paragraph(
            "4. Mandatory Production Engineering Controls & Compliance Guardrails",
            section_hdr,
        )
    )

    warn_html = (
        f"<font size=7.8 color='#B45309'><b>PRINCIPAL ARCHITECT DIRECTIVE — MANDATORY PRE-PRODUCTION CONTROLS</b></font><br/>"
        f"<font size=9 color='#1E293B'>{next_steps}</font>"
    )
    warn_table = Table([[Paragraph(warn_html, body_style)]], colWidths=[532])
    warn_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), warn_bg),
                ("LINEBEFORE", (0, 0), (0, -1), 4.0, warn_amber),
                ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.HexColor("#FDE68A")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#FDE68A")),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(warn_table)
    story.append(Spacer(1, 14))

    # Section 5: Executive Implementation Roadmap (Structured Consulting Deliverable Table)
    story.append(Paragraph("EXHIBIT 5  |  END-TO-END IMPLEMENTATION SUITE (PUBLIC BETA UNLOCKED)", exhibit_label_style))
    story.append(
        Paragraph(
            "5. Cryptographic Vault Persistence & Full Architecture Suite ($0 Beta Access)",
            section_hdr,
        )
    )

    vault_intro = (
        f"This diagnostic has been cryptographically committed to the PickModelAI Qdrant Vector Vault under "
        f"Client Reference ID <b>{_esc(client_ref_id)}</b> using an <b>HMAC-SHA256 blind index</b> and "
        f"<b>AES-256 payload encryption</b> (zero plaintext PII or credentials stored). During our Public Beta, "
        f"your executive team and lead engineers have unrestricted <b>$0 access</b> to the downstream execution modules:"
    )
    story.append(Paragraph(vault_intro, body_style))
    story.append(Spacer(1, 8))

    suite_table = Table(
        [
            [
                Paragraph("MODULE", th_style),
                Paragraph("DELIVERABLE SCOPE & ENGINEERING ARTIFACTS", th_style),
                Paragraph("STATUS", th_style),
            ],
            [
                Paragraph("<b>Product A</b><br/><font size=7.5 color='#64748B'>Genesis Diagnostic</font>", body_style),
                Paragraph(
                    "Executive scope alignment, modality verification, 4-stage pipeline architecture, "
                    "and Managed API vs. Private VPC TCO trade-off analysis.",
                    body_style,
                ),
                Paragraph("<font color='#0F766E'><b>COMPLETED</b></font>", body_style),
            ],
            [
                Paragraph("<b>Product B</b><br/><font size=7.5 color='#64748B'>DIY Engineering Blueprint</font>", body_style),
                Paragraph(
                    "Production-ready LangChain/LangGraph orchestration code, system prompts, deterministic "
                    "confidence routers, webhook schemas, and regional AWS deployment scripts.",
                    body_style,
                ),
                Paragraph("<font color='#005EB8'><b>UNLOCKED ($0)</b></font>", body_style),
            ],
            [
                Paragraph("<b>Product C</b><br/><font size=7.5 color='#64748B'>Security & Governance</font>", body_style),
                Paragraph(
                    "Regional VPC topology (e.g., eu-central-1), pre-inference PII redaction middleware, "
                    "Zero-Data-Retention DPA checklist, and SOC2/GDPR audit controls.",
                    body_style,
                ),
                Paragraph("<font color='#005EB8'><b>UNLOCKED ($0)</b></font>", body_style),
            ],
            [
                Paragraph("<b>Product D</b><br/><font size=7.5 color='#64748B'>TCO & Rollout Plan</font>", body_style),
                Paragraph(
                    "Phased prototype-to-production rollout schedule, FTE/MLOps staffing matrix, "
                    "token stress-test modeling, and 12-month CapEx/OpEx executive budget.",
                    body_style,
                ),
                Paragraph("<font color='#005EB8'><b>UNLOCKED ($0)</b></font>", body_style),
            ],
        ],
        colWidths=[105, 337, 90],
    )
    suite_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), mck_navy),
                ("BACKGROUND", (0, 1), (-1, 1), colors.white),
                ("BACKGROUND", (0, 2), (-1, 2), bg_tint),
                ("BACKGROUND", (0, 3), (-1, 3), colors.white),
                ("BACKGROUND", (0, 4), (-1, 4), bg_tint),
                ("LINEABOVE", (0, 0), (-1, 0), 1.5, mck_navy),
                ("LINEBELOW", (0, 0), (-1, 0), 1.0, mck_blue),
                ("LINEBELOW", (0, 1), (-1, -2), 0.5, rule_light),
                ("LINEBELOW", (0, -1), (-1, -1), 1.5, mck_navy),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(suite_table)
    story.append(Spacer(1, 12))

    # Vault Retrieval Callout Box
    retrieval_html = (
        f"<font size=8.5 color='#051C2C'><b>ZERO-PASSWORD VAULT RETRIEVAL & ARCHITECT FEEDBACK</b></font><br/>"
        f"<font size=8.4 color='#1E293B'>Retain your Client Vault Reference ID (<b>{_esc(client_ref_id)}</b>) to reload "
        f"your encrypted diagnostic state across Products B, C, and D in the PickModelAI Vault Retriever, or to submit "
        f"edge-case telemetry during the Public Beta.</font>"
    )
    retrieval_box = Table([[Paragraph(retrieval_html, body_style)]], colWidths=[532])
    retrieval_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg_header_tint),
                ("LINEBEFORE", (0, 0), (0, -1), 3.5, mck_navy),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(retrieval_box)

    # Build PDF with McKinsey header/footer canvas callbacks
    doc.build(
        story,
        onFirstPage=lambda c, d: _draw_mckinsey_canvas(c, d, client_ref_id, date_str),
        onLaterPages=lambda c, d: _draw_mckinsey_canvas(c, d, client_ref_id, date_str),
    )
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
