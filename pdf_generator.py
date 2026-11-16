# pdf_generator.py (Product A: Genesis Architecture Diagnostic — Public Beta)
import io
import html
from datetime import datetime
from typing import Dict, Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


def _esc(text: Any) -> str:
    """Safely escapes text for ReportLab XML Paragraphs without double-escaping."""
    if text is None:
        return ""
    unescaped = html.unescape(str(text))
    return html.escape(unescaped).replace("\n", "<br/>")


def generate_blueprint_pdf(
    blueprint_data: Dict[str, Any],
    client_ref_id: str = "PMAI-BETA-0001",
) -> bytes:
    """
    Generates a 2-page executive Platypus PDF for Product A (Genesis Architecture Diagnostic).
    Accepts the exact FreeTierBlueprint dictionary produced by app.py.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=36,
        title=f"PlanModelAI Product A Diagnostic - {client_ref_id}",
    )

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#1B365D")
    teal = colors.HexColor("#2C6E7F")
    slate = colors.HexColor("#1E293B")
    muted = colors.HexColor("#64748B")
    light_bg = colors.HexColor("#F1F5F9")
    accent_bg = colors.HexColor("#E0F2FE")
    border_gray = colors.HexColor("#CBD5E1")

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16.5,
        leading=20,
        textColor=colors.white,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#E2E8F0"),
    )
    section_hdr = ParagraphStyle(
        "SectionHdr",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=navy,
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.5,
        textColor=slate,
    )
    small_style = ParagraphStyle(
        "SmallStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.3,
        leading=11.5,
        textColor=muted,
    )

    story = []
    date_str = datetime.now().strftime("%B %d, %Y")

    project_title = _esc(blueprint_data.get("project_short_title", "AI Architecture Blueprint"))
    grade = _esc(blueprint_data.get("feasibility_grade", "Grade A (Fast & Affordable to Build)"))
    modality = _esc(blueprint_data.get("detected_modality", "TEXT_REASONING"))
    summary = _esc(blueprint_data.get("user_idea_summary", ""))
    workflow_steps = blueprint_data.get("workflow_steps", [])
    next_steps = _esc(blueprint_data.get("next_steps_recommendation", ""))

    paid = blueprint_data.get("paid_stack", {})
    oss = blueprint_data.get("open_source_stack", {})
    paid_cost_num = int(paid.get("monthly_cost_number_usd", 0))
    oss_cost_num = int(oss.get("monthly_cost_number_usd", 120))

    # =========================================================================
    # PAGE 1: GENESIS DIAGNOSTIC & DUAL-TRACK CLOUD VS. VPC ARCHITECTURE
    # =========================================================================
    header_table = Table(
        [
            [
                Paragraph(
                    f"PLANMODELAI — {project_title.upper()}<br/>"
                    f"<font size=9.5 color='#BAE6FD'>Product A: Genesis Architecture Diagnostic | Public Beta Edition</font>",
                    title_style,
                ),
                Paragraph(
                    f"<font size=8 color='#BAE6FD'><b>CLIENT VAULT ID</b></font><br/>"
                    f"<font size=10.5 color='#FFFFFF'><b>{_esc(client_ref_id)}</b></font><br/>"
                    f"<font size=8 color='#E2E8F0'>{date_str}</font>",
                    ParagraphStyle("HdrRight", parent=subtitle_style, alignment=TA_RIGHT),
                ),
            ]
        ],
        colWidths=[377, 155],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), navy),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 10))

    # KPI Scorecards
    kpi_table = Table(
        [
            [
                Paragraph(f"<font size=7.8 color='#64748B'>DETECTED MODALITY</font><br/><b><font size=9.5 color='#1B365D'>{modality}</font></b>", body_style),
                Paragraph(f"<font size=7.8 color='#64748B'>FEASIBILITY RATING</font><br/><b><font size=9.2 color='#0F766E'>{grade}</font></b>", body_style),
                Paragraph(f"<font size=7.8 color='#64748B'>MANAGED API EST.</font><br/><b><font size=10 color='#1B365D'>${paid_cost_num:,} / mo</font></b>", body_style),
                Paragraph(f"<font size=7.8 color='#64748B'>PRIVATE VPC EST.</font><br/><b><font size=10 color='#1B365D'>${oss_cost_num:,} / mo</font></b>", body_style),
            ]
        ],
        colWidths=[125, 167, 120, 120],
    )
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), light_bg),
                ("BOX", (0, 0), (-1, -1), 0.75, border_gray),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, border_gray),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 8))

    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Scope & Business Objective", section_hdr))
    story.append(Paragraph(summary, body_style))
    story.append(Spacer(1, 6))

    # Section 2: 4-Step End-to-End Pipeline Flow
    story.append(Paragraph("2. End-to-End System Workflow", section_hdr))
    step_cells = []
    for idx, step_text in enumerate(workflow_steps[:4], 1):
        step_cells.append(
            Paragraph(
                f"<font size=8 color='#2C6E7F'><b>STEP {idx}</b></font><br/>{_esc(step_text)}",
                body_style,
            )
        )
    if step_cells:
        flow_table = Table([step_cells], colWidths=[133] * len(step_cells))
        flow_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), light_bg),
                    ("BOX", (0, 0), (-1, -1), 0.5, border_gray),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, border_gray),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(flow_table)
    story.append(Spacer(1, 8))

    # Section 3: Dual-Track Architecture & TCO Comparison
    story.append(Paragraph("3. Option 1 (Managed Cloud API) vs. Option 2 (Private AWS VPC)", section_hdr))
    th_style = ParagraphStyle("Th", parent=body_style, textColor=colors.white)
    stack_table = Table(
        [
            [
                Paragraph("<b>Dimension</b>", th_style),
                Paragraph("<b>Option 1: Managed Cloud APIs</b>", th_style),
                Paragraph("<b>Option 2: Self-Hosted Private VPC</b>", th_style),
            ],
            [
                Paragraph("<b>Model & Inference Stack</b>", body_style),
                Paragraph(_esc(paid.get("model_stack", "")), body_style),
                Paragraph(_esc(oss.get("model_stack", "")), body_style),
            ],
            [
                Paragraph("<b>Vector Database</b>", body_style),
                Paragraph(_esc(paid.get("vector_db", "")), body_style),
                Paragraph(_esc(oss.get("vector_db", "")), body_style),
            ],
            [
                Paragraph("<b>Architectural Fit</b>", body_style),
                Paragraph(_esc(paid.get("why_this_stack", "")), body_style),
                Paragraph(_esc(oss.get("why_this_stack", "")), body_style),
            ],
            [
                Paragraph("<b>Monthly Run-Rate Cost</b>", body_style),
                Paragraph(_esc(paid.get("estimated_monthly_cost", "")), body_style),
                Paragraph(_esc(oss.get("estimated_monthly_cost", "")), body_style),
            ],
            [
                Paragraph("<b>Scaling Optimization</b>", body_style),
                Paragraph(_esc(paid.get("scaling_path", "")), body_style),
                Paragraph(_esc(oss.get("scaling_path", "")), body_style),
            ],
        ],
        colWidths=[116, 208, 208],
    )
    stack_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), teal),
                ("BACKGROUND", (0, 1), (0, -1), light_bg),
                ("BOX", (0, 0), (-1, -1), 0.75, border_gray),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, border_gray),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(stack_table)

    # =========================================================================
    # PAGE 2: CRITICAL ENGINEERING GLUE & PUBLIC BETA SUITE ROADMAP
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("4. Critical Production Engineering & Guardrail Warning", section_hdr))
    warn_table = Table([[Paragraph(next_steps, body_style)]], colWidths=[532])
    warn_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#F59E0B")),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(warn_table)
    story.append(Spacer(1, 14))

    # Public Beta Callout Box (Replaces BUILD50 Checkout Copy)
    cta_html = (
        f"<font size=10.5 color='#1B365D'><b>PUBLIC BETA ACCESS — COMPLETE 4-PRODUCT ARCHITECTURE SUITE ($0)</b></font><br/><br/>"
        f"<font size=8.8 color='#1E293B'>"
        f"You have completed <b>Product A: Genesis Architecture Diagnostic</b>. Your diagnostic has been cryptographically "
        f"stored in our Qdrant Vector Vault under Client Reference ID <b>{_esc(client_ref_id)}</b> using an HMAC-SHA256 blind index "
        f"and AES-256 payload encryption (zero passwords or plaintext PII stored).<br/><br/>"
        f"During our <b>Public Beta</b>, the entire PlanModelAI engineering suite is unlocked at <b>$0</b>:<br/><br/>"
        f"• <b>Product B — DIY Engineering Blueprint (Beta Unlocked):</b> Copy-paste LangChain/LangGraph setup code, "
        f"exact system prompts, deterministic guardrails, and AWS deployment steps tailored to your stack.<br/>"
        f"• <b>Product C — Enterprise Security & Governance Pack (Beta Unlocked):</b> Private VPC network topology, "
        f"PII redaction pipelines, and SOC2/SOX audit controls.<br/>"
        f"• <b>Product D — Full Delivery & TCO Execution Plan (Beta Unlocked):</b> Multi-phase rollout schedule, "
        f"engineering team sizing, and CapEx/OpEx financial modeling.<br/><br/>"
        f"• <b>Zero-Password Retrieval & Beta Feedback:</b> Save your Client Vault ID (<b>{_esc(client_ref_id)}</b>) to "
        f"retrieve your encrypted diagnostic anytime in our Vault Retriever, or share beta edge-case feedback with the architect."
        f"</font>"
    )

    cta_table = Table([[Paragraph(cta_html, body_style)]], colWidths=[532])
    cta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), accent_bg),
                ("BOX", (0, 0), (-1, -1), 1.0, teal),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    story.append(cta_table)
    story.append(Spacer(1, 14))
    story.append(
        Paragraph(
            "Disclaimer: Cost estimates reflect baseline token assumptions and standard AWS US-East-1 compute rates. "
            "Generated by PlanModelAI Product A (Public Beta) with Deterministic Modality Guardrails.",
            ParagraphStyle("Footer", parent=small_style, alignment=TA_CENTER),
        )
    )

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes