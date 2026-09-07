import io
import datetime
from typing import List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_catalogue_pdf(
    company_name: str,
    document_title: str,
    summary_text: str,
    highlight_points: List[str],
    spec_table_data: Optional[List[List[str]]] = None,
    contact_email: str = "support@kiavi.ai",
    contact_phone: str = "",
    official_website: str = ""
) -> bytes:
    """
    Generates a high-end, branded corporate PDF specification / catalogue document.
    Outputs a raw byte stream ready for streaming inline viewing or file downloading.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )
    styles = getSampleStyleSheet()

    # Custom Typography Styles
    company_style = ParagraphStyle(
        'CompanyHeader',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0284c7'), # Cyan-600
        fontName='Helvetica-Bold',
        spaceAfter=3
    )

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'), # Slate-900
        fontName='Helvetica-Bold',
        spaceAfter=6
    )

    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#64748b'), # Slate-500
        spaceAfter=14
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0f172a'),
        fontName='Helvetica-Bold',
        spaceBefore=12,
        spaceAfter=6
    )

    summary_box_style = ParagraphStyle(
        'SummaryBox',
        parent=styles['Normal'],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor('#1e293b'),
        fontName='Helvetica-Oblique'
    )

    bullet_style = ParagraphStyle(
        'BulletPoint',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=14.5,
        textColor=colors.HexColor('#334155'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=5
    )

    footer_style = ParagraphStyle(
        'DocFooter',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#94a3b8'),
        alignment=1 # Center
    )

    story = []

    # 1. Header Banner
    safe_company = company_name.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    story.append(Paragraph(safe_company.upper(), company_style))

    safe_title = document_title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    story.append(Paragraph(safe_title, title_style))

    timestamp_str = datetime.datetime.now().strftime("%B %d, %Y")
    meta_line = f"Official Specifications &amp; Technical Catalogue &bull; Verified Documentation &bull; Date: {timestamp_str}"
    story.append(Paragraph(meta_line, meta_style))

    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#06b6d4'), spaceAfter=14))

    # 2. Executive Summary Callout Box
    if summary_text:
        story.append(Paragraph("Executive Overview", section_heading))
        clean_summary = summary_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        summary_table = Table(
            [[Paragraph(f"<b>Core Summary:</b> {clean_summary}", summary_box_style)]],
            colWidths=[522]
        )
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0fdf4')), # Light Emerald/Mint
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#86efac')),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 12))

    # 3. Key Offerings & Technical Details
    if highlight_points:
        story.append(Paragraph("Key Highlights &amp; Specifications", section_heading))
        for pt in highlight_points:
            clean_pt = pt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{clean_pt}", bullet_style))
        story.append(Spacer(1, 10))

    # 4. Structured Data / Specifications Table (if provided)
    if spec_table_data and len(spec_table_data) > 1:
        story.append(Paragraph("Specifications &amp; Parameter Matrix", section_heading))
        formatted_table_data = []
        for row_idx, row in enumerate(spec_table_data):
            formatted_row = []
            for col in row:
                c_str = str(col).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                is_hdr = (row_idx == 0)
                p_s = ParagraphStyle(
                    f'cell_{row_idx}',
                    fontName='Helvetica-Bold' if is_hdr else 'Helvetica',
                    fontSize=9,
                    leading=12,
                    textColor=colors.HexColor('#ffffff') if is_hdr else colors.HexColor('#1e293b')
                )
                formatted_row.append(Paragraph(c_str, p_s))
            formatted_table_data.append(formatted_row)

        table_obj = Table(formatted_table_data, colWidths=[150, 186, 186])
        table_obj.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')), # Slate-900
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
        ]))
        story.append(table_obj)
        story.append(Spacer(1, 14))

    # 5. Verified Authenticity & Official Contact Footer
    contact_parts = []
    if official_website:
        contact_parts.append(f"<b>Website:</b> {official_website}")
    if contact_email:
        contact_parts.append(f"<b>Email:</b> {contact_email}")
    if contact_phone:
        contact_parts.append(f"<b>Phone:</b> {contact_phone}")
    contact_line = " &nbsp;|&nbsp; ".join(contact_parts) if contact_parts else "Official Enterprise Knowledge Base"

    footer_elements = [
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceBefore=14, spaceAfter=8),
        Paragraph("VERIFIED OFFICIAL CATALOGUE &bull; POWERED BY KIAVI IQ ENTERPRISE AI", ParagraphStyle(
            'Stmp', parent=footer_style, fontName='Helvetica-Bold', fontSize=8, textColor=colors.HexColor('#0891b2')
        )),
        Spacer(1, 3),
        Paragraph(contact_line, footer_style),
        Paragraph("&copy; 2026 Kiavi AI. All specifications and product data certified from official documentation.", footer_style)
    ]
    story.append(KeepTogether(footer_elements))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
