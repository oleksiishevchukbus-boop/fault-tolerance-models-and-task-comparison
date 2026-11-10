"""Word-document helpers shared by make_experiment_report.py.

Taken unchanged from the source study's make_reports.py (the cross-experiment
report builder, not part of this repository) so the per-experiment reports keep
the same styling.
"""
from __future__ import annotations

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


def fmt(x, digits=4):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "-"
    if v == 0:
        return "0"
    if abs(v) >= 1000 or abs(v) < 1e-4:
        return "%.3e" % v
    return ("%%.%df" % digits) % v



def new_document(title: str, subtitle: str) -> Document:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.left_margin = section.right_margin = Inches(0.9)

    h = doc.add_heading(title, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p = doc.add_paragraph(subtitle)
    p.runs[0].italic = True
    p.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    return doc


def add_table(doc: Document, headers: list[str], rows: list[list], caption: str = "",
              landscape_hint: bool = False) -> None:
    if caption:
        c = doc.add_paragraph(caption)
        c.runs[0].bold = True
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, text in enumerate(headers):
        hdr[i].text = str(text)
        for par in hdr[i].paragraphs:
            for run in par.runs:
                run.bold = True
                run.font.size = Pt(8.5)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for par in cells[i].paragraphs:
                for run in par.runs:
                    run.font.size = Pt(8.5)
    doc.add_paragraph()


def add_bullets(doc: Document, items: list[str]) -> None:
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def add_numbered(doc: Document, items: list[str]) -> None:
    for it in items:
        doc.add_paragraph(it, style="List Number")
