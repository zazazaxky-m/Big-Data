"""Membangun laporan .docx dan .pdf dari laporan/Laporan_Tugas_PySpark.md.

Kebutuhan: pandoc, python-docx, LibreOffice (untuk PDF).
    python scripts/build_laporan.py
"""
import os
import shutil
import subprocess

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAP = os.path.join(ROOT, "laporan")
MD = os.path.join(LAP, "Laporan_Tugas_PySpark.md")
DOCX = os.path.join(LAP, "Laporan_Tugas_PySpark.docx")

INK = RGBColor(0x0B, 0x0B, 0x0B)
ACCENT = RGBColor(0x18, 0x4F, 0x95)


def set_cell_border(cell, color="C3C2B7", size="4"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), size)
        el.set(qn("w:color"), color)
        borders.append(el)
    tc_pr.append(borders)


def shade(cell, fill="E8EEF7"):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def style_docx(path):
    doc = Document(path)
    styles = {s.name: s for s in doc.styles}
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        if name in styles:
            st = styles[name]
            st.font.name = "Calibri"
            st.font.size = Pt(11)
            if st.element.rPr is not None and st.element.rPr.rFonts is not None:
                st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    for lvl, size in ((1, 16), (2, 13), (3, 11.5)):
        name = f"Heading {lvl}"
        if name in styles:
            st = styles[name]
            st.font.name = "Calibri"
            st.font.size = Pt(size)
            st.font.bold = True
            st.font.color.rgb = ACCENT if lvl == 1 else INK
    if "Title" in styles:
        st = styles["Title"]
        st.font.name = "Calibri"
        st.font.size = Pt(20)
        st.font.color.rgb = INK
    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for r, row in enumerate(table.rows):
            for cell in row.cells:
                set_cell_border(cell)
                if r == 0:
                    shade(cell)
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(9.5)
                        if r == 0:
                            run.font.bold = True
    # Gambar di tengah
    for p in doc.paragraphs:
        if p._p.xpath(".//pic:pic"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for sec in doc.sections:
        from docx.shared import Cm
        sec.left_margin = sec.right_margin = Cm(2.2)
        sec.top_margin = sec.bottom_margin = Cm(2.0)
    doc.save(path)


def main():
    subprocess.run(["pandoc", MD, "-o", DOCX, "--resource-path", LAP], check=True, cwd=LAP)
    style_docx(DOCX)
    if shutil.which("soffice"):
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", LAP, DOCX],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("OK:", DOCX)


if __name__ == "__main__":
    main()
