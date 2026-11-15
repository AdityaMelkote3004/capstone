"""Render paper/SUPPLEMENT.md to paper/SUPPLEMENT.pdf (A4) for submission.

Markdown -> HTML (python-markdown, tables extension) -> PDF (PyMuPDF Story).
Usage: python paper/build_supplement_pdf.py
"""
import os

import fitz  # PyMuPDF
import markdown

H = os.path.dirname(os.path.abspath(__file__))
SRC, OUT = os.path.join(H, "SUPPLEMENT.md"), os.path.join(H, "SUPPLEMENT.pdf")

CSS = """
body { font-family: serif; font-size: 9.5pt; line-height: 1.3; }
h1 { font-size: 15pt; margin-bottom: 2pt; }
h2 { font-size: 12pt; margin-top: 10pt; margin-bottom: 3pt; }
h3 { font-size: 10.5pt; margin-top: 8pt; margin-bottom: 2pt; }
p { margin-top: 2pt; margin-bottom: 3pt; }
ul { margin-top: 1pt; margin-bottom: 2pt; }
li { margin-bottom: 1pt; }
table { border-collapse: collapse; margin-top: 3pt; margin-bottom: 5pt; font-size: 8pt; }
th, td { border: 0.5pt solid #888; padding: 1.5pt 3pt; text-align: left; }
th { font-weight: bold; }
code { font-family: monospace; font-size: 8pt; }
"""


def normalise(text):
    """python-markdown needs a blank line before a list and 4-space nesting; the
    supplement is written with 2-space nesting (GitHub style)."""
    out, prev = [], ""
    for line in text.splitlines():
        n = len(line) - len(line.lstrip(" "))
        line = " " * (2 * n) + line.lstrip(" ")
        is_item = line.lstrip().startswith(("- ", "1. ", "2. ", "3. "))
        prev_item = prev.lstrip().startswith(("- ", "1. ", "2. ", "3. ")) or prev.startswith(" ")
        if is_item and prev.strip() and not prev_item:
            out.append("")
        if line.lstrip().startswith("|") and prev.strip() and not prev.lstrip().startswith("|"):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def main():
    text = normalise(open(SRC, encoding="utf-8").read())
    html = markdown.markdown(text, extensions=["tables", "sane_lists"])
    story = fitz.Story(html=html, user_css=CSS)
    writer = fitz.DocumentWriter(OUT)
    page = fitz.paper_rect("a4")
    where = page + (50, 50, -50, -50)
    more = 1
    while more:
        dev = writer.begin_page(page)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    doc = fitz.open(OUT)
    for i, p in enumerate(doc, 1):           # page numbers
        p.insert_text((page.width / 2 - 8, page.height - 25), str(i), fontsize=8)
    doc.saveIncr()
    print(f"wrote {OUT}: {doc.page_count} pages")


if __name__ == "__main__":
    main()
