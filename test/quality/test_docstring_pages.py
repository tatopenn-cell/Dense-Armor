"""Item C6: docstrings that cite pages must say which convention they use.

A docstring that cites a page number must state whether the number is a
printed page of the journal or a PDF page index. If it says "printed
pages", the number must exist in the source document (checked against
the PDF in /content when available); if it says "PDF page indices", the
statement alone is enough, since the PDF page range is not shipped with
the library.

The check runs on the modules in ``forest/`` and ``rules/`` that cite at
least one page. It fails when a module cites pages without declaring the
convention, or when a "printed page" number is outside the document.
"""

import re
from pathlib import Path

import pytest

MODULES = [
    "dense_armor/utility/forest/arf.py",
    "dense_armor/utility/forest/amf.py",
    "dense_armor/utility/forest/oxt.py",
    "dense_armor/utility/rules/amrules.py",
]

# The only PDF present in a Colab session that supports a printed-page
# claim is Mastelini's thesis (OXT). Its first content page is PDF page 1
# (frontispiece), so the printed range is 1..N-1 for a PDF of N pages.
THESIS_PDF = (
    "/content/mastelini2023_thesis_efficient_online_tree_rule_"
    "distance_algorithms.pdf"
)

PAGE_RE = re.compile(r"pages?\s+(\d+)(?:\s*(?:-|–|and|,)\s*(\d+))?", re.I)


def _module_doc(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    m = re.match(r'^"""(.*?)"""', text, re.DOTALL)
    if m is None:
        return ""
    return m.group(1)


def _cited_pages(doc: str) -> list[int]:
    found = []
    for m in PAGE_RE.finditer(doc):
        found.append(int(m.group(1)))
        if m.group(2):
            found.append(int(m.group(2)))
    return found


@pytest.mark.parametrize("path", MODULES)
def test_docstring_declares_page_convention(path: str):
    doc = _module_doc(path)
    assert doc, f"{path}: no module docstring"
    pages = _cited_pages(doc)
    if not pages:
        pytest.skip(f"{path}: no page citations")

    # Look for the positive declaration, not the words themselves:
    # "page numbers below are printed pages of the thesis" for a
    # printed-page claim, "Page numbers below are PDF page indices" for
    # a PDF-index claim. A sentence that says "not the printed page
    # numbers of the journal" contains the words but is not a claim.
    printed_claim = bool(re.search(
        r"page numbers below are printed pages", doc, re.I
    ))
    pdf_claim = bool(re.search(
        r"page numbers below are PDF page indices", doc, re.I
    ))
    assert printed_claim or pdf_claim, (
        f"{path}: cites {pages} without a positive declaration of the "
        "convention (printed pages or PDF page indices)"
    )
    assert not (printed_claim and pdf_claim), (
        f"{path}: declares both conventions at once"
    )


def test_printed_pages_exist_in_the_thesis():
    """The OXT docstring says 'printed pages of the thesis'.

    Every printed page it cites must be inside the thesis: for a PDF of
    N pages the printed pages run from 1 to N-1 (page 1 of the PDF is
    the frontispiece, printed page 1 is PDF page 2 in the digital copy).
    """
    doc = _module_doc("dense_armor/utility/forest/oxt.py")
    if not re.search(r"page numbers below are printed pages", doc, re.I):
        pytest.skip("oxt.py does not claim printed pages")

    pdf = Path(THESIS_PDF)
    if not pdf.exists():
        pytest.skip(f"{pdf} not available; cannot verify page numbers")

    from pypdf import PdfReader

    n_pdf = len(PdfReader(str(pdf)).pages)
    printed_max = n_pdf - 1
    cited = [p for p in _cited_pages(doc) if p > 0]
    too_high = [p for p in cited if p > printed_max]
    assert not too_high, (
        f"oxt.py cites printed pages {too_high} beyond the thesis "
        f"(printed range 1..{printed_max}, PDF has {n_pdf} pages)"
    )
