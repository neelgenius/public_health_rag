import pdfplumber
from pathlib import Path
from typing import List, Dict

def extract_pdf_pages(pdf_path: Path) -> List[Dict]:
    """
    Extract text page-by-page to preserve citation grounding.
    """
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({
                    "page_number": idx + 1,
                    "text": text
                })

    return pages
