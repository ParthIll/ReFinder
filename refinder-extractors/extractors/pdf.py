"""PDF: paragraph-like chunks, each prefixed with page number; OCR not included."""
import re
from pathlib import Path
from pypdf import PdfReader
from ..normalize import normalize_text, split_paragraphs


def _page_paragraphs(text: str) -> list[str]:
    paragraphs = split_paragraphs(text)
    # PDFs frequently lack blank lines. Use complete-line sentence endings as
    # conservative fallback boundaries, not arbitrary single visual line breaks.
    if len(paragraphs) <= 1 and "\n" in text:
        lines = [normalize_text(s) for s in text.splitlines()]
        groups, current = [], []
        for line in lines:
            if not line:
                if current:
                    groups.append(" ".join(current))
                    current = []
                continue
            current.append(line)
            if re.search(r"[.!?][\"'”’)]?$", line):
                groups.append(" ".join(current))
                current = []
        if current:
            groups.append(" ".join(current))
        if groups:
            return groups
    return paragraphs


def extract_pdf(path: Path) -> list[str]:
    reader = PdfReader(path)
    if reader.is_encrypted:
        if reader.decrypt("") == 0:
            raise ValueError("Encrypted PDF requires a password")
    chunks = []
    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        for paragraph_number,paragraph in enumerate(_page_paragraphs(text),1):
            if paragraph:
                chunks.append(f"Page {page_number} Paragraph {paragraph_number}\n{paragraph}")
    return chunks
