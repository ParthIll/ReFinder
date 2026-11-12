"""PDF: paragraph-like chunks, each prefixed with page number. Images on each page (including scanned pages with no
text layer) are OCR'd into "Page N Image K" chunks."""
import re
from pathlib import Path
from pypdf import PdfReader
from ..normalize import normalize_text, split_paragraphs
from .ocr import ocr_embedded


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


def _page_image_texts(page, seen: set[str]) -> list[str]:
    """OCR text of each image on a page. Skips images that can't be decoded (some PDF image formats aren't
    supported by Pillow) rather than failing the whole file."""
    try:
        images = list(page.images)
    except Exception:  # pypdf raises on some broken image streams
        return []

    texts = []
    for image in images:
        try:
            decoded = image.image
        except Exception:
            continue
        if decoded is None:
            continue
        text = ocr_embedded(decoded, seen)
        if text:
            texts.append(text)
    return texts


def extract_pdf(path: Path) -> list[str]:
    reader = PdfReader(path)
    if reader.is_encrypted:
        if reader.decrypt("") == 0:
            raise ValueError("Encrypted PDF requires a password")
    chunks = []
    seen_images: set[str] = set()
    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        for paragraph_number, paragraph in enumerate(_page_paragraphs(text), 1):
            if paragraph:
                chunks.append(f"Page {page_number} Paragraph {paragraph_number}\n{paragraph}")
        for image_number, image_text in enumerate(_page_image_texts(page, seen_images), 1):
            chunks.append(f"Page {page_number} Image {image_number}\n{image_text}")
    return chunks
