"""DOCX: ordered body paragraphs and entire tables; no page-layout inference. Images embedded in the file are
OCR'd into "Image N" chunks after the body."""
import io
from pathlib import Path
from collections.abc import Iterator
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image, UnidentifiedImageError
from ..normalize import normalize_text
from .ocr import ocr_embedded


def _embedded_images(doc) -> Iterator[Image.Image]:
    for rel in doc.part.rels.values():
        if rel.reltype != RT.IMAGE:
            continue
        try:
            img = Image.open(io.BytesIO(rel.target_part.blob))
            img.load()  # surface truncated-file errors here instead of later
        except (UnidentifiedImageError, OSError):
            continue  # EMF/WMF/SVG and corrupt images that Pillow can't read
        yield img


def extract_docx(path: Path) -> list[str]:
    doc = Document(path)
    chunks = []
    table_number = 0
    for item in doc.iter_inner_content():
        if isinstance(item, Paragraph):
            text = normalize_text(item.text)
            if text:
                chunks.append(text)
        elif isinstance(item, Table):
            table_number += 1
            rows = []
            for row in item.rows:
                cells = [normalize_text(cell.text) for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                chunks.append(f"Table {table_number}:\n" + "\n".join(rows))

    seen_images: set[str] = set()
    image_number = 0
    for img in _embedded_images(doc):
        text = ocr_embedded(img, seen_images)
        if text:
            image_number += 1
            chunks.append(f"Image {image_number}:\n{text}")
    return chunks
