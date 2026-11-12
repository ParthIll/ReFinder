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
    # Include images referenced from headers/footers too; never fetch external links.
    relationships = (rel for part in doc.part.package.parts for rel in part.rels.values())
    for rel in relationships:
        if rel.reltype != RT.IMAGE or rel.is_external:
            continue
        try:
            img = Image.open(io.BytesIO(rel.target_part.blob))
            img.load()  # surface truncated-file errors here instead of later
        except (UnidentifiedImageError, OSError):
            continue  # EMF/WMF/SVG and corrupt images that Pillow can't read
        yield img


def extract_docx(path: Path, on_visual=None) -> list[str]:
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
    for source_image_number, img in enumerate(_embedded_images(doc), 1):
        fallback = (lambda image, info, i=source_image_number: on_visual(image, {"image": i}, info)) if on_visual else None
        try:
            text = ocr_embedded(img, seen_images, on_fallback=fallback)
        finally:
            img.close()
        if text:
            chunks.append(f"Image {source_image_number}:\n{text}")
    return chunks
