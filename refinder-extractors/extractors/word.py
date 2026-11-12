"""DOCX: ordered body paragraphs and entire tables; no page-layout inference."""
from pathlib import Path
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from ..normalize import normalize_text


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
    return chunks
