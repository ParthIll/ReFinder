"""Individual format readers; each yields plain normalized chunk strings."""
from pathlib import Path
from collections.abc import Iterator
from .word import extract_docx
from .excel import extract_xlsx
from .powerpoint import extract_pptx
from .pdf import extract_pdf
from .text import extract_text, extract_delimited

EXTRACTORS = {
    ".docx": extract_docx,
    ".xlsx": extract_xlsx,
    ".pptx": extract_pptx,
    ".pdf": extract_pdf,
    ".txt": extract_text,
    ".md": extract_text,
    ".markdown": extract_text,
    ".csv": extract_delimited,
    ".tsv": extract_delimited,
}


def extract(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix not in EXTRACTORS:
        raise ValueError(f"Unsupported extension: {suffix}")
    return [chunk for chunk in EXTRACTORS[suffix](path) if chunk]
