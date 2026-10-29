"""Plain text/Markdown: paragraphs; CSV/TSV: header-aware rows."""
import csv
from pathlib import Path
from ..normalize import split_paragraphs, normalize_text


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    # Avoid indexing binary files disguised as .txt or .md.
    if b"\x00" in data:
        raise ValueError("Unexpected binary data in text file")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252")


def extract_text(path: Path) -> list[str]:
    return split_paragraphs(_read_text(path))


def extract_delimited(path: Path) -> list[str]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    text = _read_text(path)
    rows = csv.reader(text.splitlines(), delimiter=delimiter)
    headers = next(rows, None)
    if headers is None:
        return []
    headers = [normalize_text(v) or f"Column {n}" for n, v in enumerate(headers, 1)]
    chunks = []
    for row_number, row in enumerate(rows, 2):
        fields = [
            f"{headers[i] if i < len(headers) else f'Column {i + 1}'}: {normalize_text(value)}"
            for i, value in enumerate(row) if normalize_text(value)
        ]
        if fields:
            chunks.append(f"Row {row_number}: " + "; ".join(fields))
    return chunks
