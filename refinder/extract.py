"""Bridge to the team's format extractors in refinder-extractors/ (copied from main, kept unmodified).

That folder isn't importable as-is (hyphenated name, and its relative imports expect a package), so it is
loaded here under the name `refinder_extractors`.

The extractors return pieces (paragraphs, slides, sheet tiles, CSV rows) with location labels written into the
text ("Page 3 Paragraph 2\\n...", "Slide 4\\n...", "Row 12: ..."). The labels carry no meaning for the embedding
model, so we parse them off into a `location` dict that stays attached to the piece, and to every chunk built
from it, for showing where a match came from.
"""

import importlib.util
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_PKG_NAME = "refinder_extractors"
_PKG_DIR = Path(__file__).resolve().parent.parent / "refinder-extractors"


def _load_package() -> None:
    if _PKG_NAME in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        _PKG_NAME, _PKG_DIR / "__init__.py", submodule_search_locations=[str(_PKG_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_NAME] = module
    spec.loader.exec_module(module)


_load_package()
# pypdf logs every recoverable quirk in a PDF ("Ignoring wrong pointing object ..."); keep only real errors.
logging.getLogger("pypdf").setLevel(logging.ERROR)
from refinder_extractors.extractors import EXTRACTORS, extract  # noqa: E402

SUPPORTED_SUFFIXES = frozenset(EXTRACTORS)


class UnsupportedFile(Exception):
    pass


@dataclass
class Piece:
    text: str
    location: dict  # always has "piece" (1-based order in the file); plus page/paragraph/slide/row/table/sheet/range


def _ints(**groups: str | None) -> dict:
    return {k: int(v) for k, v in groups.items() if v is not None}


# Label formats per file type, as written by refinder-extractors. Only applied to the matching suffixes so a
# .txt paragraph that happens to start with "Row 5:" isn't misread.
LABELS: dict[str, tuple[re.Pattern, Callable[[re.Match], dict]]] = {
    ".pdf": (re.compile(r"^Page (\d+)(?: Paragraph (\d+))?\n"), lambda m: _ints(page=m[1], paragraph=m[2])),
    ".pptx": (re.compile(r"^Slide (\d+)\n"), lambda m: _ints(slide=m[1])),
    ".csv": (re.compile(r"^Row (\d+): "), lambda m: _ints(row=m[1])),
    ".tsv": (re.compile(r"^Row (\d+): "), lambda m: _ints(row=m[1])),
    ".docx": (re.compile(r"^Table (\d+):\n"), lambda m: _ints(table=m[1])),
    # Keep the sheet name in the text (it's often meaningful, e.g. "Budget 2026"); drop only the cell range.
    ".xlsx": (re.compile(r"^(Sheet: .*?); Range: ([A-Z]+\d+:[A-Z]+\d+)\n"), lambda m: {"sheet": m[1][7:], "range": m[2]}),
}


def _strip_label(raw: str, suffix: str) -> tuple[str, dict]:
    pattern, parse = LABELS.get(suffix, (None, None))
    m = pattern.match(raw) if pattern else None
    if not m:
        return raw, {}
    rest = raw[m.end():]
    return (f"{m[1]}\n{rest}" if suffix == ".xlsx" else rest), parse(m)


def extract_pieces(path: Path) -> list[Piece]:
    """The file's pieces in order, with location labels moved from the text into `location`."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES or path.name.startswith(("~$", ".")):
        raise UnsupportedFile(f"skipped type {suffix or path.name}")
    pieces = []
    for i, raw in enumerate(extract(path), 1):
        text, location = _strip_label(raw, suffix)
        if text.strip():
            pieces.append(Piece(text, {"piece": i, **location}))
    return pieces


def pieces_from_text(text: str) -> list[Piece]:
    """Split already-extracted text into paragraph pieces (blank-line separated, else one per line)."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.splitlines() if p.strip()]
    return [Piece(p, {"piece": i}) for i, p in enumerate(parts, 1)]


def extract_text(path: Path) -> str:
    """Returns the file's text without location labels (possibly empty, e.g. a scanned PDF)."""
    return "\n\n".join(p.text for p in extract_pieces(path))


def describe_location(chunk_number: int, n_chunks: int, first: dict, last: dict) -> str:
    """Where a chunk sits in its file, e.g. "chunk 3/12, page 2", "chunk 5/9, rows 27-30" or "chunk 2/4"."""
    def rng(key: str) -> str:
        a, b = first[key], last.get(key, first[key])
        return f"{a}" if a == b else f"{a}-{b}"

    where = f"chunk {chunk_number}/{n_chunks}"
    for key, singular, plural in (("page", "page", "pages"), ("slide", "slide", "slides"), ("row", "row", "rows")):
        if key in first:
            return f"{where}, {singular if first[key] == last.get(key, first[key]) else plural} {rng(key)}"
    if "sheet" in first:
        return f"{where}, sheet {first['sheet']}"
    return where
