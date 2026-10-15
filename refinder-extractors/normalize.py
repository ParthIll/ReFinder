"""Conservative, retrieval-friendly text normalization."""
import re
import unicodedata


def normalize_text(value: object, *, multiline: bool = False) -> str:
    """Keep meaning and punctuation; remove control chars and repeated whitespace."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    if multiline:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return re.sub(r"\s+", " ", text).strip()


def split_paragraphs(text: str) -> list[str]:
    """Split blank-line-separated text; join soft-wrapped lines inside paragraphs."""
    text = normalize_text(text, multiline=True)
    if not text:
        return []
    return [normalize_text(part) for part in re.split(r"\n\s*\n", text) if normalize_text(part)]
