"""OCR for standalone images and for images embedded in PDFs and DOCX files, using Tesseract.

Needs the `tesseract` program installed (brew install tesseract), plus the pytesseract and pillow packages.
"""
import hashlib
import re
import unicodedata
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"})
OCR_MIN_CONF = 60  # Tesseract word confidence (0-100); words below this are dropped as noise.
OCR_MIN_SIDE = 1000  # Small images OCR poorly, so upscale until the shorter side is this many pixels.
EMBEDDED_MIN_SIDE = 100  # Skip logos, bullets, and other decorative images inside documents.


def ocr_image(img: Image.Image) -> str:
    """OCR one open PIL image. Returns "" when no readable text is found."""
    # Phone photos store rotation in EXIF metadata; apply it so text reads upright.
    img = ImageOps.exif_transpose(img)
    gray = ImageOps.grayscale(img)

    # Tesseract reads small text poorly, so scale up small images (capped at 3x).
    short_side = min(gray.size)
    if short_side < OCR_MIN_SIDE:
        scale = min(3.0, OCR_MIN_SIDE / short_side)
        new_size = (round(gray.width * scale), round(gray.height * scale))
        gray = gray.resize(new_size, Image.LANCZOS)

    data = pytesseract.image_to_data(gray, output_type=Output.DICT)
    return _clean_ocr(_words_to_lines(data))


def ocr_embedded(img: Image.Image, seen: set[str]) -> str:
    """OCR an image found inside a document. Returns "" for tiny images, for images already OCR'd in this
    file (`seen` holds their digests, so a logo repeated on every page is read once), and for no text."""
    if min(img.size) < EMBEDDED_MIN_SIDE:
        return ""
    digest = hashlib.sha256(img.tobytes()).hexdigest()
    if digest in seen:
        return ""
    seen.add(digest)
    return ocr_image(img)


def extract_image(path: Path) -> list[str]:
    """Standalone image file: one piece with all of its text, or nothing if it has none."""
    with Image.open(path) as img:
        text = ocr_image(img)
    return [text] if text else []


def _words_to_lines(data: dict) -> str:
    """Rebuild lines and paragraphs from Tesseract's word table, dropping low-confidence words."""
    # Step 1: group the words that survive the confidence filter by their line.
    lines: dict[tuple, list[str]] = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0

        if not word or conf < OCR_MIN_CONF:
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)

    # Step 2: start a new paragraph whenever the block/paragraph id changes.
    # Tesseract emits words in reading order, and dict keeps that order.
    paragraphs: list[list[str]] = []
    last_par = None
    for key, words in lines.items():
        par = key[:2]
        if par != last_par:
            paragraphs.append([])
            last_par = par
        paragraphs[-1].append(" ".join(words))

    return "\n\n".join("\n".join(p) for p in paragraphs if p)


def _clean_ocr(text: str) -> str:
    """Tidy OCR output: normalize Unicode, rejoin hyphenated line breaks, drop junk lines."""
    # Normalize Unicode forms (ligatures, full-width characters) and strip control characters.
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())

    # Rejoin words split across a line break: "exam-\nple" -> "example".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Drop junk lines. Speckled backgrounds produce stray symbols, so keep only
    # lines with at least 2 letters/digits that make up at least half the line.
    kept = []
    for line in text.splitlines():
        line = line.strip()
        alnum = sum(ch.isalnum() for ch in line)
        if alnum < 2 or alnum / len(line) < 0.5:
            continue
        kept.append(line)

    return "\n".join(kept)
