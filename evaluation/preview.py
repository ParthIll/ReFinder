"""File previews for the desktop app: what the preview panel shows for each file type.

PDF and DOCX are rendered in the app from the raw file (pdf.js, docx-preview); spreadsheets and CSVs come back as
tables, slide decks as per-slide text plus a Quick Look image of the first slide (macOS), everything else as text.
Every function takes a path the caller has already checked is indexed.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MAX_ROWS = 500  # per sheet / CSV
MAX_COLS = 40
MAX_SHEETS = 10
MAX_TEXT_CHARS = 400_000
THUMB_DIR = Path(tempfile.gettempdir()) / "refinder-thumbnails"

RENDERED = {".pdf": "pdf", ".docx": "docx"}


def preview(path: Path, text: str) -> dict[str, Any]:
    """{"kind": "pdf" | "docx" | "table" | "slides" | "text", ...}. `text` is the extracted text (fallback)."""
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
        return {"kind": "image"}
    if suffix in RENDERED:
        return {"kind": RENDERED[suffix]}
    try:
        if suffix in (".csv", ".tsv"):
            return {"kind": "table", "sheets": [_csv_sheet(path, "\t" if suffix == ".tsv" else ",")]}
        if suffix == ".xlsx":
            sheets = [sh for sh in _xlsx_sheets(path) if sh["rows"]]  # skip empty and chart-only sheets
            if sheets:
                return {"kind": "table", "sheets": sheets}
        if suffix == ".pptx":
            slides = _slides(path)
            key = quicklook_html(path)
            if key:  # the real slides (layout, images, fonts) as HTML; slide text kept for highlighting/fallback
                return {"kind": "html", "key": key, "slides": slides}
            return {"kind": "slides", "slides": slides, "thumbnail": sys.platform == "darwin"}
    except Exception:  # unreadable or unusual file: fall back to the extracted text
        pass
    if suffix in (".txt", ".md", ".markdown", ".csv", ".tsv"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return {"kind": "text", "text": text[:MAX_TEXT_CHARS], "markdown": suffix in (".md", ".markdown")}


def _csv_sheet(path: Path, delimiter: str) -> dict[str, Any]:
    rows, truncated = [], False
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        for i, row in enumerate(csv.reader(f, delimiter=delimiter)):
            if i >= MAX_ROWS:
                truncated = True
                break
            rows.append([c for c in row[:MAX_COLS]])
    return {"name": path.name, "rows": rows, "truncated": truncated}


def _xlsx_sheets(path: Path) -> list[dict[str, Any]]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheets = []
        for ws in wb.worksheets[:MAX_SHEETS]:
            ws.reset_dimensions()  # stored dimensions are often wrong; read what is actually there
            rows, truncated = [], False
            for i, row in enumerate(ws.iter_rows(values_only=True, max_col=MAX_COLS)):
                if i >= MAX_ROWS:
                    truncated = True
                    break
                rows.append(["" if v is None else str(v) for v in row])
            # Drop trailing empty rows and columns (read-only sheets often report a too-large range).
            while rows and not any(rows[-1]):
                rows.pop()
            width = max((max((j + 1 for j, v in enumerate(r) if v), default=0) for r in rows), default=0)
            sheets.append({"name": ws.title, "rows": [r[:width] for r in rows], "truncated": truncated})
        return sheets
    finally:
        wb.close()


def _slides(path: Path) -> list[dict[str, Any]]:
    from refinder.extract import extract_pieces

    slides: dict[int, list[str]] = {}
    for piece in extract_pieces(path):
        slides.setdefault(piece.location.get("slide", 0), []).append(piece.text)
    return [{"number": n, "text": "\n\n".join(t)} for n, t in sorted(slides.items())]


def thumbnail(path: Path, size: int = 1200) -> Path | None:
    """A PNG of the file's first page/slide from macOS Quick Look, cached by path + modified time."""
    if sys.platform != "darwin":
        return None
    try:
        key = hashlib.sha1(f"{path}|{path.stat().st_mtime}|{size}".encode()).hexdigest()
    except OSError:
        return None
    out_dir = THUMB_DIR / key
    out = out_dir / f"{path.name}.png"
    if not out.is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["qlmanage", "-t", "-s", str(size), "-o", str(out_dir), str(path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20, check=False)
    return out if out.is_file() else None


def open_file(path: Path, reveal: bool = False, app: Path | None = None) -> None:
    """Open with the default app (or `app`, macOS), or show it in Finder / Explorer / the file manager."""
    if sys.platform == "darwin":
        if reveal:
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["open", "-a", str(app), str(path)] if app else ["open", str(path)])
    elif sys.platform == "win32":
        if reveal:
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(path.parent if reveal else path)])


# Quick Look writes lengths without units (quirks-mode CSS); the app shows the page in standards mode.
_UNITLESS = re.compile(r"(?<![\w-])(top|left|right|bottom|width|height|font-size|margin(?:-[a-z]+)?|padding(?:-[a-z]+)?)"
                       r"(\s*:\s*)(-?\d*\.?\d+)(?=\s*[;}\"'])")
QUICKLOOK_FILE = re.compile(r"^[A-Za-z0-9_.-]+$")


def quicklook_dir(key: str) -> Path:
    return THUMB_DIR / "html" / key


def quicklook_html(path: Path) -> str | None:
    """Render the whole file to HTML with macOS Quick Look (every slide of a deck, with images) and return a cache
    key; the page is quicklook_dir(key) / "index.html". PDF images inside are converted to PNG so any web view
    can show them. None if Quick Look isn't available or fails."""
    if sys.platform != "darwin":
        return None
    try:
        key = hashlib.sha1(f"{path}|{path.stat().st_mtime}".encode()).hexdigest()
    except OSError:
        return None
    out = quicklook_dir(key)
    if (out / "index.html").is_file():
        return key
    tmp = out.with_name(key + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        subprocess.run(["qlmanage", "-p", "-o", str(tmp), str(path)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=60, check=False)
        bundle = next(tmp.glob("*.qlpreview"), None)
        if bundle is None or not (bundle / "Preview.html").is_file():
            return None
        html = (bundle / "Preview.html").read_text(encoding="utf-8", errors="replace")
        for pdf in bundle.glob("*.pdf"):
            png = pdf.with_suffix(".png")
            if not png.exists():
                subprocess.run(["sips", "-s", "format", "png", str(pdf), "--out", str(png)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, check=False)
            if png.exists():
                html = html.replace(f'"{pdf.name}"', f'"{png.name}"')
        html = _UNITLESS.sub(lambda m: f"{m[1]}{m[2]}{m[3]}px", html)
        (bundle / "index.html").write_text(html, encoding="utf-8")
        shutil.rmtree(out, ignore_errors=True)
        bundle.rename(out)
        return key
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- "Open With" (macOS LaunchServices)

_LS: Any = None


def _launch_services():
    """CoreFoundation + LaunchServices via ctypes (no PyObjC needed)."""
    global _LS
    if _LS is None:
        import ctypes
        from ctypes import c_bool, c_char_p, c_long, c_uint32, c_void_p

        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        cs = ctypes.CDLL("/System/Library/Frameworks/CoreServices.framework/CoreServices")
        for fn, res, args in (
            (cf.CFURLCreateFromFileSystemRepresentation, c_void_p, [c_void_p, c_char_p, c_long, c_bool]),
            (cf.CFURLGetFileSystemRepresentation, c_bool, [c_void_p, c_bool, c_char_p, c_long]),
            (cf.CFArrayGetCount, c_long, [c_void_p]),
            (cf.CFArrayGetValueAtIndex, c_void_p, [c_void_p, c_long]),
            (cf.CFRelease, None, [c_void_p]),
            (cs.LSCopyApplicationURLsForURL, c_void_p, [c_void_p, c_uint32]),
            (cs.LSCopyDefaultApplicationURLForURL, c_void_p, [c_void_p, c_uint32, c_void_p]),
        ):
            fn.restype, fn.argtypes = res, args
        _LS = (ctypes, cf, cs)
    return _LS


def _cfurl_path(ctypes, cf, url) -> str | None:
    buf = ctypes.create_string_buffer(4096)
    return buf.value.decode() if cf.CFURLGetFileSystemRepresentation(url, True, buf, len(buf)) else None


def _app_info(app: Path) -> dict[str, str] | None:
    import plistlib

    try:
        with (app / "Contents" / "Info.plist").open("rb") as f:
            info = plistlib.load(f)
    except (OSError, plistlib.InvalidFileException, ValueError):
        return None
    name = info.get("CFBundleDisplayName") or info.get("CFBundleName") or app.stem
    return {"path": str(app), "name": str(name)}


def is_app(app: Path) -> bool:
    return app.suffix == ".app" and (app / "Contents" / "Info.plist").is_file()


def open_with_apps(path: Path) -> dict[str, Any]:
    """{"default": app | None, "apps": [app, ...]} with app = {"path", "name"}: the apps macOS offers for this
    file in Finder's Open With menu, default first, then by name. Empty off macOS."""
    if sys.platform != "darwin":
        return {"default": None, "apps": []}
    ctypes, cf, cs = _launch_services()
    raw = os.fsencode(str(path))
    url = cf.CFURLCreateFromFileSystemRepresentation(None, raw, len(raw), False)
    if not url:
        return {"default": None, "apps": []}
    try:
        default_url = cs.LSCopyDefaultApplicationURLForURL(url, 0xFFFFFFFF, None)
        default = _cfurl_path(ctypes, cf, default_url) if default_url else None
        if default_url:
            cf.CFRelease(default_url)
        arr = cs.LSCopyApplicationURLsForURL(url, 0xFFFFFFFF)
        found = []
        if arr:
            found = [_cfurl_path(ctypes, cf, cf.CFArrayGetValueAtIndex(arr, i)) for i in range(cf.CFArrayGetCount(arr))]
            cf.CFRelease(arr)
    finally:
        cf.CFRelease(url)
    apps, seen = [], set()
    for p in ([default] if default else []) + found:
        # Skip apps buried in hidden folders (tool caches) and duplicate copies of the same app.
        if not p or any(part.startswith(".") for part in Path(p).parts):
            continue
        info = _app_info(Path(p))
        if info and info["name"] not in seen:
            seen.add(info["name"])
            apps.append(info)
    default_info = apps[0] if default and apps and apps[0]["path"] == default else None
    rest = sorted(apps[1:] if default_info else apps, key=lambda a: a["name"].lower())
    return {"default": default_info, "apps": rest}


def app_icon(app: Path, size: int = 64) -> Path | None:
    """The app's icon as a PNG (from its .icns), cached."""
    import plistlib

    if sys.platform != "darwin" or not is_app(app):
        return None
    try:
        with (app / "Contents" / "Info.plist").open("rb") as f:
            icon = plistlib.load(f).get("CFBundleIconFile")
        icns = app / "Contents" / "Resources" / (icon if str(icon).endswith(".icns") else f"{icon}.icns")
        key = hashlib.sha1(f"{app}|{icns.stat().st_mtime}|{size}".encode()).hexdigest()
    except (OSError, plistlib.InvalidFileException, ValueError, TypeError):
        return None
    out = THUMB_DIR / "icons" / f"{key}.png"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["sips", "-s", "format", "png", "-Z", str(size), str(icns), "--out", str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
    return out if out.is_file() else None
