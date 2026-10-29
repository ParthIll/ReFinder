# ReFinder extraction MVP

Builds a **single SQLite `.db` file** with **one row per file**. Chunks are a JSON array in `chunks_json`. No vector index, embedding model, FTS, OCR, or continuous watcher is included in this stage.

## Setup

Python 3.10+:

``` bash
pip install -e .
```

## Run

``` bash
refinder-index --folder "/path/to/Documents" --folder "/path/to/Desktop" \
  --exclude "/path/to/Documents/private" --db ./refinder.db
```

Or without installing the CLI wrapper: `python -m refinder.cli --folder ./samples --db ./refinder.db`.

Repeat a folder/exclusion argument to supply several. Exclusion paths include their descendants. Run again to skip unchanged files (based on size + nanosecond mtime and, when changed, SHA-256). Use `--force` to reprocess everything. The tool ignores unsupported extensions and Office `~$` temporary files. Folder symlinks aren't followed. The DB file itself is ignored if located in a scanned folder.

## Schema

`files(path PRIMARY KEY, filename, extension, size_bytes, modified_ns, sha256, indexed_at, status, error, chunk_count, chunks_json)`

`chunks_json`: ordered **JSON list of strings**, e.g. `["First paragraph", "Table 1:\\nA | B"]`. To get chunks:

``` python
import sqlite3, json
with sqlite3.connect('refinder.db') as connection:
    for path, value in connection.execute('SELECT path, chunks_json FROM files WHERE status="indexed"'):
        for chunk in json.loads(value):
            print(path, chunk)
```

`status`: `indexed`, `empty` (no extractable text), or `error` (see `error` column). Errors from unreadable individual documents are recorded; the scan continues.

## Chunking/normalization

- `.xlsx`: nonoverlapping 3×3 tiles in left-to-right/top-to-bottom tile order, **column-first within each tile**, labeled `Sheet: ...; Range: ...`, then `A1: value` etc. Formula expressions ignored; **cached formula results may still be included** (`data_only=True`). A future version can explicitly omit all formula cells or index formula expressions. This is not semantic understanding of layout.
- `.pptx`: **one chunk per nonempty slide**. Slide prefix retained in chunk text. Grouped shapes and text tables included. Speaker notes are v2.
- `.docx`: one chunk per nonempty top-level paragraph, one chunk per table, in body order. Not headers/footers, nested table structure, or revisions.
- `.pdf`: page-labeled paragraph-like chunks; text-layer only. PDF layout may require heuristic boundaries and scanned PDFs produce zero chunks (v2 OCR).
- `.txt`, `.md`, `.markdown`: blank-line-separated paragraphs. UTF-8, UTF-8 BOM, UTF-16 BOM, and Windows-1252 fallback.
- `.csv`, `.tsv`: one chunk per nonempty row, with headers attached to values.

Whitespace and Unicode are conservatively normalized. Real-world Office content can be larger than the sentence-transformer token limit; a later chunk splitter should subdivide oversized paragraphs/tables/slides before embedding.

## Scope notes

- `.doc`, `.xls`, `.ppt`, encrypted documents, images, scanned PDFs, charts/SmartArt text, and OCR are out of scope.
- Unchanged files are skipped. Records for moved or deleted files **are not automatically pruned** in this MVP. Rebuild the DB or implement reconciliation in v2.
- SQLite contains raw document text. Treat `refinder.db` as sensitive and store it in a user-controlled location.
- For a large spreadsheet, 3×3 tiles produce many tiny chunks; benchmark retrieval and storage before adopting this as final.
