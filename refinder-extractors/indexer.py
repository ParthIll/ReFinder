"""Create/update a single-row-per-file SQLite database with JSON chunks."""
from __future__ import annotations
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from .extractors import EXTRACTORS, extract

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    extension TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_ns INTEGER NOT NULL,
    sha256 TEXT,
    indexed_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('indexed','empty','error')),
    error TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    chunks_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_files_extension ON files(extension);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
"""


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def _inside(child: Path, root: Path) -> bool:
    return child == root or root in child.parents


def discover(roots: Sequence[Path], excluded: Sequence[Path], db_path: Path):
    """Walk roots; excluded directory trees are pruned, even if nested."""
    seen = set()
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        for directory, dirs, filenames in os.walk(root, followlinks=False):
            current = Path(directory).resolve()
            dirs[:] = sorted(name for name in dirs
                             if not _excluded((current / name).resolve(), excluded, db_path))
            if _excluded(current, excluded, db_path):
                dirs[:] = []
                continue
            for name in sorted(filenames):
                if name.startswith('~$'):
                    continue
                path = (current / name).resolve()
                if path in seen or _excluded(path, excluded, db_path):
                    continue
                if not path.is_file() or path.suffix.lower() not in EXTRACTORS:
                    continue
                seen.add(path)
                yield path


def _excluded(path: Path, excluded: Sequence[Path], db_path: Path) -> bool:
    return path == db_path or any(_inside(path, exclude) for exclude in excluded)


def index_folders(folders: Sequence[str | Path], excluded_folders: Sequence[str | Path],
                  db_path: str | Path, *, force: bool = False) -> dict[str, int]:
    roots = [Path(p).expanduser().resolve() for p in folders]
    excluded = [Path(p).expanduser().resolve() for p in excluded_folders]
    database = Path(db_path).expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    counts = {'indexed': 0, 'empty': 0, 'error': 0, 'skipped': 0}
    with sqlite3.connect(database) as connection:
        connection.executescript(SCHEMA)
        for path in discover(roots, excluded, database):
            try:
                stat = path.stat()
            except OSError:
                counts['error'] += 1
                continue
            existing = connection.execute(
                'SELECT size_bytes, modified_ns, status FROM files WHERE path = ?',
                (str(path),)).fetchone()
            if (not force and existing and existing[0] == stat.st_size
                    and existing[1] == stat.st_mtime_ns
                    and existing[2] in ('indexed', 'empty')):
                counts['skipped'] += 1
                continue
            try:
                digest = file_hash(path)
                # Reuse existing extraction when file bytes unchanged.
                if not force and existing and existing[2] in ('indexed', 'empty'):
                    prior_hash = connection.execute(
                        'SELECT sha256 FROM files WHERE path = ?', (str(path),)
                    ).fetchone()[0]
                    if prior_hash == digest:
                        connection.execute('UPDATE files SET size_bytes=?, modified_ns=? WHERE path=?',
                                           (stat.st_size, stat.st_mtime_ns, str(path)))
                        connection.commit()
                        counts['skipped'] += 1
                        continue
                chunks = extract(path)
                status = 'indexed' if chunks else 'empty'
                error = None
            except Exception as exc:
                # Explicit failure state; never silently mark corrupt/locked files indexed.
                digest = None
                chunks = []
                status = 'error'
                error = f'{type(exc).__name__}: {exc}'[:2000]
            connection.execute('''
                INSERT INTO files (path,filename,extension,size_bytes,modified_ns,sha256,
                                   indexed_at,status,error,chunk_count,chunks_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(path) DO UPDATE SET
                  filename=excluded.filename, extension=excluded.extension,
                  size_bytes=excluded.size_bytes, modified_ns=excluded.modified_ns,
                  sha256=excluded.sha256, indexed_at=excluded.indexed_at,
                  status=excluded.status,error=excluded.error,
                  chunk_count=excluded.chunk_count,chunks_json=excluded.chunks_json
            ''', (str(path), path.name, path.suffix.lower(), stat.st_size, stat.st_mtime_ns,
                  digest, datetime.now(timezone.utc).isoformat(), status, error,
                  len(chunks), json.dumps(chunks, ensure_ascii=False)))
            connection.commit()
            counts[status] += 1
    return counts
