"""Watches the indexed folders and reports changed paths in batches, so the index can follow edits, new files,
renames and deletions without re-indexing whole folders. Uses watchdog (FSEvents on macOS, inotify on Linux,
ReadDirectoryChangesW on Windows)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from refinder.extract import SUPPORTED_SUFFIXES

QUIET_SECONDS = 2.0  # wait for a burst of changes (a save, an unzip) to settle before updating


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: "FolderWatcher"):
        self.watcher = watcher

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type in ("opened", "closed_no_write"):
            return
        for raw in (event.src_path, getattr(event, "dest_path", "") or ""):
            path = raw.decode() if isinstance(raw, bytes) else raw
            if not path:
                continue
            if event.is_directory:
                # A folder modified event only means something inside changed (reported separately); a folder
                # created, deleted or moved means everything under it did.
                if event.event_type != "modified":
                    self.watcher.note(path, folder=True)
            elif Path(path).suffix.lower() in SUPPORTED_SUFFIXES:
                self.watcher.note(path, folder=False)


class FolderWatcher:
    """Calls `on_changes(files, folders)` with sets of changed file paths and folder paths once changes have
    been quiet for QUIET_SECONDS. If the callback can't run yet (returns False), the batch is retried."""

    def __init__(self, on_changes: Callable[[set[str], set[str]], bool]):
        self.on_changes = on_changes
        self.observer: Observer | None = None
        self.lock = threading.Lock()
        self.files: set[str] = set()
        self.folders: set[str] = set()
        self.last_event = 0.0
        self.roots: list[str] = []
        threading.Thread(target=self._flush_loop, daemon=True, name="refinder-watch-flush").start()

    def watch(self, roots: list[str]) -> None:
        """(Re)start watching exactly these folders."""
        if sorted(roots) == sorted(self.roots) and self.observer is not None:
            return
        if self.observer is not None:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.observer = None
        self.roots = list(roots)
        if not roots:
            return
        observer = Observer()
        handler = _Handler(self)
        for root in roots:
            if Path(root).is_dir():
                observer.schedule(handler, root, recursive=True)
        observer.daemon = True
        observer.start()
        self.observer = observer

    def note(self, path: str, folder: bool) -> None:
        with self.lock:
            (self.folders if folder else self.files).add(path)
            self.last_event = time.monotonic()

    def _flush_loop(self) -> None:
        while True:
            time.sleep(0.5)
            with self.lock:
                if not (self.files or self.folders) or time.monotonic() - self.last_event < QUIET_SECONDS:
                    continue
                files, folders = self.files, self.folders
                self.files, self.folders = set(), set()
            try:
                done = self.on_changes(files, folders)
            except Exception:
                done = True  # a broken batch shouldn't wedge the watcher; the next startup scan catches up
            if not done:
                with self.lock:
                    self.files |= files
                    self.folders |= folders
                    self.last_event = time.monotonic()
