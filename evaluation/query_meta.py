"""Reads file-type and date hints out of a search ("tax pdf from last year", "slides I downloaded this week").

The hints become small ranking boosts (see ChromaIndex.search); the search itself still uses the full wording,
since "during May" or "pitch deck" often describe the content too. `content` (the query minus hints) only decides
whether the query is nothing but hints ("pdfs from last week"), which then lists matching files. Only explicit cues count:
a bare year ("lease 2025") stays part of the content query, while "from 2025" or "in 2025" is a date.
"""

from __future__ import annotations

import calendar
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

DAY = 86400.0

# Phrases naming a file type -> extensions. Longest phrases are matched first.
TYPE_PHRASES: dict[str, tuple[str, ...]] = {
    "pdf": ("pdf",), "pdfs": ("pdf",),
    "word doc": ("docx",), "word docs": ("docx",), "word document": ("docx",), "word documents": ("docx",),
    "word file": ("docx",), "word files": ("docx",), "docx": ("docx",),
    "spreadsheet": ("xlsx", "csv"), "spreadsheets": ("xlsx", "csv"), "excel": ("xlsx",), "xlsx": ("xlsx",),
    "csv": ("csv",), "csvs": ("csv",),
    "slides": ("pptx",), "slide deck": ("pptx",), "deck": ("pptx",), "powerpoint": ("pptx",),
    "presentation": ("pptx",), "presentations": ("pptx",), "pptx": ("pptx",), "ppt": ("pptx",),
    "markdown": ("md",), "md file": ("md",), "md files": ("md",),
    "text file": ("txt",), "text files": ("txt",), "txt": ("txt",),
    "email": ("eml",), "emails": ("eml",), "eml": ("eml",),
}
TYPE_LABELS = {"pdf": "PDF", "docx": "Word", "xlsx": "Excel", "csv": "CSV", "pptx": "Slides", "md": "Markdown",
               "txt": "Text", "eml": "Email"}
ADDED_VERBS = r"(?:downloaded|added|saved|got|received)"
MODIFIED_VERBS = r"(?:edited|modified|updated|changed|worked on|wrote)"
RECENT = re.compile(r"\b(?:recent|recently|latest|newest)\b", re.I)
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m} | \
         {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
# Words left dangling at the ends once the hints are gone ("tax pdf from last year" -> "tax from").
EDGE_FILLER = {"file", "files", "document", "documents", "doc", "docs", "from", "in", "during", "since", "on", "that",
               "which", "i", "my", "the", "a", "an", "of", "this", "was", "were", "about", "with", "for", "me", "show",
               "find", "all", "any", "some", "and", "or", "to"}


@dataclass
class QueryMeta:
    content: str  # the query without the hints
    types: set[str] = field(default_factory=set)
    after: float | None = None  # Unix seconds
    before: float | None = None
    date_field: str = "any"  # "modified", "added" or "any" (either date counts)
    recent: bool = False
    labels: list[str] = field(default_factory=list)  # what was understood, for the UI

    def active(self) -> bool:
        return bool(self.types or self.after is not None or self.before is not None or self.recent)


def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _month_range(year: int, month: int) -> tuple[float, float]:
    start = datetime(year, month, 1)
    end = datetime(year + (month == 12), month % 12 + 1, 1)
    return start.timestamp(), end.timestamp() - 1


def _date_hint(q: str, now: datetime) -> tuple[tuple[float | None, float | None], str, re.Match] | None:
    """The first date phrase in q: ((after, before), label, match)."""
    today = _start_of_day(now)
    relative = [
        (r"\btoday\b", lambda m: ((today.timestamp(), None), "today")),
        (r"\byesterday\b", lambda m: (((today - timedelta(days=1)).timestamp(), today.timestamp() - 1), "yesterday")),
        (r"\b(?:this|past|last) week\b", lambda m: (((now - timedelta(days=7)).timestamp(), None), "past week")),
        (r"\bthis month\b", lambda m: ((today.replace(day=1).timestamp(), None), "this month")),
        (r"\blast month\b", lambda m: (_month_range(*(lambda d: (d.year, d.month))(today.replace(day=1) - timedelta(days=1))), "last month")),
        (r"\bpast month\b", lambda m: (((now - timedelta(days=30)).timestamp(), None), "past month")),
        (r"\bthis year\b", lambda m: ((datetime(now.year, 1, 1).timestamp(), None), f"{now.year}")),
        (r"\blast year\b", lambda m: ((datetime(now.year - 1, 1, 1).timestamp(), datetime(now.year, 1, 1).timestamp() - 1), f"{now.year - 1}")),
        (r"\bpast year\b", lambda m: (((now - timedelta(days=365)).timestamp(), None), "past year")),
        (r"\b(?:last|past) (\d{1,3}) (day|week|month)s?\b",
         lambda m: (((now - timedelta(days=int(m[1]) * {"day": 1, "week": 7, "month": 30}[m[2].lower()])).timestamp(), None),
                    f"past {m[1]} {m[2].lower()}s")),
    ]
    found = []
    for pattern, make in relative:
        if (m := re.search(pattern, q, re.I)):
            found.append((m, make(m)))
    # "in/from/during/since March [2025]" and "in/from 2024": only with a preposition, so "lease 2025" stays content.
    month_names = "|".join(sorted(MONTHS, key=len, reverse=True))
    if (m := re.search(rf"\b(in|from|during|since)\s+({month_names})\.?(?:\s+(\d{{4}}))?\b", q, re.I)):
        month = MONTHS[m[2].lower()]
        year = int(m[3]) if m[3] else (now.year if month <= now.month else now.year - 1)
        lo, hi = _month_range(year, month)
        found.append((m, ((lo, None if m[1].lower() == "since" else hi), f"{calendar.month_abbr[month]} {year}")))
    elif (m := re.search(r"\b(in|from|during|since)\s+((?:19|20)\d{2})\b", q, re.I)):
        year = int(m[2])
        lo, hi = datetime(year, 1, 1).timestamp(), datetime(year + 1, 1, 1).timestamp() - 1
        found.append((m, ((lo, None if m[1].lower() == "since" else hi), str(year))))
    if not found:
        return None
    m, (bounds, label) = min(found, key=lambda x: x[0].start())
    return bounds, label, m


def parse(query: str, now: float | None = None) -> QueryMeta:
    now_dt = datetime.fromtimestamp(now if now is not None else time.time())
    q = query
    meta = QueryMeta(content=query)
    # File types (whole words; longest phrases first so "word document" wins over "document").
    for phrase in sorted(TYPE_PHRASES, key=len, reverse=True):
        pattern = rf"(?<![\w.])\.?{re.escape(phrase)}(?!\w)"
        if re.search(pattern, q, re.I):
            meta.types.update(TYPE_PHRASES[phrase])
            q = re.sub(pattern, " ", q, flags=re.I)
    if meta.types:
        meta.labels.append(" or ".join(sorted({TYPE_LABELS.get(t, t.upper()) for t in meta.types})))
    # Which date: "downloaded/added ..." -> date added, "edited/modified ..." -> date modified.
    if re.search(rf"\b{ADDED_VERBS}\b", q, re.I):
        meta.date_field = "added"
    elif re.search(rf"\b{MODIFIED_VERBS}\b", q, re.I):
        meta.date_field = "modified"
    hint = _date_hint(q, now_dt)
    if hint:
        (meta.after, meta.before), label, m = hint
        verb = {"added": "added", "modified": "modified"}.get(meta.date_field, "from")
        meta.labels.append(f"{verb} {label}")
        q = q[:m.start()] + " " + q[m.end():]
    if RECENT.search(q):
        meta.recent = True
        meta.labels.append("recent first")
        q = RECENT.sub(" ", q)
    if meta.active():
        q = re.sub(rf"\b(?:{ADDED_VERBS}|{MODIFIED_VERBS})\b", " ", q, flags=re.I)
        words = q.split()
        while words and words[0].lower().strip("',.") in EDGE_FILLER:
            words.pop(0)
        while words and words[-1].lower().strip("',.") in EDGE_FILLER:
            words.pop()
        meta.content = " ".join(words)
    return meta
