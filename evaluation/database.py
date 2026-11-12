"""Hybrid search index for the ReFinder API: BM25 keyword search over whole files plus dense vectors in ChromaDB.

Indexing happens in two stages so a folder is searchable almost immediately:
  1. `build_keyword` builds a BM25 index over every file's full text (seconds, even for thousands of files).
  2. `embed` embeds a capped sample of each file's paragraph chunks and adds them to Chroma in batches, usually
     from a background thread. Search uses whatever vectors exist so far.
Ranking puts literal matches first: a file scores
    W_NAME   * (share of the query's words in its filename) ** NAME_POW   (fuzzy: typos, plurals, compounds;
               folder names count half; power 1 for queries of <= SHORT_QUERY_WORDS words)
  + W_PHRASE * 1 if the query (or every "quoted part") appears word for word in the file
  + W_KW     / (RRF_K + BM25 rank)       BM25 over full text, with typo'd query words mapped to close index words
  + W_DENSE  / (RRF_K + embedding rank)  file rank = its best chunk
so a filename that covers the query, or an exact phrase, outranks anything found only by meaning.
Weights were picked on the team corpus, OrgSmith, FinanceBench and the sample corpus, with natural-language
questions plus generated filename / filename-with-typo / exact-phrase queries.

This is the setup that indexed fastest at equal or better accuracy in the benchmark (evaluation/custom.py with
--max-chunks 20 --hybrid): ~15x less embedding than embedding every chunk.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import re
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import bm25s
import chromadb
import numpy as np
import Stemmer
from bm25s.stopwords import STOPWORDS_EN
from chromadb.config import Settings
from rapidfuzz import fuzz, process

if TYPE_CHECKING:
    from refinder.embedder import Embedder


# ---------------------------------------------------------------- file metadata (for filters)

class _AttrList(ctypes.Structure):
    _fields_ = [("bitmapcount", ctypes.c_ushort), ("reserved", ctypes.c_uint16), ("commonattr", ctypes.c_uint32),
                ("volattr", ctypes.c_uint32), ("dirattr", ctypes.c_uint32), ("fileattr", ctypes.c_uint32),
                ("forkattr", ctypes.c_uint32)]


class _AddedTime(ctypes.Structure):
    _pack_ = 4  # getattrlist packs attributes on 4-byte boundaries
    _fields_ = [("length", ctypes.c_uint32), ("sec", ctypes.c_int64), ("nsec", ctypes.c_int64)]


_LIBC = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True) if sys.platform == "darwin" else None
_ATTR_CMN_ADDEDTIME = 0x10000000


def _date_added(path: str) -> float | None:
    """macOS "Date Added" (when the file arrived in its folder, e.g. downloaded), as Finder shows it."""
    if _LIBC is None:
        return None
    attrs, buf = _AttrList(5, 0, _ATTR_CMN_ADDEDTIME, 0, 0, 0, 0), _AddedTime()
    if _LIBC.getattrlist(os.fsencode(path), ctypes.byref(attrs), ctypes.byref(buf), ctypes.sizeof(buf), 0) != 0:
        return None
    return float(buf.sec) if buf.length >= ctypes.sizeof(buf) else None


def file_meta(path: str) -> tuple[float, float, int]:
    """(modified, added, size in bytes). Added falls back to creation time, then modified; missing files are 0."""
    try:
        st = os.stat(path)
    except OSError:
        return 0.0, 0.0, 0
    added = _date_added(path) or getattr(st, "st_birthtime", None) or st.st_mtime
    return st.st_mtime, float(added), st.st_size


_TYPE_ALIASES = {"markdown": "md", "htm": "html", "jpeg": "jpg"}


def file_type(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".") or "other"
    return _TYPE_ALIASES.get(ext, ext)


@dataclass
class Filters:
    """Search filters; None means no limit. Times are Unix seconds, sizes bytes, folder a case-insensitive
    substring of the file's folder path relative to the indexed root."""
    types: set[str] | None = None
    modified_after: float | None = None
    modified_before: float | None = None
    added_after: float | None = None
    added_before: float | None = None
    min_size: int | None = None
    max_size: int | None = None
    folder: str | None = None

    def active(self) -> bool:
        return any(v not in (None, "", set()) for v in vars(self).values())

COLLECTION_NAME = "refinder"
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
MANIFEST_NAME = "manifest.json"
EMBED_BATCH = 256  # chunks per encode + Chroma add; also how often embedding checks for cancellation
MAX_CHUNKS_PER_FILE = 20  # long files: first 10 chunks + 10 spread over the rest; BM25 still sees the full text
RRF_K = 60  # reciprocal rank fusion constant; a file scores sum(1 / (RRF_K + rank)) over the lists it appears in
DENSE_CANDIDATES = 400  # nearest chunks fetched from Chroma per query
SNIPPET_CHARS = 300
PASSAGE_CHARS = 3000  # matched text sent for highlighting in the preview
# Ranking weights (see module docstring). One RRF list's top rank is worth 1/61 ~ 0.016, so W_NAME = 0.05 lets a
# full filename match beat a file ranked first by both BM25 and embeddings.
W_NAME = 0.05
W_PHRASE = 0.03
W_KW = 1.0
W_DENSE = 1.0
# Filename share is raised to this power: a name covering the whole query keeps the full boost, one sharing a
# couple of words with a long question (a client name, say) barely moves it.
NAME_POW = 2
SHORT_QUERY_WORDS = 3  # queries this short are usually name lookups: their filename share counts linearly
FUZZY_MIN_LEN = 4  # shorter words only match exactly ("tax" must not fuzzy-match "tex")
FUZZY_CUTOFF = 85  # rapidfuzz ratio, 0-100
PHRASE_CANDIDATES = 300  # files (by BM25) checked for an exact phrase
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(STOPWORDS_EN)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _name_words(path: str) -> tuple[list[str], list[str]]:
    """(filename words, folder words); camelCase and digit runs split, e.g. 'TaxNotes2025.txt' -> tax, notes, 2025."""
    p = Path(path)
    split = lambda t: _words(re.sub(r"([a-z])([A-Z])|([A-Za-z])(\d)|(\d)([A-Za-z])", r"\1\3\5 \2\4\6", t))
    # Keep unsplit words too, so "w4" still matches form_w4 after the split into "w", "4".
    both = lambda t: list(dict.fromkeys(split(t) + _words(t)))
    return both(p.stem), [w for part in p.parent.parts for w in both(part)]


class Cancelled(Exception):
    """A newer index replaced the one being embedded."""


class ChromaIndex:
    def __init__(self, persist_dir: Path | None = None):
        self.persist_dir = Path(persist_dir) if persist_dir else DEFAULT_PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.persist_dir / MANIFEST_NAME
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"})
        self.stemmer = Stemmer.Stemmer("english")
        self.lock = threading.Lock()  # guards swapping the index; searches read a consistent snapshot
        self.generation = 0  # bumped by every build_keyword, so a stale background embed can stop
        self.manifest: dict[str, Any] | None = None
        self.texts: list[str] = []
        self.bm25: bm25s.BM25 | None = None
        self.keyword: dict[str, Any] = {}
        self.embedded_chunks = 0
        self.total_chunks = 0
        self.embedded_docs = 0  # files [0, embedded_docs) have all their chunks in Chroma

    # ------------------------------------------------------------ state

    def ready(self) -> bool:
        return self.bm25 is not None

    def count(self) -> int:
        return int(self.collection.count())

    def model_name(self) -> str | None:
        return self.manifest.get("model_name") if self.manifest else None

    def embedding_complete(self) -> bool:
        return bool(self.manifest and self.embedded_docs >= len(self.manifest["paths"]))

    def roots(self) -> list[str]:
        return list(self.manifest["roots"]) if self.manifest else []

    def _write_manifest(self) -> None:
        self.manifest["embedding_complete"] = self.embedding_complete()
        self.manifest["embedded_docs"] = self.embedded_docs
        self.manifest["total_chunks"] = self.total_chunks
        self.manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        """Restore the last index from disk (BM25 is rebuilt from the saved text; vectors live in Chroma).
        Returns False if there is nothing usable to restore."""
        if not self.manifest_path.is_file():
            return False
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if "pieces" not in data:  # manifest from the old window-chunk index: rebuild instead of mixing chunkings
            return False
        if "roots" not in data:  # single-folder manifest with relative paths
            root = data.pop("corpus_root")
            data["roots"], data["root_of"] = [root], [0] * len(data["paths"])
            data["paths"] = [str(Path(root) / p) for p in data["paths"]]
            data["embedded_docs"] = len(data["paths"]) if data.get("embedding_complete") else 0
        self._set_keyword(data)
        self.embedded_docs = int(data.get("embedded_docs", 0))
        self.embedded_chunks = self.count()
        self.total_chunks = data.get("total_chunks", self.embedded_chunks)
        return True

    # ------------------------------------------------------------ indexing

    def _tokenize(self, texts: list[str], ids: bool = True):
        return bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, return_ids=ids, show_progress=False)

    def _set_keyword(self, manifest: dict[str, Any]) -> None:
        """Build BM25 and the filename/filter tables over every file in the manifest (paths are absolute)."""
        texts = ["\n\n".join(ps) for ps in manifest["pieces"]]
        tokenized = self._tokenize(texts)
        bm25 = bm25s.BM25()
        bm25.index(tokenized, show_progress=False)
        roots = manifest["roots"]
        # Folder words come from the path below the indexed folder, so e.g. "Users" never counts as a match.
        rel = [os.path.relpath(p, roots[r]) for p, r in zip(manifest["paths"], manifest["root_of"])]
        names = [_name_words(p) for p in rel]
        keyword = {
            "vocab": [w for w in tokenized.vocab if w],  # stemmed index words, for typo correction
            "file_words": [set(f) for f, _ in names],
            "folder_words": [set(d) for _, d in names],
            "name_vocab": sorted({w for f, d in names for w in f + d}),
            "flat": [" ".join(_words(t)) for t in texts],  # for exact-phrase checks
        }
        # Dates and sizes are read fresh (cheap: a few ms per thousand files) rather than stored in the manifest.
        meta = np.array([file_meta(p) for p in manifest["paths"]], dtype=np.float64).reshape(-1, 3)
        keyword["modified"], keyword["added"], keyword["size"] = meta[:, 0], meta[:, 1], meta[:, 2]
        keyword["type"] = np.array([file_type(p) for p in manifest["paths"]])
        keyword["folder"] = [str(Path(p).parent).lower() for p in manifest["paths"]]
        # Unstemmed words in file text with their counts, for "Did you mean" suggestions.
        freq = Counter(w for f in keyword["flat"] for w in f.split() if len(w) >= 3 and not w.isdigit())
        keyword["word_freq"] = freq
        keyword["words"] = list(freq)
        with self.lock:
            self.manifest, self.texts, self.bm25, self.keyword = manifest, texts, bm25, keyword

    def replace(self, root: Path, paths: list[str], pieces: list[list[str]], model_name: str) -> int:
        """Stage 1: replace the whole index with a keyword-only one over these files (absolute paths) from one
        folder. Returns the new generation."""
        with self.lock:
            self.generation += 1
            generation = self.generation
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME, embedding_function=None,
                metadata={"hnsw:space": "cosine", "model_name": model_name})
            self.embedded_chunks = self.total_chunks = self.embedded_docs = 0
        self._set_keyword({"roots": [str(root)], "root_of": [0] * len(paths), "model_name": model_name,
                           "paths": paths, "pieces": pieces})
        self._write_manifest()
        return generation

    def add(self, root: Path, paths: list[str], pieces: list[list[str]]) -> int:
        """Stage 1 for an extra folder: append these files (absolute paths, none already indexed) and rebuild the
        keyword index over everything. Existing files keep their ids and vectors; `embed` then only embeds the new
        files. A folder containing already-indexed folders absorbs them. Returns the new generation."""
        old = self.manifest
        root_s = str(root)
        inside = lambda child, parent: child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)
        # Indexed folders inside the new one merge into it; the new one inside an indexed folder adds no root.
        parent = next((i for i, r in enumerate(old["roots"]) if inside(root_s, r)), None)
        if parent is None:
            roots = [r for r in old["roots"] if not inside(r, root_s)] + [root_s]
            remap = {i: (roots.index(r) if r in roots else len(roots) - 1) for i, r in enumerate(old["roots"])}
            root_of = [remap[r] for r in old["root_of"]] + [len(roots) - 1] * len(paths)
        else:
            roots, root_of = list(old["roots"]), list(old["root_of"]) + [parent] * len(paths)
        manifest = {**old, "roots": roots, "root_of": root_of, "paths": old["paths"] + paths,
                    "pieces": old["pieces"] + pieces}
        with self.lock:
            self.generation += 1  # a running embed stops; the next one resumes from embedded_docs
            generation = self.generation
        self._set_keyword(manifest)
        self._write_manifest()
        return generation

    def embed(self, embedder: "Embedder", generation: int, model_lock: threading.Lock,
              on_progress: Callable[[int, int], None] | None = None) -> None:
        """Stage 2: embed capped paragraph chunks of files not embedded yet (ids >= embedded_docs) into Chroma.
        Raises Cancelled if the index changed meanwhile. `model_lock` serializes model use with query encoding."""
        from evaluation.common import chunk_corpus

        first = self.embedded_docs
        with self.lock:  # drop chunks of a file that was half done when a previous embed stopped
            self.collection.delete(where={"doc_id": {"$gte": first}})
            done_chunks = self.count()
        pieces = self.manifest["pieces"][first:]
        texts = self.texts[first:]
        chunk_texts, owner, _ = chunk_corpus(embedder, texts, pieces, "paragraph", max_chunks=MAX_CHUNKS_PER_FILE)
        owner = np.asarray(owner, dtype=int) + first
        ordinal = np.zeros(len(owner), dtype=int)  # chunk position within its file (after capping)
        for i in range(1, len(owner)):
            ordinal[i] = ordinal[i - 1] + 1 if owner[i] == owner[i - 1] else 0
        n_docs = len(self.manifest["paths"])
        self.embedded_chunks, self.total_chunks = done_chunks, done_chunks + len(chunk_texts)
        paths = self.manifest["paths"]
        for start in range(0, len(chunk_texts), EMBED_BATCH):
            if generation != self.generation:
                raise Cancelled
            batch = chunk_texts[start:start + EMBED_BATCH]
            with model_lock:
                vecs = embedder.encode(batch, batch_size=64)
            end = start + len(batch)
            with self.lock:
                if generation != self.generation:
                    raise Cancelled
                self.collection.add(
                    ids=[f"{owner[i]}:{ordinal[i]}" for i in range(start, end)],
                    embeddings=vecs,
                    documents=batch,
                    metadatas=[{"path": paths[owner[i]], "doc_id": int(owner[i]), "chunk_index": int(ordinal[i])}
                               for i in range(start, end)],
                )
                self.embedded_chunks = done_chunks + end
                # Chunks are in file order, so every file before the next chunk's owner is complete.
                self.embedded_docs = int(owner[end]) if end < len(owner) else n_docs
            if on_progress:
                on_progress(self.embedded_chunks, self.total_chunks)
        with self.lock:
            if generation != self.generation:
                raise Cancelled
            self.embedded_docs = n_docs
            self._write_manifest()

    def file_text(self, path: str) -> str | None:
        try:
            return self.texts[self.manifest["paths"].index(path)]
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------ search

    def _close(self, word: str, vocab: list[str]) -> list[str]:
        if len(word) < FUZZY_MIN_LEN:
            return []
        return [m for m, _, _ in process.extract(word, vocab, scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF, limit=3)]

    def file_types(self) -> dict[str, int]:
        """Indexed file count per type (extension), most common first."""
        if not self.ready():
            return {}
        return dict(Counter(self.keyword["type"].tolist()).most_common())

    def filter_mask(self, f: Filters | None) -> np.ndarray | None:
        """Boolean mask of files passing the filters, or None when no filter is set."""
        if f is None or not f.active():
            return None
        kw = self.keyword
        mask = np.ones(len(self.manifest["paths"]), dtype=bool)
        if f.types:
            mask &= np.isin(kw["type"], list(f.types))
        for col, lo, hi in (("modified", f.modified_after, f.modified_before), ("added", f.added_after, f.added_before),
                            ("size", f.min_size, f.max_size)):
            if lo is not None:
                mask &= kw[col] >= lo
            if hi is not None:
                mask &= kw[col] <= hi
        if f.folder:
            needle = f.folder.lower().strip("/")
            mask &= np.array([needle in d for d in kw["folder"]], dtype=bool)
        return mask

    def _result_meta(self, f: int) -> dict[str, Any]:
        kw = self.keyword
        return {"modified": float(kw["modified"][f]), "added": float(kw["added"][f]), "size": int(kw["size"][f]),
                "type": str(kw["type"][f])}

    def _keyword_ranking(self, query: str) -> tuple[list[int], list[str]]:
        """Files with any keyword hit, best BM25 first. Query words missing from the index (typos) are replaced
        by the closest index words."""
        tokens = self._tokenize([query], ids=False)[0]
        known = self.bm25.vocab_dict
        expanded = []
        for t in tokens:
            expanded += [t] if t in known else self._close(t, self.keyword["vocab"])
        if not expanded:
            return [], tokens
        scores = self.bm25.get_scores(expanded)
        hits = np.flatnonzero(scores > 0)
        return hits[np.argsort(-scores[hits], kind="stable")].tolist(), expanded

    def _name_scores(self, words: list[str]) -> np.ndarray:
        """Per file: share of query words found in its filename (1 each) or only its folder names (0.5 each)."""
        kw = self.keyword
        n = len(self.manifest["paths"])
        if not words:
            return np.zeros(n)
        hits = np.zeros((len(words), n))
        for i, w in enumerate(words):
            # A query word matches a name word if equal, same apart from a plural "s", a close spelling, or the
            # start of it ("recurs" -> "recursion"). Matching inside a word is too loose ("lease" in "freelance").
            forms = {w, w.rstrip("s")} | set(self._close(w, kw["name_vocab"]))
            prefix = len(w) >= FUZZY_MIN_LEN
            for f in range(n):
                names = kw["file_words"][f]
                if forms & names or (prefix and any(x.startswith(w) for x in names)):
                    hits[i, f] = 1.0
                elif forms & kw["folder_words"][f]:
                    hits[i, f] = 0.5
        # Adjacent query words written as one name word: "pay stub" -> "paystub".
        for i in range(len(words) - 1):
            joined = words[i] + words[i + 1]
            for f in range(n):
                if joined in kw["file_words"][f]:
                    hits[i, f] = hits[i + 1, f] = 1.0
        return hits.sum(axis=0) / len(words)

    def _phrase_hits(self, query: str, candidates: list[int]) -> set[int]:
        """Files containing the query word for word (or every "quoted part" of it). Needs 2+ words."""
        quoted = re.findall(r'"([^"]+)"', query)
        phrases = [" ".join(_words(q)) for q in quoted] if quoted else [" ".join(_words(query))]
        phrases = [p for p in phrases if " " in p]
        if not phrases:
            return set()
        flat = self.keyword["flat"]
        return {f for f in candidates if all(p in flat[f] for p in phrases)}

    def _dense_ranking(self, query_vec: np.ndarray | None,
                       allowed: np.ndarray | None = None) -> list[tuple[int, str, int]]:
        """(doc_id, chunk text, chunk index) of each file's best chunk, best file first. With `allowed` (a file
        mask), only those files' chunks are searched, so a narrow filter still gets meaning-based matches."""
        n = self.count()
        if query_vec is None or n == 0:
            return []
        where = None
        if allowed is not None:
            ids = np.flatnonzero(allowed).tolist()
            if not ids:
                return []
            where = {"doc_id": {"$in": ids}}
        raw = self.collection.query(query_embeddings=[np.asarray(query_vec, dtype=np.float32)],
                                    n_results=min(n, DENSE_CANDIDATES), where=where,
                                    include=["documents", "metadatas"])
        best, seen = [], set()
        for doc, meta in zip(raw["documents"][0], raw["metadatas"][0]):  # already sorted by distance
            if meta and meta["doc_id"] not in seen:
                seen.add(meta["doc_id"])
                best.append((int(meta["doc_id"]), doc or "", int(meta["chunk_index"])))
        return best

    def signals(self, query: str, query_vec: np.ndarray | None, allowed: np.ndarray | None = None) -> dict[str, Any]:
        """Everything ranking needs for one query (split out so evaluation can try other weights)."""
        words = [w for w in _words(query) if w not in _STOP]
        keyword, tokens = self._keyword_ranking(query)
        return {
            "n_words": len(words),
            "keyword": keyword,
            "tokens": tokens,
            "name": self._name_scores(words),
            "phrase": self._phrase_hits(query, keyword[:PHRASE_CANDIDATES]),
            "dense": self._dense_ranking(query_vec, allowed),
        }

    @staticmethod
    def fuse(sig: dict[str, Any], n_files: int, w_name=W_NAME, w_phrase=W_PHRASE, w_kw=W_KW,
             w_dense=W_DENSE, name_pow=NAME_POW) -> np.ndarray:
        pw = 1 if sig["n_words"] <= SHORT_QUERY_WORDS else name_pow
        score = w_name * sig["name"] ** pw
        for f in sig["phrase"]:
            score[f] += w_phrase
        for rank, f in enumerate(sig["keyword"], 1):
            score[f] += w_kw / (RRF_K + rank)
        for rank, (f, _, _) in enumerate(sig["dense"], 1):
            score[f] += w_dense / (RRF_K + rank)
        return score

    def _keyword_snippet(self, doc_id: int, tokens: list[str], phrase: str | None) -> str:
        """The paragraph holding the exact phrase, else the one sharing the most (stemmed) query words."""
        pieces = self.manifest["pieces"][doc_id]
        if not pieces:
            return ""
        if phrase:
            for p in pieces:
                if phrase in " ".join(_words(p)):
                    return p
        # Substring match on stems is crude but fast, and only picks which paragraph to show.
        return max(pieces, key=lambda p: sum(s in p.lower() for s in set(tokens)))

    def suggest(self, query: str) -> str | None:
        """The query with each word that no file contains replaced by the closest word that one does (more
        common words win close calls), or None if every word is found or nothing is close."""
        kw, changed, out = self.keyword, False, []
        for w in re.findall(r"\S+", query):
            core = w.lower().strip(".,;:!?\"'()")
            stems = self._tokenize([core], ids=False)[0] if core else []
            if len(core) < 3 or core in _STOP or not stems or all(t in self.bm25.vocab_dict for t in stems):
                out.append(w)
                continue
            cands = process.extract(core, kw["words"], scorer=fuzz.ratio, score_cutoff=75, limit=5)
            if not cands:
                out.append(w)
                continue
            best = max(cands, key=lambda c: (round(c[1] / 5), kw["word_freq"][c[0]]))[0]
            out.append(best)
            changed = True
        return " ".join(out) if changed else None

    def browse(self, filters: Filters, k: int = 20) -> list[dict[str, Any]]:
        """Files passing the filters, most recently modified first (for a filter-only search with no query)."""
        mask = self.filter_mask(filters)
        if mask is None:
            return []
        ids = np.flatnonzero(mask)
        ids = ids[np.argsort(-self.keyword["modified"][ids], kind="stable")][:k]
        out = []
        for f in ids.tolist():
            pieces = self.manifest["pieces"][f]
            out.append({"path": self.manifest["paths"][f], "score": 1.0, "chunk_id": None,
                        "snippet": " ".join((pieces[0] if pieces else "").split())[:SNIPPET_CHARS],
                        **self._result_meta(f)})
        return out

    def keyword_search(self, query: str, k: int = 5, filters: Filters | None = None) -> list[dict[str, Any]]:
        """Pure BM25 over file text: only files containing the query words (stemmed), no typo correction,
        filename boost or meaning. Score is BM25 relative to the top result."""
        if not self.ready():
            return []
        tokens = [t for t in self._tokenize([query], ids=False)[0] if t in self.bm25.vocab_dict]
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        mask = self.filter_mask(filters)
        hits = np.flatnonzero((scores > 0) if mask is None else (scores > 0) & mask)
        top = hits[np.argsort(-scores[hits], kind="stable")][:k]
        best = float(scores[top[0]]) if len(top) else 1.0
        phrase = " ".join(_words(query))
        out = []
        for f in top.tolist():
            text = " ".join(self._keyword_snippet(f, tokens, phrase if " " in phrase else None).split())
            out.append({"path": self.manifest["paths"][f], "score": float(scores[f]) / best,
                        "snippet": text[:SNIPPET_CHARS], "passage": text[:PASSAGE_CHARS], "chunk_id": None,
                        **self._result_meta(f)})
        return out

    def search(self, query: str, query_vec: np.ndarray | None, k: int = 5,
               filters: Filters | None = None) -> list[dict[str, Any]]:
        """Top-k files. Score is scaled to 0-1, where 1 means a literal match plus top ranks from both searches."""
        if not self.ready():
            return []
        mask = self.filter_mask(filters)
        sig = self.signals(query, query_vec, mask)
        score = self.fuse(sig, len(self.manifest["paths"]))
        if mask is not None:
            score[~mask] = 0
        top = [int(f) for f in np.argsort(-score, kind="stable")[:k] if score[f] > 0]
        best_possible = W_NAME + (W_KW + W_DENSE) / (RRF_K + 1)
        dense_hit = {f: (text, ci) for f, text, ci in sig["dense"]}
        phrase = " ".join(_words(query))
        results = []
        for f in top:
            # Show the matching paragraph for literal matches, else the closest chunk by meaning.
            literal = f in sig["phrase"] or f not in dense_hit
            text, chunk_index = ((self._keyword_snippet(f, sig["tokens"], phrase if f in sig["phrase"] else None), None)
                                 if literal else dense_hit[f])
            results.append({
                "path": self.manifest["paths"][f],
                "score": min(1.0, float(score[f]) / best_possible),
                "snippet": " ".join(text.split())[:SNIPPET_CHARS],
                "passage": " ".join(text.split())[:PASSAGE_CHARS],
                "chunk_id": chunk_index,
                **self._result_meta(f),
            })
        return results
