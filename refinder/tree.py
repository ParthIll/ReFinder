"""Vector tree: files grouped into a hierarchy of semantic categories.

    root
     ├─ category        (centroid of its members, e.g. "matrix, eigenvalue, vector")
     │   ├─ subcategory
     │   │   └─ file parts ── chunks
     │   └─ ...
     └─ ...

Build: each file is summarized by one or more "part" vectors. A small file gets one (the normalized mean of its
chunk vectors); a big file gets up to `max_parts`, one per k-means cluster of its own chunks, so a textbook that
covers matrices, eigenvalues and proofs can sit in each of those branches instead of being averaged into one
blurry vector. Parts are split recursively with spherical k-means until a node holds at most `leaf_size` of
them. Every node stores the normalized centroid of its parts, plus keywords for display.

Two ways to use it at search time:

  prune  beam search from the root, keeping the `beam` best-matching nodes per level; only the files reached
         are scored (best chunk wins, as in flat search). Cheaper, but a pruned branch hides its files.
  boost  score every file as in flat search, then add `weight` x how well the query matches the categories the
         file sits in (its best part's mean similarity to that part's ancestors). Nothing is hidden; the tree
         acts as context, rewarding a chunk match that is also in the right topic area.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def spherical_kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 30) -> np.ndarray:
    """Cluster unit vectors by cosine similarity; returns a cluster label per row (k-means++ init)."""
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        dist = np.clip(1 - np.max(x @ np.array(centers).T, axis=1), 0, None)
        centers.append(x[rng.choice(len(x), p=dist / dist.sum())] if dist.sum() > 0 else x[rng.integers(len(x))])
    c = np.array(centers)
    labels = np.argmax(x @ c.T, axis=1)
    for _ in range(iters):
        c = _normalize(np.array([x[labels == j].mean(0) if (labels == j).any() else c[j] for j in range(k)]))
        new_labels = np.argmax(x @ c.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels


@dataclass
class Node:
    id: int
    centroid: np.ndarray
    parts: np.ndarray  # every part id under this node
    depth: int
    children: list["Node"] = field(default_factory=list)
    label: str = ""

    @property
    def is_leaf(self) -> bool:
        return not self.children


STOPWORDS = frozenset("""
the and for are but not you all any can had her was one our out has have this that with from they will would
there their what when which who whom were been being into than then them these those its his him she also such
use used using each other more most some only over very just about after before between both under while where
how why may might must should could does did doing done yes per via etc see page pages chapter section figure
table example examples note notes http https www com org edu pdf doc
""".split())
WORD = re.compile(r"[a-z][a-z]{2,}")
LABEL_CHARS = 300_000  # enough text per file to find its themes, without tokenizing whole textbooks


class VectorTree:
    def __init__(
        self,
        vecs: np.ndarray,
        owner: np.ndarray,
        n_files: int,
        branching: int = 8,
        leaf_size: int = 10,
        max_parts: int = 1,
        part_chunks: int = 40,
        seed: int = 0,
    ):
        """vecs/owner: chunk vectors (unit length) and their file ids, stored contiguously per file.

        A file gets min(max_parts, round(chunks / part_chunks)) part vectors (at least one).
        """
        self.vecs, self.n_files = vecs, n_files
        self.branching, self.leaf_size, self.seed = branching, leaf_size, seed
        starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        ends = np.r_[starts[1:], len(owner)]
        self.chunk_range = {int(owner[s]): (int(s), int(e)) for s, e in zip(starts, ends)}

        part_vecs, part_file = [], []
        for f, (s, e) in self.chunk_range.items():  # files with no text have no chunks: not in the tree
            m = min(max_parts, max(1, round((e - s) / part_chunks)))
            if m == 1:
                part_vecs.append(_normalize(vecs[s:e].mean(0)))
                part_file.append(f)
                continue
            labels = spherical_kmeans(vecs[s:e], m, seed=seed + f)
            for j in np.unique(labels):
                part_vecs.append(_normalize(vecs[s:e][labels == j].mean(0)))
                part_file.append(f)
        self.part_vecs, self.part_file = np.array(part_vecs), np.array(part_file, dtype=int)

        self._nodes: list[Node] = []
        self.root = self._build(np.arange(len(self.part_vecs)), depth=0)
        self.centroids = np.array([n.centroid for n in self._nodes])
        # Per part: equal weight on each ancestor category below the root, so part_bonus = W @ (centroids @ q).
        self.ancestor_weights = np.zeros((len(self.part_vecs), len(self._nodes)), dtype=vecs.dtype)
        self._fill_ancestors(self.root, [])

    def _build(self, parts: np.ndarray, depth: int) -> Node:
        node = Node(len(self._nodes), _normalize(self.part_vecs[parts].mean(0)), parts, depth)
        self._nodes.append(node)
        if len(parts) <= self.leaf_size:
            return node
        k = min(self.branching, math.ceil(len(parts) / self.leaf_size))
        labels = spherical_kmeans(self.part_vecs[parts], k, seed=self.seed + depth)
        groups = [parts[labels == j] for j in range(k) if (labels == j).any()]
        if len(groups) > 1:  # identical vectors can't be split further; keep them as one leaf
            node.children = [self._build(g, depth + 1) for g in groups]
        return node

    def _fill_ancestors(self, node: Node, path: list[int]) -> None:
        path = path + [node.id] if node.depth > 0 else path
        if node.is_leaf:
            for p in node.parts:
                self.ancestor_weights[p, path] = 1.0 / max(len(path), 1)
        for c in node.children:
            self._fill_ancestors(c, path)

    def files_of(self, node: Node) -> np.ndarray:
        return np.unique(self.part_file[node.parts])

    def nodes(self):
        return iter(self._nodes)

    @property
    def depth(self) -> int:
        return max(n.depth for n in self._nodes)

    def label(self, texts: list[str], n_terms: int = 4) -> None:
        """Name each node by c-TF-IDF: terms frequent in its files but rare across the corpus."""
        counts = {f: Counter(w for w in WORD.findall(texts[f][:LABEL_CHARS].lower()) if w not in STOPWORDS)
                  for f in self.chunk_range}
        freqs = {f: {w: n / max(sum(c.values()), 1) for w, n in c.items()} for f, c in counts.items()}
        df = Counter(w for c in counts.values() for w in c)
        n = len(counts)
        for node in self._nodes:
            files = self.files_of(node)
            total = Counter()
            for f in files:
                total.update(freqs[int(f)])
            score = {w: tf * math.log(n / df[w]) for w, tf in total.items() if df[w] > 1 or len(files) == 1}
            node.label = ", ".join(sorted(score, key=score.get, reverse=True)[:n_terms])

    # ---- search

    def candidates(self, q: np.ndarray, beam: int) -> np.ndarray:
        """File ids reached by keeping the `beam` best-matching nodes at each level of the tree."""
        frontier, found = [self.root], []
        while frontier:
            if len(frontier) > beam:
                sims = self.centroids[[n.id for n in frontier]] @ q
                frontier = [frontier[i] for i in np.argsort(-sims)[:beam]]
            found += [n.parts for n in frontier if n.is_leaf]
            frontier = [c for n in frontier for c in n.children]
        return np.unique(self.part_file[np.concatenate(found)]) if found else np.array([], dtype=int)

    def score_files(self, q: np.ndarray, files: np.ndarray) -> tuple[np.ndarray, int]:
        """Best-chunk cosine score for each given file; also returns how many chunks were scored."""
        ranges = [self.chunk_range[int(f)] for f in files]
        idx = np.concatenate([np.arange(s, e) for s, e in ranges])
        seg_starts = np.cumsum([0] + [e - s for s, e in ranges[:-1]])
        return np.maximum.reduceat(self.vecs[idx] @ q, seg_starts), len(idx)

    def category_scores(self, q_vecs: np.ndarray) -> np.ndarray:
        """(n_queries, n_files): how well each query matches the categories each file sits in (best part)."""
        part_scores = (q_vecs @ self.centroids.T) @ self.ancestor_weights.T
        out = np.full((len(q_vecs), self.n_files), -np.inf, dtype=np.float32)
        for p, f in enumerate(self.part_file):
            np.maximum(out[:, f], part_scores[:, p], out=out[:, f])
        return out

    def outline(self, paths: list[str], max_depth: int = 2) -> str:
        """Readable tree: each node's size, label and the folder most of its files come from."""
        lines = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            stack.extend(reversed(node.children))
            if node.depth == 0 or node.depth > max_depth:
                continue
            files = self.files_of(node)
            top, count = Counter("/".join(paths[f].split("/")[:2]) for f in files).most_common(1)[0]
            lines.append(f"{'    ' * (node.depth - 1)}[{len(files):3d} files] {node.label}"
                         f"   ({count}/{len(files)} from {top})")
        return "\n".join(lines)
