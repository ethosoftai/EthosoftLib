"""Retrieval over local text: BM25 by default, optional embeddings, no dependencies.

    >>> from ethosoftlib.rag import Index
    >>> index = Index()
    >>> index.add_directory("docs", pattern="*.md")
    >>> for hit in index.search("Mercan modeli nasıl indirilir?", k=3):
    ...     print(hit.source, hit.line, hit.score)

Documents are split into overlapping chunks of lines, so every hit points to
a file and a starting line. Turkish is agglutinative ("Ankara'nın",
"başkentidir"), so terms are cut to their first five letters after Turkish
lower-casing: a simple prefix stemmer that works well for Turkish retrieval
and harmlessly for English. Pass ``prefix_len=None`` to keep whole words.

An ``embedder`` (any object with ``embed(texts) -> list[vector]``) turns the
index hybrid: BM25 and cosine rankings are merged with reciprocal rank fusion.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import fnmatch
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Protocol, Sequence

from ..core.errors import EthosoftError

__all__ = ["Chunk", "Embedder", "Hit", "Index", "RagError", "tokenize"]

FORMAT_VERSION = 1
SKIP_DIRS = frozenset({".git", ".hg", ".svn", ".venv", "venv", "__pycache__",
                       "node_modules", ".mypy_cache", ".pytest_cache", "build", "dist"})
MAX_FILE_BYTES = 2 * 1024 * 1024
RRF_K = 60

_APOSTROPHE_SUFFIX = re.compile(r"['’]\w*")
_WORD = re.compile(r"[^\W_]+")
_STOPWORDS = frozenset("""
acaba ama ancak bana bazı belki ben beni benim bile bir biraz birçok biri birkaç
biz bize bizi bu buna bunu bunun da daha de defa diye en gibi hem hep her hiç
için ile ise kadar ki kim mı mi mu mü nasıl ne neden nerede niçin o olan olarak
on ona onu onun sen siz şey şu ve veya ya yani
a an and are as at be by for from in is it of on or that the this to was with
""".split())


class RagError(EthosoftError):
    """Indexing or searching failed."""


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
        ...


def _lower(text: str) -> str:
    # str.lower() maps "I" to "i" and "İ" to "i̇"; Turkish needs "ı" and "i".
    return text.replace("I", "ı").replace("İ", "i").lower()


def tokenize(text: str, prefix_len: int | None = 5) -> list[str]:
    """Search terms of ``text``: Turkish lower-case, no suffix after apostrophes,
    no stop words, words cut to ``prefix_len`` letters (numbers kept whole)."""
    terms = []
    for word in _WORD.findall(_APOSTROPHE_SUFFIX.sub("", _lower(text))):
        if word in _STOPWORDS:
            continue
        if prefix_len and not word.isdigit():
            word = word[:prefix_len]
        terms.append(word)
    return terms


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    line: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    line: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_lines(text: str, chunk_words: int, overlap_words: int) -> list[tuple[int, str]]:
    """Group lines into ``(first_line, text)`` chunks of about ``chunk_words``
    words; consecutive chunks share about ``overlap_words`` words."""
    lines = text.splitlines()
    counts = [len(line.split()) for line in lines]
    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(lines):
        while start < len(lines) and not counts[start]:
            start += 1
        if start == len(lines):
            break
        end, words = start, 0
        while end < len(lines) and (words < chunk_words or end == start):
            words += counts[end]
            end += 1
        chunks.append((start + 1, "\n".join(lines[start:end]).strip()))
        if end >= len(lines):
            break
        # Step back over whole lines until the overlap is covered.
        back, carried = end, 0
        while back - 1 > start and carried < overlap_words:
            back -= 1
            carried += counts[back]
        start = back if back > start else end
    return chunks


def _is_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        return b"\0" in handle.read(4096)


class Index:
    """In-memory BM25 index of text chunks, optionally hybrid with embeddings."""

    def __init__(self, *, embedder: Embedder | None = None, chunk_words: int = 120,
                 overlap_words: int = 30, prefix_len: int | None = 5,
                 k1: float = 1.5, b: float = 0.75) -> None:
        if chunk_words < 1 or overlap_words < 0:
            raise ValueError("chunk_words must be >= 1 and overlap_words >= 0")
        self.embedder = embedder
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.prefix_len = prefix_len
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self.chunks)

    # -------------------------------------------------------------- adding

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        new = [chunk for chunk in chunks if chunk.text.strip()]
        if self.embedder is not None and new:
            vectors = self.embedder.embed([chunk.text for chunk in new])
            if len(vectors) != len(new):
                raise RagError("Embedder returned a different number of vectors than texts.")
            self._vectors.extend(_normalize(vector) for vector in vectors)
        for chunk in new:
            doc = len(self.chunks)
            self.chunks.append(chunk)
            terms = Counter(tokenize(chunk.text, self.prefix_len))
            self._lengths.append(sum(terms.values()))
            for term, freq in terms.items():
                self._postings.setdefault(term, []).append((doc, freq))
        return len(new)

    def add(self, text: str, source: str = "", *, line: int = 1,
            metadata: dict[str, Any] | None = None) -> int:
        """Chunk ``text`` and index it; returns the number of chunks added."""
        return self.add_chunks(
            Chunk(body, source, line + first - 1, dict(metadata or {}))
            for first, body in split_lines(text, self.chunk_words, self.overlap_words)
        )

    def add_file(self, path: str | os.PathLike[str], *, source: str | None = None) -> int:
        file = Path(path)
        if not file.is_file():
            raise RagError(f"Not a file: {file}")
        if file.stat().st_size > MAX_FILE_BYTES or _is_binary(file):
            return 0
        text = file.read_text(encoding="utf-8", errors="replace")
        return self.add(text, source if source is not None else str(file))

    def add_directory(self, path: str | os.PathLike[str] = ".", *, pattern: str = "*",
                      recursive: bool = True, relative_to: str | os.PathLike[str] | None = None) -> int:
        """Index text files matching ``pattern``; binary, huge and tool
        directories (.git, .venv, node_modules, ...) are skipped."""
        root = Path(path)
        if not root.is_dir():
            raise RagError(f"Not a directory: {root}")
        base = Path(relative_to) if relative_to is not None else root
        added = 0
        for directory, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(files):
                if fnmatch.fnmatch(name, pattern):
                    file = Path(directory) / name
                    try:
                        source = str(file.relative_to(base))
                    except ValueError:
                        source = str(file)
                    added += self.add_file(file, source=source)
            if not recursive:
                break
        return added

    # ------------------------------------------------------------ searching

    def _bm25(self, query: str) -> list[tuple[int, float]]:
        terms = set(tokenize(query, self.prefix_len))
        if not terms or not self.chunks:
            return []
        count = len(self.chunks)
        average = sum(self._lengths) / count or 1.0
        scores: dict[int, float] = {}
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc, freq in postings:
                norm = self.k1 * (1 - self.b + self.b * self._lengths[doc] / average)
                scores[doc] = scores.get(doc, 0.0) + idf * freq * (self.k1 + 1) / (freq + norm)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    def _dense(self, query: str) -> list[tuple[int, float]]:
        vector = _normalize(self.embedder.embed([query])[0])
        scores = [(doc, sum(a * b for a, b in zip(vector, row)))
                  for doc, row in enumerate(self._vectors)]
        return sorted(scores, key=lambda item: (-item[1], item[0]))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Best ``k`` chunks for ``query``; empty when nothing matches."""
        if k < 1 or not self.chunks:
            return []
        ranked = self._bm25(query)
        if self.embedder is not None and self._vectors:
            dense = self._dense(query)[: max(k * 4, 20)]
            fused: dict[int, float] = {}
            for ranking in (ranked[: max(k * 4, 20)], dense):
                for rank, (doc, _) in enumerate(ranking):
                    fused[doc] = fused.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return [
            Hit(self.chunks[doc].text, self.chunks[doc].source, self.chunks[doc].line,
                round(score, 6), dict(self.chunks[doc].metadata))
            for doc, score in ranked[:k]
        ]

    # ---------------------------------------------------------- persistence

    def save(self, path: str | os.PathLike[str]) -> None:
        payload = {
            "format": FORMAT_VERSION,
            "settings": {"chunk_words": self.chunk_words, "overlap_words": self.overlap_words,
                         "prefix_len": self.prefix_len, "k1": self.k1, "b": self.b},
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "vectors": self._vectors,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, embedder: Embedder | None = None) -> "Index":
        """Load a saved index. Pass the same ``embedder`` it was built with to
        keep hybrid search; vectors are dropped when none is given."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT_VERSION:
            raise RagError(f"Unsupported index format: {payload.get('format')!r}")
        index = cls(**payload["settings"])
        index.add_chunks(Chunk(**chunk) for chunk in payload["chunks"])
        if embedder is not None:
            if len(payload["vectors"]) != len(index.chunks):
                raise RagError("Saved index has no vectors for every chunk; rebuild it.")
            index.embedder = embedder
            index._vectors = payload["vectors"]
        return index


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        raise RagError("Embedder returned a zero vector.")
    return [value / norm for value in values]
