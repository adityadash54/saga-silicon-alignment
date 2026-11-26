#!/usr/bin/env python3
"""Hybrid (BM25 + embedding) search over the alignment literature.

Indexes the PDFs in docs/TrackingLiterature/ (STDIAG_LITERATURE_DIR) into one
SQLite file, literature_index/literature.sqlite (STDIAG_LITERATURE_INDEX):
  - BM25 keyword search with SQLite FTS5 (standard library);
  - embedding search with a local Ollama model (default
    granite-embedding:30m, STDIAG_EMBED_MODEL) on localhost
    (STDIAG_OLLAMA_URL), cosine similarity with numpy;
  - the two rankings merged with reciprocal rank fusion.

Indexing is incremental. Files are identified by the SHA-256 of their
content, so only new or changed files are extracted and embedded; a renamed
or duplicated file is relinked without embedding, and removed files are
dropped. FTS5 keeps its BM25 statistics up to date by itself. A full
rebuild (`index --rebuild`) is needed only when the embedding model or the
chunking changes; `index` refuses to mix them.

Text comes from `pdftotext` (Homebrew poppler) in reading order, split into
page-aware windows of CHUNK_WORDS words. Nothing leaves this machine:
Ollama runs locally. Search opens the index read-only.

Usage:
  stdiag-literature index [--rebuild]
  stdiag-literature search "weak modes in the barrel" [-k 8] [--mode hybrid]
  stdiag-literature status
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np

from stdiag.paths import LITERATURE_DIR, LITERATURE_INDEX

SCHEMA_VERSION = "1"
CHUNKER_VERSION = "1"
DEFAULT_MODEL = os.environ.get("STDIAG_EMBED_MODEL", "granite-embedding:30m")
OLLAMA_URL = os.environ.get("STDIAG_OLLAMA_URL", "http://localhost:11434")
# granite-embedding:30m reads at most 512 tokens. Physics text (symbols,
# numbers, units) uses many tokens per word, so 250-word windows can exceed
# that; 200 words keeps nearly all chunks below the limit, and embed_chunks
# splits the rare longer one.
CHUNK_WORDS = 200
OVERLAP_WORDS = 30
MIN_SPLIT_WORDS = 20
EMBED_BATCH = 32
CANDIDATES = 50
RRF_K = 60
TITLE_WEIGHT = 0.5
MAX_QUERY_TERMS = 64
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has
have how i if in into is it its may might no not of on or our should so
such than that the their them then there these they this those to was we
were what when where which while who why will with would you your
""".split())


class LiteratureError(RuntimeError):
    """Problem the user has to fix (missing tool, index or model)."""


class OllamaError(LiteratureError):
    """Ollama unreachable or returned an error."""


class ContextOverflow(OllamaError):
    """An input is longer than the embedding model's context."""


# --------------------------------------------------------------------------
# Text extraction and chunking


def find_pdftotext() -> str:
    found = shutil.which("pdftotext")
    if found:
        return found
    if Path("/opt/homebrew/bin/pdftotext").exists():
        return "/opt/homebrew/bin/pdftotext"
    raise LiteratureError("pdftotext not found; install it with "
                          "`brew install poppler`.")


def extract_pages(path: Path) -> list[str]:
    """Text of each page, in reading order (handles two-column papers)."""
    result = subprocess.run([find_pdftotext(), "-enc", "UTF-8", str(path), "-"],
                            capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise LiteratureError(f"pdftotext failed on {path.name}: "
                              f"{result.stderr.strip()}")
    pages = result.stdout.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return pages


def clean_words(text: str) -> list[str]:
    """Words of a page with hyphenated line breaks joined."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.split()


@dataclasses.dataclass(frozen=True)
class Chunk:
    words: tuple[str, ...]
    pages: tuple[int, ...]  # 1-based page of each word

    @property
    def text(self) -> str:
        return " ".join(self.words)

    @property
    def page_start(self) -> int:
        return self.pages[0]

    @property
    def page_end(self) -> int:
        return self.pages[-1]

    def halves(self) -> list["Chunk"]:
        middle = len(self.words) // 2
        return [Chunk(self.words[:middle], self.pages[:middle]),
                Chunk(self.words[middle:], self.pages[middle:])]


def chunk_pages(pages: list[str], size: int = CHUNK_WORDS,
                overlap: int = OVERLAP_WORDS) -> list[Chunk]:
    """Overlapping windows of `size` words across pages."""
    words: list[str] = []
    numbers: list[int] = []
    for number, page in enumerate(pages, start=1):
        page_words = clean_words(page)
        words.extend(page_words)
        numbers.extend([number] * len(page_words))
    chunks = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunks.append(Chunk(tuple(words[start:start + size]),
                            tuple(numbers[start:start + size])))
        if start + size >= len(words):
            break
    return chunks


def title_of(path: Path) -> str:
    name = path.name
    return name[:-4] if name.lower().endswith(".pdf") else name


def embed_input(title: str, chunk: Chunk) -> str:
    return f"{title}\n{chunk.text}"


# --------------------------------------------------------------------------
# Embeddings


class Embedder(Protocol):
    model: str

    def digest(self) -> str: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class OllamaEmbedder:
    """Embeddings from a local Ollama server (no data leaves the machine)."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL,
                 timeout: float = 300):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _request(self, endpoint: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.url + endpoint, data=data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                return json.loads(reply.read())
        except urllib.error.HTTPError as error:
            message = error.read().decode(errors="replace")
            try:
                message = json.loads(message).get("error", message)
            except (ValueError, AttributeError):
                pass
            if "context" in message.lower() or "too long" in message.lower():
                raise ContextOverflow(message) from None
            raise OllamaError(f"Ollama {endpoint} failed ({error.code}): "
                              f"{message}") from None
        except (urllib.error.URLError, OSError) as error:
            raise OllamaError(
                f"cannot reach Ollama at {self.url} ({error}); start it with "
                "`brew services start ollama`.") from None

    def digest(self) -> str:
        models = self._request("/api/tags").get("models", [])
        wanted = {self.model, f"{self.model}:latest"}
        for entry in models:
            if entry.get("name") in wanted or entry.get("model") in wanted:
                return entry.get("digest", "")
        raise OllamaError(f"model {self.model} is not pulled; run "
                          f"`ollama pull {self.model}`.")

    def embed(self, texts: list[str]) -> np.ndarray:
        reply = self._request("/api/embed", {"model": self.model,
                                             "input": texts,
                                             "truncate": False})
        vectors = reply.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise OllamaError(f"Ollama returned {len(vectors or [])} "
                              f"embeddings for {len(texts)} inputs.")
        return normalize(np.array(vectors))


def embed_chunks(embedder: Embedder, title: str,
                 chunks: list[Chunk]) -> list[tuple[Chunk, np.ndarray]]:
    """Embed chunks in batches; a chunk too long for the model is split in
    half (recursively) instead of being truncated."""
    out: list[tuple[Chunk, np.ndarray]] = []
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start:start + EMBED_BATCH]
        try:
            vectors = embedder.embed([embed_input(title, c) for c in batch])
            out.extend(zip(batch, vectors))
            continue
        except ContextOverflow:
            pass
        for chunk in batch:
            try:
                out.append((chunk,
                            embedder.embed([embed_input(title, chunk)])[0]))
            except ContextOverflow:
                if len(chunk.words) < MIN_SPLIT_WORDS:
                    raise
                out.extend(embed_chunks(embedder, title, chunk.halves()))
    return out


# --------------------------------------------------------------------------
# Index


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files(
  path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, size INTEGER,
  mtime REAL, indexed_utc TEXT);
CREATE TABLE IF NOT EXISTS documents(
  sha256 TEXT PRIMARY KEY, title TEXT NOT NULL, n_pages INTEGER,
  n_chunks INTEGER);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, sha256 TEXT NOT NULL, title TEXT NOT NULL,
  page_start INTEGER, page_end INTEGER, text TEXT NOT NULL,
  embedding BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_sha ON chunks(sha256);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  title, text, content='chunks', content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, title, text)
  VALUES (new.id, new.title, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, title, text)
  VALUES ('delete', old.id, old.title, old.text);
END;
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_pdfs(folder: Path) -> list[Path]:
    """PDFs by content (%PDF header), with or without a .pdf extension."""
    found = []
    for path in sorted(folder.rglob("*")):
        if (path.is_file() and not path.name.startswith(".")
                and not any(p.startswith(".") for p in
                            path.relative_to(folder).parts)):
            with path.open("rb") as handle:
                if handle.read(5) == b"%PDF-":
                    found.append(path)
    return found


@dataclasses.dataclass
class UpdateSummary:
    new: int = 0
    failed: int = 0
    unchanged: int = 0
    relinked: int = 0
    removed: int = 0
    no_text: int = 0
    chunks_embedded: int = 0

    def line(self) -> str:
        return (f"{self.new} new/changed, {self.unchanged} unchanged, "
                f"{self.relinked} renamed/duplicate, {self.removed} removed, "
                f"{self.no_text} without text, {self.failed} failed; "
                f"embedded {self.chunks_embedded} chunks")


@dataclasses.dataclass
class Hit:
    id: int
    title: str
    path: str
    page_start: int
    page_end: int
    text: str
    score: float
    bm25_rank: int | None
    dense_rank: int | None

    def pages(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


def fts_query(text: str) -> str:
    """OR of quoted terms: safe for any input (no FTS5 syntax errors)."""
    terms: list[str] = []
    for term in re.findall(r"\w+", text.lower()):
        if (len(term) > 1 or term.isdigit()) and term not in STOPWORDS \
                and term not in terms:
            terms.append(term)
    return " OR ".join(f'"{term}"' for term in terms[:MAX_QUERY_TERMS])


def reciprocal_rank_fusion(rankings: Iterable[list[int]],
                           k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


class LiteratureIndex:
    def __init__(self, path: Path = LITERATURE_INDEX, readonly: bool = False):
        self.path = Path(path)
        if readonly:
            if not self.path.exists():
                raise LiteratureError(
                    f"no literature index at {self.path}; run "
                    "`stdiag-literature index` first.")
            self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro",
                                      uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path)
            self.db.executescript(SCHEMA)
        self._matrix: tuple[np.ndarray, np.ndarray] | None = None

    def close(self) -> None:
        self.db.close()

    # meta -----------------------------------------------------------------

    def meta(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT key, value FROM meta"))

    def _set_meta(self, values: dict[str, str]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                            values.items())

    def _check_meta(self, wanted: dict[str, str], rebuild: bool) -> None:
        stored = self.meta()
        has_data = self.db.execute("SELECT 1 FROM documents LIMIT 1").fetchone()
        mismatched = {key: (stored.get(key), value)
                      for key, value in wanted.items()
                      if stored.get(key) != value}
        if has_data and mismatched and not rebuild:
            details = "; ".join(f"{key}: index has {old!r}, now {new!r}"
                                for key, (old, new) in mismatched.items())
            raise LiteratureError(
                f"the index was built differently ({details}). Mixing "
                "embeddings is not possible; run `stdiag-literature index "
                "--rebuild`.")
        with self.db:
            if rebuild or mismatched:
                self.db.execute("DELETE FROM chunks")
                self.db.execute("DELETE FROM documents")
                self.db.execute("DELETE FROM files")
                self.db.execute("DELETE FROM meta WHERE key = 'embed_dim'")
            self._set_meta(wanted)

    # indexing -------------------------------------------------------------

    def update(self, folder: Path, embedder: Embedder,
               extractor: Callable[[Path], list[str]] = extract_pages,
               rebuild: bool = False,
               log: Callable[[str], None] = print) -> UpdateSummary:
        folder = Path(folder).resolve()
        self._check_meta({"schema_version": SCHEMA_VERSION,
                          "chunker_version": CHUNKER_VERSION,
                          "chunk_words": str(CHUNK_WORDS),
                          "embed_model": embedder.model,
                          "embed_digest": embedder.digest()}, rebuild)
        with self.db:
            self._set_meta({"literature_dir": str(folder)})
        summary = UpdateSummary()
        known = dict(self.db.execute("SELECT path, sha256 FROM files"))
        seen = set()
        for path in find_pdfs(folder):
            relative = path.relative_to(folder).as_posix()
            seen.add(relative)
            sha = sha256_file(path)
            stat = path.stat()
            row = (relative, sha, stat.st_size, stat.st_mtime,
                   dt.datetime.now(dt.timezone.utc).isoformat(
                       timespec="seconds"))
            if self.db.execute("SELECT 1 FROM documents WHERE sha256 = ?",
                               (sha,)).fetchone():
                if known.get(relative) == sha:
                    summary.unchanged += 1
                else:
                    summary.relinked += 1
                    log(f"  relinked (same content as an indexed file): "
                        f"{relative}")
                with self.db:
                    self.db.execute("INSERT OR REPLACE INTO files VALUES "
                                    "(?, ?, ?, ?, ?)", row)
                continue
            title = title_of(path)
            try:
                pages = extractor(path)
            except LiteratureError as error:
                summary.failed += 1  # an older entry, if any, is kept
                log(f"  WARNING: skipped {relative}: {error}")
                continue
            chunks = chunk_pages(pages)
            embedded = embed_chunks(embedder, title, chunks) if chunks else []
            if not chunks:
                summary.no_text += 1
                log(f"  WARNING: no text in {relative} (scanned? needs OCR)")
            else:
                summary.new += 1
                summary.chunks_embedded += len(embedded)
                log(f"  indexed {relative}: {len(pages)} pages, "
                    f"{len(embedded)} chunks")
            self._store(sha, title, len(pages), embedded, row)
        with self.db:
            for relative in set(known) - seen:
                self.db.execute("DELETE FROM files WHERE path = ?",
                                (relative,))
                summary.removed += 1
                log(f"  removed {relative}")
            # Content no file points to any more (removed or changed files).
            self.db.execute("DELETE FROM chunks WHERE sha256 NOT IN "
                            "(SELECT sha256 FROM files)")
            self.db.execute("DELETE FROM documents WHERE sha256 NOT IN "
                            "(SELECT sha256 FROM files)")
        return summary

    def _store(self, sha: str, title: str, n_pages: int,
               embedded: list[tuple[Chunk, np.ndarray]], row: tuple) -> None:
        """One transaction per file, so an interrupted run resumes cleanly."""
        with self.db:
            if embedded:
                dim = str(len(embedded[0][1]))
                stored = self.meta().get("embed_dim")
                if stored and stored != dim:
                    raise LiteratureError(f"embedding size {dim} differs from "
                                          f"the index ({stored}).")
                self._set_meta({"embed_dim": dim})
            self.db.executemany(
                "INSERT INTO chunks(sha256, title, page_start, page_end, "
                "text, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                [(sha, title, chunk.page_start, chunk.page_end, chunk.text,
                  np.asarray(vector, dtype=np.float32).tobytes())
                 for chunk, vector in embedded])
            self.db.execute("INSERT OR REPLACE INTO documents VALUES "
                            "(?, ?, ?, ?)", (sha, title, n_pages,
                                             len(embedded)))
            self.db.execute("INSERT OR REPLACE INTO files VALUES "
                            "(?, ?, ?, ?, ?)", row)

    # status ---------------------------------------------------------------

    def status(self, folder: Path | None = None) -> dict:
        meta = self.meta()
        info = {
            "index": str(self.path),
            "files": self.db.execute("SELECT COUNT(*) FROM files").fetchone()[0],
            "documents": self.db.execute(
                "SELECT COUNT(*) FROM documents").fetchone()[0],
            "chunks": self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            **{key: meta.get(key) for key in (
                "literature_dir", "embed_model", "embed_digest", "embed_dim",
                "chunker_version", "chunk_words")},
        }
        if folder is not None and Path(folder).exists():
            folder = Path(folder).resolve()
            indexed = dict(self.db.execute("SELECT path, sha256 FROM files"))
            on_disk = {p.relative_to(folder).as_posix(): p
                       for p in find_pdfs(folder)}
            info["not_indexed_or_changed"] = sorted(
                rel for rel, path in on_disk.items()
                if indexed.get(rel) != sha256_file(path))
            info["indexed_but_missing"] = sorted(set(indexed) - set(on_disk))
        return info

    # search ---------------------------------------------------------------

    def bm25_ranking(self, query: str, limit: int = CANDIDATES) -> list[int]:
        match = fts_query(query)
        if not match:
            return []
        rows = self.db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            f"ORDER BY bm25(chunks_fts, {TITLE_WEIGHT}, 1.0) LIMIT ?",
            (match, limit))
        return [row[0] for row in rows]

    def _embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        if self._matrix is None:
            rows = self.db.execute("SELECT id, embedding FROM chunks "
                                   "ORDER BY id").fetchall()
            ids = np.array([row[0] for row in rows], dtype=np.int64)
            matrix = (np.vstack([np.frombuffer(row[1], dtype=np.float32)
                                 for row in rows])
                      if rows else np.zeros((0, 0), dtype=np.float32))
            self._matrix = (ids, matrix)
        return self._matrix

    def dense_ranking(self, query_vector: np.ndarray,
                      limit: int = CANDIDATES) -> list[int]:
        ids, matrix = self._embeddings()
        if not len(ids):
            return []
        scores = matrix @ np.asarray(query_vector, dtype=np.float32)
        order = np.argsort(-scores)[:limit]
        return [int(ids[i]) for i in order]

    def check_embedder(self, embedder: Embedder) -> str | None:
        """None if the embedder matches the index, else the reason."""
        meta = self.meta()
        if meta.get("embed_model") != embedder.model:
            return (f"index built with {meta.get('embed_model')}, not "
                    f"{embedder.model}")
        if meta.get("embed_digest") != embedder.digest():
            return (f"the local {embedder.model} differs from the one the "
                    "index was built with; run `stdiag-literature index "
                    "--rebuild`")
        return None

    def search(self, query: str, k: int = 8, mode: str = "hybrid",
               embedder: Embedder | None = None,
               max_per_document: int = 3) -> tuple[list[Hit], list[str]]:
        """Top-k hits and warnings (for example "embedding search off")."""
        warnings: list[str] = []
        rankings: dict[str, list[int]] = {}
        if mode in ("hybrid", "bm25"):
            rankings["bm25"] = self.bm25_ranking(query)
        if mode in ("hybrid", "dense"):
            try:
                if embedder is None:
                    raise OllamaError("no embedder")
                problem = self.check_embedder(embedder)
                if problem:
                    raise OllamaError(problem)
                rankings["dense"] = self.dense_ranking(
                    embedder.embed([query])[0])
            except OllamaError as error:
                warnings.append(f"embedding search off ({error}); "
                                "results are BM25 only.")
                if mode == "dense":
                    return [], warnings
                rankings.setdefault("bm25", self.bm25_ranking(query))
        scores = reciprocal_rank_fusion(rankings.values())
        ranks = {name: {item: rank for rank, item in enumerate(ranking, 1)}
                 for name, ranking in rankings.items()}
        directory = self.meta().get("literature_dir", "")
        hits: list[Hit] = []
        per_document: dict[str, int] = {}
        for item in sorted(scores, key=lambda i: (-scores[i], i)):
            sha, title, start, end, text = self.db.execute(
                "SELECT sha256, title, page_start, page_end, text FROM chunks "
                "WHERE id = ?", (item,)).fetchone()
            if per_document.get(sha, 0) >= max_per_document:
                continue
            per_document[sha] = per_document.get(sha, 0) + 1
            path = self.db.execute("SELECT path FROM files WHERE sha256 = ? "
                                   "ORDER BY path LIMIT 1", (sha,)).fetchone()
            hits.append(Hit(
                id=item, title=title,
                path=str(Path(directory) / path[0]) if path else "",
                page_start=start, page_end=end, text=text,
                score=scores[item], bm25_rank=ranks.get("bm25", {}).get(item),
                dense_rank=ranks.get("dense", {}).get(item)))
            if len(hits) >= k:
                break
        return hits, warnings


def format_hits(hits: list[Hit], warnings: list[str] = ()) -> str:
    lines = [f"> WARNING: {warning}" for warning in warnings]
    if not hits:
        lines.append("No matching passages.")
    for number, hit in enumerate(hits, start=1):
        ranks = ", ".join(f"{name} #{rank}" for name, rank in
                          (("bm25", hit.bm25_rank), ("dense", hit.dense_rank))
                          if rank is not None)
        lines.append(f"[{number}] {hit.title}, {hit.pages()} ({ranks})\n"
                     f"    file: {hit.path}\n\n{hit.text}\n")
    return "\n".join(lines)


def search_literature(query: str, k: int = 6, mode: str = "hybrid",
                      index_path: Path = LITERATURE_INDEX,
                      embedder: Embedder | None = None,
                      max_per_document: int = 2) -> tuple[list[Hit], list[str]]:
    """Convenience wrapper used by the tracking expert."""
    index = LiteratureIndex(index_path, readonly=True)
    try:
        return index.search(query, k=k, mode=mode,
                            embedder=embedder or OllamaEmbedder(),
                            max_per_document=max_per_document)
    finally:
        index.close()


def search_command() -> str:
    """The command the tracking expert may run (for its Bash allowlist)."""
    beside_python = Path(sys.executable).parent / "stdiag-literature"
    found = shutil.which("stdiag-literature")
    if beside_python.exists():
        return str(beside_python)
    return found or "stdiag-literature"


# --------------------------------------------------------------------------
# Command line


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="stdiag-literature", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=LITERATURE_INDEX,
                        help=f"index file (default {LITERATURE_INDEX})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama embedding model (default {DEFAULT_MODEL})")
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser("index", help="add new/changed files")
    index_parser.add_argument("--dir", type=Path, default=LITERATURE_DIR,
                              help=f"literature folder (default {LITERATURE_DIR})")
    index_parser.add_argument("--rebuild", action="store_true",
                              help="re-embed everything (model/chunking change)")

    search_parser = commands.add_parser("search", help="hybrid search")
    search_parser.add_argument("query", nargs="+")
    search_parser.add_argument("-k", type=int, default=8)
    search_parser.add_argument("--mode", default="hybrid",
                               choices=["hybrid", "bm25", "dense"])
    search_parser.add_argument("--max-per-doc", type=int, default=3)
    search_parser.add_argument("--json", action="store_true")

    status_parser = commands.add_parser("status", help="what is indexed")
    status_parser.add_argument("--dir", type=Path, default=LITERATURE_DIR)

    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            if not args.dir.is_dir():
                raise LiteratureError(f"no literature folder at {args.dir}")
            index = LiteratureIndex(args.index)
            summary = index.update(args.dir, OllamaEmbedder(args.model),
                                   rebuild=args.rebuild)
            print(summary.line())
        elif args.command == "search":
            index = LiteratureIndex(args.index, readonly=True)
            hits, warnings = index.search(
                " ".join(args.query), k=args.k, mode=args.mode,
                embedder=OllamaEmbedder(args.model),
                max_per_document=args.max_per_doc)
            if args.json:
                print(json.dumps({"warnings": warnings, "hits": [
                    dataclasses.asdict(hit) for hit in hits]}, indent=2))
            else:
                print(format_hits(hits, warnings))
        else:
            index = LiteratureIndex(args.index, readonly=True)
            print(json.dumps(index.status(args.dir), indent=2))
    except LiteratureError as error:
        sys.exit(f"ERROR: {error}")


if __name__ == "__main__":
    main()
