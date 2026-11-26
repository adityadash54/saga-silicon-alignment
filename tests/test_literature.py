"""Literature index: incremental updates, hybrid search, tracking-expert use.

Uses fake "PDFs" (a %PDF- header followed by text, pages split by form
feeds), a fake extractor and a deterministic fake embedder, so neither
poppler nor Ollama is needed.
"""

from __future__ import annotations

import functools
import hashlib
import re
import sqlite3

import numpy as np
import pytest

from stdiag import literature, tracking_expert
from stdiag.literature import (Chunk, ContextOverflow, LiteratureError,
                               LiteratureIndex, OllamaError, chunk_pages,
                               fts_query, reciprocal_rank_fusion)


class FakeEmbedder:
    """Hashed bag of words; records what it embedded."""

    def __init__(self, model="fake-embed", digest="d1", max_words=None,
                 down=False):
        self.model = model
        self._digest = digest
        self.max_words = max_words
        self.down = down
        self.embedded: list[str] = []

    def digest(self):
        if self.down:
            raise OllamaError("cannot reach Ollama (test)")
        return self._digest

    def embed(self, texts):
        if self.down:
            raise OllamaError("cannot reach Ollama (test)")
        if self.max_words and any(len(t.split()) > self.max_words
                                  for t in texts):
            raise ContextOverflow("input length exceeds the context length")
        self.embedded.extend(texts)
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % 64
                vectors[row, slot] += 1
        return literature.normalize(vectors + 1e-6)


def fake_extract(path):
    return path.read_bytes()[len(b"%PDF-\n"):].decode().split("\f")


def write_pdf(folder, name, *pages):
    path = folder / name
    path.write_bytes(b"%PDF-\n" + "\f".join(pages).encode())
    return path


def filler(n, word="detector"):
    return " ".join(f"{word}{i % 7}" for i in range(n))


@pytest.fixture()
def library(tmp_path):
    folder = tmp_path / "lit"
    folder.mkdir()
    write_pdf(folder, "Paper A.pdf",
              "Twist and telescope weak modes in a silicon barrel. "
              + filler(300),
              "Constraining the twist with cosmic rays. " + filler(100))
    write_pdf(folder, "Paper B.pdf",
              "Track model with multiple scattering for "
              "Millepede alignment. " + filler(120))
    index = LiteratureIndex(tmp_path / "index" / "lit.sqlite")
    embedder = FakeEmbedder()
    update = functools.partial(index.update, folder, extractor=fake_extract,
                               log=lambda line: None)
    yield folder, index, embedder, update
    index.close()


def chunk_count(index, title=None):
    if title is None:
        return index.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return index.db.execute("SELECT COUNT(*) FROM chunks WHERE title = ?",
                            (title,)).fetchone()[0]


def test_chunks_are_page_aware_windows():
    pages = [" ".join(f"a{i}" for i in range(200)),
             " ".join(f"b{i}" for i in range(200))]
    chunks = chunk_pages(pages, size=250, overlap=40)
    assert [len(c.words) for c in chunks] == [250, 190]
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)
    assert chunks[1].words[0] == "b10"  # 40-word overlap
    assert chunk_pages(["  ", ""]) == []


def test_hyphenated_line_breaks_are_joined():
    assert chunk_pages(["mis-\nalignment of the de-\ntector"])[0].text == \
        "misalignment of the detector"


def test_find_pdfs_by_content_not_extension(tmp_path):
    write_pdf(tmp_path, "Paper D", "text")
    write_pdf(tmp_path, "paper.pdf", "text")
    (tmp_path / "notes.txt").write_text("not a pdf")
    (tmp_path / ".DS_Store").write_bytes(b"%PDF-")
    assert [p.name for p in literature.find_pdfs(tmp_path)] == \
        ["Paper D", "paper.pdf"]


def test_first_index_then_rerun_does_nothing(library):
    folder, index, embedder, update = library
    first = update(embedder)
    assert (first.new, first.unchanged) == (2, 0)
    assert first.chunks_embedded == chunk_count(index) > 0
    calls = len(embedder.embedded)

    second = update(embedder)
    assert (second.new, second.unchanged, second.removed) == (0, 2, 0)
    assert len(embedder.embedded) == calls


def test_adding_a_file_embeds_only_that_file(library):
    folder, index, embedder, update = library
    update(embedder)
    before = len(embedder.embedded)
    write_pdf(folder, "Paper C.pdf", "Detector halves alignment. " + filler(50))

    summary = update(embedder)
    assert (summary.new, summary.unchanged) == (1, 2)
    new_inputs = embedder.embedded[before:]
    assert new_inputs and all(t.startswith("Paper C\n") for t in new_inputs)


def test_changed_file_replaces_its_chunks(library):
    folder, index, embedder, update = library
    update(embedder)
    write_pdf(folder, "Paper B.pdf",
              "Revised: scattering angles and the covariance. " + filler(40))

    summary = update(embedder)
    assert (summary.new, summary.unchanged) == (1, 1)
    assert index.bm25_ranking("Millepede") == []          # old text gone
    assert index.bm25_ranking("covariance") != []
    assert index.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2


def test_removed_file_is_dropped(library):
    folder, index, embedder, update = library
    update(embedder)
    (folder / "Paper B.pdf").unlink()

    summary = update(embedder)
    assert summary.removed == 1
    assert chunk_count(index, "Paper B") == 0
    assert index.bm25_ranking("Millepede") == []


def test_rename_and_duplicate_are_not_reembedded(library):
    folder, index, embedder, update = library
    update(embedder)
    before = len(embedder.embedded)
    chunks = chunk_count(index)
    (folder / "Paper B.pdf").rename(folder / "Paper B renamed")
    (folder / "copy of Paper A.pdf").write_bytes(
        (folder / "Paper A.pdf").read_bytes())

    summary = update(embedder)
    assert (summary.new, summary.relinked, summary.removed) == (0, 2, 1)
    assert len(embedder.embedded) == before
    assert chunk_count(index) == chunks
    hits, _ = index.search("track model Millepede", k=1, embedder=embedder)
    assert hits[0].path.endswith("/Paper B renamed")


def test_model_change_is_refused_until_rebuild(library):
    folder, index, embedder, update = library
    update(embedder)
    with pytest.raises(LiteratureError, match="--rebuild"):
        update(FakeEmbedder(digest="d2"))

    other = FakeEmbedder(digest="d2")
    summary = update(other, rebuild=True)
    assert summary.new == 2 and other.embedded
    assert index.meta()["embed_digest"] == "d2"


def test_overlong_chunk_is_split_not_truncated(library):
    folder, index, _, update = library
    embedder = FakeEmbedder(max_words=150)
    summary = update(embedder)
    assert summary.new == 2
    assert all(len(text.split()) <= 150 for text in embedder.embedded)
    stored = " ".join(row[0] for row in index.db.execute(
        "SELECT text FROM chunks WHERE title = 'Paper B' "
        "ORDER BY id"))
    assert "Track model with multiple" in stored and "detector6" in stored


def test_extraction_failure_skips_the_file_only(library):
    folder, index, embedder, update = library

    def extractor(path):
        if "Paper A" in path.name:
            raise LiteratureError("pdftotext failed (test)")
        return fake_extract(path)

    summary = index.update(folder, embedder, extractor=extractor,
                           log=lambda line: None)
    assert (summary.new, summary.failed) == (1, 1)
    assert update(embedder).new == 1   # retried on the next run


def test_hybrid_search_finds_the_right_document(library):
    folder, index, embedder, update = library
    update(embedder)
    hits, warnings = index.search("twist weak mode cosmic", k=3,
                                  embedder=embedder)
    assert warnings == []
    assert hits[0].title == "Paper A"
    assert hits[0].bm25_rank and hits[0].dense_rank
    assert hits[0].path.endswith("Paper A.pdf")

    other, _ = index.search("multiple scattering track model", k=1,
                            embedder=embedder)
    assert other[0].title == "Paper B"


def test_search_falls_back_to_bm25_with_a_warning(library):
    folder, index, embedder, update = library
    update(embedder)
    hits, warnings = index.search("twist", embedder=FakeEmbedder(down=True))
    assert hits and all(hit.dense_rank is None for hit in hits)
    assert warnings and "BM25 only" in warnings[0]

    _, warnings = index.search("twist", embedder=FakeEmbedder(digest="other"))
    assert "differs" in warnings[0]


def test_max_hits_per_document(library):
    folder, index, embedder, update = library
    update(embedder)
    hits, _ = index.search("detector1 detector2", k=10, embedder=embedder,
                           max_per_document=1)
    titles = [hit.title for hit in hits]
    assert len(titles) == len(set(titles))


@pytest.mark.parametrize("query", ['χ² "weak" -mode', "twist AND (OR",
                                   "NEAR(a b)", "x*", ":", ""])
def test_any_query_text_is_safe(library, query):
    folder, index, embedder, update = library
    update(embedder)
    index.bm25_ranking(query)  # must not raise an FTS5 syntax error


def test_fts_query_quotes_terms_and_drops_stopwords():
    assert fts_query("The twist of the silicon barrel") == \
        '"twist" OR "silicon" OR "barrel"'
    assert fts_query("χ² chi2") == '"χ²" OR "chi2"'


def test_reciprocal_rank_fusion_prefers_agreement():
    scores = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1]], k=60)
    assert max(scores, key=scores.get) == 2


def test_search_opens_the_index_read_only(library):
    folder, index, embedder, update = library
    update(embedder)
    readonly = LiteratureIndex(index.path, readonly=True)
    with pytest.raises(sqlite3.OperationalError):
        readonly.db.execute("DELETE FROM chunks")
    readonly.close()


def test_missing_index_is_a_clear_error(tmp_path):
    with pytest.raises(LiteratureError, match="stdiag-literature index"):
        LiteratureIndex(tmp_path / "none.sqlite", readonly=True)


def test_tracking_request_contains_literature_passages(library, monkeypatch):
    folder, index, embedder, update = library
    update(embedder)
    monkeypatch.setattr(tracking_expert, "LITERATURE_INDEX", index.path)
    monkeypatch.setattr(tracking_expert, "search_literature", functools.partial(
        literature.search_literature, embedder=embedder))
    request = {"finding": {"id": "F1", "title": "MVTX layer 0 twist-like "
                           "rphi shift", "evidence": "weak mode suspected",
                           "entries": [{"detector": "MVTX", "coord": "rphi"}]}}
    answer = "### Summary\nA twist weak mode from cosmic constraints.\n### Hypotheses\n|x|"

    text = tracking_expert.build_request(request, answer)
    assert "## Literature passages" in text
    assert "Paper A, p." in text
    query = tracking_expert.literature_query(request, answer)
    assert "r-phi" in query and "Hypotheses" not in query
    assert "MVTX" not in query and "pixel vertex detector" in query


def test_tracking_request_without_index_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking_expert, "LITERATURE_INDEX",
                        tmp_path / "missing.sqlite")
    text = tracking_expert.literature_section({"finding": {}}, "answer")
    assert "Not available" in text and "stdiag-literature index" in text


def test_tracking_command_allows_only_literature_search(monkeypatch):
    monkeypatch.setattr(tracking_expert, "search_command",
                        lambda: "/venv/bin/stdiag-literature")
    command = tracking_expert.claude_command("claude", "model", "high")
    assert "Bash(/venv/bin/stdiag-literature search *)" in command
    prompt = command[command.index("--append-system-prompt") + 1]
    assert "/venv/bin/stdiag-literature search" in prompt
    assert "{search}" not in prompt and "{#anchor}" in prompt


def test_chunk_halves_keep_pages():
    chunk = Chunk(tuple("abcd"), (1, 1, 2, 2))
    first, second = chunk.halves()
    assert (first.page_end, second.page_start) == (1, 2)
