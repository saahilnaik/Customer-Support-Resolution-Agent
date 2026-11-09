"""Small local RAG: chunk markdown by '## ' headings, embed, cosine search.

Backends: 'fastembed' (local ONNX model, free; downloads once) with automatic
fallback to TF-IDF so the app still works offline.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import get_settings

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    source: str
    heading: str
    text: str


def load_chunks(kb_dir: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        title, heading, buf = path.stem, path.stem, []

        def flush():
            body = "\n".join(buf).strip()
            if body:
                chunks.append(Chunk(path.name, heading, f"{title} - {heading}\n{body}"))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                flush()
                heading, buf = line[3:].strip(), []
            elif line.startswith("# "):
                title = line[2:].strip()
            else:
                buf.append(line)
        flush()
    return chunks


class _TfidfEmbedder:
    min_score = 0.12

    def __init__(self, corpus: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(corpus)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.vec.transform(texts).toarray()


class _FastEmbedder:
    min_score = 0.45

    def __init__(self, model: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.embed(texts)))


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


class Retriever:
    def __init__(self, kb_dir: str | None = None, backend: str | None = None):
        s = get_settings()
        self.chunks = load_chunks(kb_dir or s.kb_dir)
        if not self.chunks:
            raise RuntimeError(f"no knowledge-base documents found in {kb_dir or s.kb_dir}")
        texts = [c.text for c in self.chunks]
        backend = backend or s.embedding_backend
        self.embedder = None
        if backend == "fastembed":
            try:
                self.embedder = _FastEmbedder(s.embedding_model)
            except Exception as e:  # offline, missing model, unsupported platform
                log.warning("fastembed unavailable (%s); falling back to TF-IDF", e)
        if self.embedder is None:
            self.embedder = _TfidfEmbedder(texts)
        self.matrix = _normalize(self.embedder.embed(texts))
        self.backend = type(self.embedder).__name__

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = k or get_settings().rag_top_k
        q = _normalize(self.embedder.embed([query]))[0]
        scores = self.matrix @ q
        out = []
        for i in np.argsort(-scores)[:k]:
            if scores[i] >= self.embedder.min_score:
                c = self.chunks[i]
                out.append({"source": c.source, "heading": c.heading,
                            "text": c.text, "score": round(float(scores[i]), 3)})
        return out


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
