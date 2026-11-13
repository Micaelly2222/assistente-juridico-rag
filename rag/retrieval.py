import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from rag.config import EMBEDDING_MODEL, EMBEDDINGS, TOP_K
from rag.ingest import Chunk, load_corpus

STOPWORDS = set("""
a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas para com sem sob
e ou que se ao aos à às é ser são foi como mais menos quando qual quais quem onde sua seu suas seus
este esta estes estas esse essa isso isto aquele aquela lei art artigo nº
""".split())


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", normalize(text))
    # Radical simples: corta plural e algumas terminações para casar "dados"/"dado", "prazos"/"prazo"
    out = []
    for t in tokens:
        if t in STOPWORDS or len(t) < 2:
            continue
        for suffix in ("coes", "cao", "mente", "ais", "es", "s"):
            if len(t) > 5 and t.endswith(suffix):
                t = t[: -len(suffix)]
                break
        out.append(t)
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.lengths = np.array([len(d) for d in self.docs], dtype=float)
        self.avg = self.lengths.mean()
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
            out[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        return out


@lru_cache(maxsize=1)
def embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    vectors = np.array(list(embedder().embed(texts)), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def chunk_text_for_embedding(c: Chunk) -> str:
    return f"{c.law_short}, {c.heading}. {c.text}"


def build_embeddings(chunks: list[Chunk]) -> np.ndarray:
    vectors = embed([chunk_text_for_embedding(c) for c in chunks])
    np.save(EMBEDDINGS, vectors)
    return vectors


def rrf(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion: soma 1/(k + posição) de cada ranking. Não depende da escala dos scores."""
    fused = np.zeros(len(rankings[0]))
    for order in rankings:
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(len(order))
        fused += 1.0 / (k + ranks + 1)
    return fused


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float
    dense: float


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, vectors: np.ndarray | None = None):
        self.chunks = chunks or load_corpus()
        self.bm25 = BM25([f"{c.heading} {c.text}" for c in self.chunks])
        if vectors is None:
            vectors = np.load(EMBEDDINGS) if EMBEDDINGS.exists() else build_embeddings(self.chunks)
        self.vectors = vectors

    def search(self, query: str, k: int = TOP_K, mode: str = "hybrid", law: str | None = None) -> list[Hit]:
        bm25 = self.bm25.scores(query)
        dense = self.vectors @ embed([query])[0] if mode != "bm25" else np.zeros(len(self.chunks))
        if mode == "bm25":
            final = bm25
        elif mode == "dense":
            final = dense
        else:
            final = rrf([np.argsort(-bm25), np.argsort(-dense)])
        if law:
            final = np.where([c.law == law for c in self.chunks], final, -np.inf)

        # Um resultado por artigo: o melhor trecho de cada artigo
        hits, seen = [], set()
        for i in np.argsort(-final):
            c = self.chunks[i]
            key = (c.law, c.article)
            if key in seen or final[i] == -np.inf:
                continue
            seen.add(key)
            hits.append(Hit(c, float(final[i]), float(bm25[i]), float(dense[i])))
            if len(hits) == k:
                break
        return hits

    def article(self, law: str, number: int) -> list[Chunk]:
        return [c for c in self.chunks if c.law == law and c.article == number]
