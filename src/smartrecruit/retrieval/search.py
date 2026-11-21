"""Job offer -> resume search: BM25, dense retrieval (FAISS), cross-encoder re-ranking."""
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

BI_ENCODER = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN = re.compile(r"[a-z0-9+#]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, texts):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int = 10):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return top, scores[top]


class DenseIndex:
    def __init__(self, embeddings: np.ndarray, encoder=None):
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.encoder = encoder

    def encode_query(self, query: str) -> np.ndarray:
        return self.encoder.encode([QUERY_PREFIX + query], normalize_embeddings=True)

    def search(self, query: str, k: int = 10):
        scores, idx = self.index.search(self.encode_query(query).astype(np.float32), k)
        return idx[0], scores[0]


class Reranker:
    def __init__(self, model_name: str = CROSS_ENCODER, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=max_length, device="cpu")

    def rerank(self, query: str, texts: list[str], candidates, k: int = 10):
        scores = self.model.predict([(query, texts[i]) for i in candidates], batch_size=16,
                                    show_progress_bar=False)
        order = np.argsort(scores)[::-1][:k]
        return np.asarray(candidates)[order], np.asarray(scores)[order]


def chunk(text: str, words: int = 80, stride: int = 60) -> list[str]:
    tokens = text.split()
    return [" ".join(tokens[i:i + words]) for i in range(0, max(len(tokens) - words + stride, 1), stride)]

