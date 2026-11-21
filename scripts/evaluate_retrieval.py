#!/usr/bin/env python3
"""Offer -> resume retrieval on the 2,483 resumes. A resume is relevant if it belongs to the
category of the offer. Compares BM25, dense retrieval, hybrid fusion and cross-encoder re-ranking."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartrecruit.nlp.data import load  # noqa: E402
from smartrecruit.retrieval.metrics import mrr, ndcg_at_k, precision_at_k  # noqa: E402
from smartrecruit.retrieval.search import BI_ENCODER, BM25Index, DenseIndex, Reranker  # noqa: E402

K, CANDIDATES = 10, 50


def rrf(*rankings, k: int = 60) -> np.ndarray:
    scores = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + r + 1)
    return np.array(sorted(scores, key=scores.get, reverse=True))


def main():
    df = load()
    offers = json.loads((ROOT / "data" / "job_offers.json").read_text())
    emb_path = ROOT / "models" / "resume_embeddings.npy"
    encoder = SentenceTransformer(BI_ENCODER, device="cpu")
    encoder.max_seq_length = 512
    if emb_path.exists():
        emb = np.load(emb_path)
    else:
        emb = encoder.encode(list(df["text"]), batch_size=16, normalize_embeddings=True)
        np.save(emb_path, emb)
    bm25, dense, reranker = BM25Index(df["text"]), DenseIndex(emb, encoder), Reranker()
    texts, labels = list(df["text"]), df["label"].to_numpy()

    runs = {m: [] for m in ("BM25", "Dense (bge-small + FAISS)", "Hybrid (BM25 + dense, RRF)",
                            "Hybrid + cross-encoder re-ranking")}
    for offer in offers:
        q = offer["text"]
        b_idx, _ = bm25.search(q, CANDIDATES)
        d_idx, _ = dense.search(q, CANDIDATES)
        h_idx = rrf(b_idx, d_idx)[:CANDIDATES]
        r_idx, _ = reranker.rerank(q, texts, h_idx, K)
        for name, ranking in zip(runs, (b_idx, d_idx, h_idx, r_idx)):
            rel = (labels[ranking[:K]] == offer["category"]).astype(float)
            n_rel = int((labels == offer["category"]).sum())
            runs[name].append({"offer": offer["category"], "P@10": precision_at_k(rel, K),
                               "nDCG@10": ndcg_at_k(rel, K, n_rel), "MRR": mrr(rel)})

    table = pd.DataFrame({m: pd.DataFrame(r)[["P@10", "nDCG@10", "MRR"]].mean() for m, r in runs.items()}).T
    table.round(3).to_csv(ROOT / "results" / "retrieval.csv")
    per_offer = pd.DataFrame(runs["Hybrid + cross-encoder re-ranking"]).set_index("offer")
    per_offer.round(3).to_csv(ROOT / "results" / "retrieval_per_offer.csv")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
