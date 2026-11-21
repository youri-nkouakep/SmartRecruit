import numpy as np


def precision_at_k(relevant: np.ndarray, k: int) -> float:
    return float(np.mean(relevant[:k]))


def mrr(relevant: np.ndarray) -> float:
    hits = np.flatnonzero(relevant)
    return 1.0 / (hits[0] + 1) if len(hits) else 0.0


def ndcg_at_k(relevant: np.ndarray, k: int, n_relevant: int) -> float:
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevant[:k] * discounts[: len(relevant[:k])]))
    ideal = float(np.sum(discounts[: min(k, n_relevant)]))
    return dcg / ideal if ideal else 0.0
