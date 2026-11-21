import numpy as np

from smartrecruit.fairness.audit import NAMES, summarize, variants
from smartrecruit.nlp.data import strip_title
from smartrecruit.retrieval.explain import grounding, requirements, sentences
from smartrecruit.retrieval.metrics import mrr, ndcg_at_k, precision_at_k
from smartrecruit.retrieval.search import BM25Index, chunk


def test_metrics():
    rel = np.array([0, 1, 1, 0])
    assert precision_at_k(rel, 4) == 0.5
    assert mrr(rel) == 0.5
    assert ndcg_at_k(np.ones(3), 3, 3) == 1.0


def test_bm25_finds_keyword():
    idx = BM25Index(["chef menu kitchen", "python developer", "accountant ledger"])
    assert idx.search("senior python engineer", 1)[0][0] == 1


def test_chunk_covers_text():
    text = " ".join(str(i) for i in range(200))
    parts = chunk(text, words=80, stride=60)
    assert parts[0].startswith("0 ") and parts[-1].endswith("199")


def test_grounding():
    answer = "- Leads kitchens [E1]\n- Has ServSafe [E9]\n- No budget experience"
    assert grounding(answer, 3) == (1, 3)


def test_requirements_and_sentences():
    reqs = requirements("Chef. Lead the kitchen team, design menus. Requirements: culinary degree.")
    assert "design menus" in reqs
    assert sentences("Managed a team of 12 cooks. Built menus for events.")[1].startswith("Built")


def test_variants_and_summary():
    v = variants("Experienced accountant.")
    assert len(v) == sum(len(n) for n in NAMES.values())
    s = summarize({k: np.array([1.0]) for k in v})
    assert s["gender_gap"] == 0 and s["origin_gap"] == 0


def test_strip_title():
    assert strip_title("   HR MANAGER       Summary   Ten years in HR.") == "Summary Ten years in HR."
