"""Field-level precision / recall / F1 of an extracted resume against its ground truth."""
import re
from difflib import SequenceMatcher

from smartrecruit.schema import Resume

SINGLE_FIELDS = ("name", "email", "phone")
LIST_FIELDS = {
    "skills": lambda r: r.skills,
    "job_titles": lambda r: [e.title for e in r.experience],
    "companies": lambda r: [e.company for e in r.experience],
    "degrees": lambda r: [e.degree for e in r.education],
    "schools": lambda r: [e.school for e in r.education],
}
FIELDS = SINGLE_FIELDS + tuple(LIST_FIELDS)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+@.]", "", s.lower())


def same(a: str, b: str, threshold: float = 0.9) -> bool:
    a, b = norm(a), norm(b)
    return bool(a) and (a == b or SequenceMatcher(None, a, b).ratio() >= threshold)


def match_counts(pred: list[str], gold: list[str]) -> tuple[int, int, int]:
    """(true positives, predicted, gold) with one-to-one fuzzy matching."""
    remaining = list(gold)
    tp = 0
    for p in pred:
        for g in remaining:
            if same(p, g):
                tp += 1
                remaining.remove(g)
                break
    return tp, len(pred), len(gold)


def field_counts(pred: Resume, gold: Resume) -> dict[str, tuple[int, int, int]]:
    out = {}
    for f in SINGLE_FIELDS:
        p, g = getattr(pred, f), getattr(gold, f)
        out[f] = (int(same(p, g)), int(bool(p)), int(bool(g)))
    for f, get in LIST_FIELDS.items():
        out[f] = match_counts(get(pred), get(gold))
    return out


def f1_table(counts: list[dict[str, tuple[int, int, int]]]) -> dict[str, float]:
    """Micro-averaged F1 per field over a set of documents, plus the overall micro F1."""
    scores, total = {}, [0, 0, 0]
    for f in FIELDS:
        tp = sum(c[f][0] for c in counts)
        npred = sum(c[f][1] for c in counts)
        ngold = sum(c[f][2] for c in counts)
        total = [total[0] + tp, total[1] + npred, total[2] + ngold]
        p = tp / npred if npred else 0.0
        r = tp / ngold if ngold else 0.0
        scores[f] = 2 * p * r / (p + r) if p + r else 0.0
    p = total[0] / total[1] if total[1] else 0.0
    r = total[0] / total[2] if total[2] else 0.0
    scores["overall"] = 2 * p * r / (p + r) if p + r else 0.0
    return scores
