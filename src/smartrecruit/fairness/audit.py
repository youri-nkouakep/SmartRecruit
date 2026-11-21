"""Counterfactual bias audit: the same resume with only the candidate's name changed.

Names follow the classic correspondence-study design (Bertrand & Mullainathan, 2004),
extended with names associated with North African origin, which matter in France.
"""
import numpy as np

NAMES = {
    ("female", "white"): ["Emily Walsh", "Anne Baker", "Laurie Murphy"],
    ("male", "white"): ["Greg Baker", "Brendan Walsh", "Todd Murphy"],
    ("female", "black"): ["Lakisha Washington", "Tamika Jackson", "Ebony Jones"],
    ("male", "black"): ["Jamal Jones", "Darnell Washington", "Tyrone Jackson"],
    ("female", "north_african"): ["Fatima Benali", "Samira Haddad", "Nadia Bouzid"],
    ("male", "north_african"): ["Mohamed Benali", "Karim Haddad", "Youssef Bouzid"],
}
PRONOUNS = {"female": "She", "male": "He"}


def with_identity(text: str, name: str, gender: str) -> str:
    return f"{name}. {PRONOUNS[gender]} is applying for this position. {text}"


def variants(text: str) -> dict[tuple[str, str, str], str]:
    return {(g, o, n): with_identity(text, n, g) for (g, o), names in NAMES.items() for n in names}


def summarize(scores: dict[tuple[str, str, str], np.ndarray]) -> dict[str, float]:
    """Mean score per gender and per origin group, and the largest gap between groups."""
    by_gender, by_origin = {}, {}
    for (g, o, _), s in scores.items():
        by_gender.setdefault(g, []).append(s)
        by_origin.setdefault(o, []).append(s)
    gm = {g: float(np.mean(v)) for g, v in by_gender.items()}
    om = {o: float(np.mean(v)) for o, v in by_origin.items()}
    return {**{f"gender:{k}": v for k, v in gm.items()}, **{f"origin:{k}": v for k, v in om.items()},
            "gender_gap": max(gm.values()) - min(gm.values()),
            "origin_gap": max(om.values()) - min(om.values())}
