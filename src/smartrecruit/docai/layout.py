"""Layout analysis: column detection, reading order and section segmentation."""
import re
from difflib import SequenceMatcher

from smartrecruit.docai.ocr import Line, Page

SECTION_ALIASES = {
    "summary": ["summary", "profile", "about me", "objective", "professional summary"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment history", "work history", "career history"],
    "education": ["education", "academic background", "qualifications", "education and training"],
    "skills": ["skills", "technical skills", "core skills", "competencies", "key skills"],
    "contact": ["contact", "contact details", "personal details"],
    "languages": ["languages"],
    "certifications": ["certifications", "certificates"],
    "projects": ["projects"],
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def heading_type(text: str, threshold: float = 0.85) -> str | None:
    t = _normalize(text)
    if not t or len(t.split()) > 4:
        return None
    best, best_score = None, 0.0
    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            score = SequenceMatcher(None, t, alias).ratio()
            if score > best_score:
                best, best_score = section, score
    return best if best_score >= threshold else None


def find_column_split(page: Page, min_share: float = 0.2) -> float | None:
    """x coordinate of a vertical gutter crossed by (almost) no line, if there is one."""
    lines = page.lines
    if len(lines) < 8:
        return None
    best_x, best_cross = None, None
    for i in range(15, 61):
        x = page.width * i / 100
        left = sum(l.x1 <= x for l in lines)
        right = sum(l.x0 >= x for l in lines)
        cross = len(lines) - left - right
        if min(left, right) < min_share * len(lines):
            continue
        if best_cross is None or cross < best_cross:
            best_x, best_cross = x, cross
    if best_x is None or best_cross > 0.05 * len(lines):
        return None
    return best_x


def reading_order(page: Page) -> list[Line]:
    split = find_column_split(page)
    by_y = lambda l: (round(l.y0), l.x0)  # noqa: E731
    if split is None:
        return sorted(page.lines, key=by_y)
    left = sorted((l for l in page.lines if l.x1 <= split), key=by_y)
    right = sorted((l for l in page.lines if l.x1 > split), key=by_y)
    return left + right


def segment(lines: list[Line]) -> dict[str, list[Line]]:
    """Group lines under the last heading seen. Lines before any heading go to 'header'."""
    sections: dict[str, list[Line]] = {"header": []}
    current = "header"
    for line in lines:
        kind = heading_type(line.text)
        if kind:
            current = kind
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return sections
