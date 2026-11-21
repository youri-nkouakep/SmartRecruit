"""Rule-based field extraction on top of the layout analysis."""
import re
from pathlib import Path

from smartrecruit.docai.layout import reading_order, segment
from smartrecruit.docai.ocr import Line, Page, plain_text, read_document
from smartrecruit.schema import Education, Experience, Resume

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\+?\d[\d ().-]{7,}\d")
YEAR = re.compile(r"\b(19|20)\d{2}\b")
SPLIT = re.compile(r"\s*(?:,|\||·|•|;)\s*")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" -–•|·")


def _merge_wrapped(lines: list[Line]) -> list[Line]:
    """Join consecutive lines of the same size that are vertically adjacent (wrapped titles)."""
    merged: list[Line] = []
    for l in lines:
        prev = merged[-1] if merged else None
        if (prev and abs(prev.size - l.size) < 0.5 and -0.5 * l.size <= l.y0 - prev.y1 < 0.8 * l.size
                and abs(prev.x0 - l.x0) < 2 * l.size):
            merged[-1] = Line(f"{prev.text} {l.text}", prev.x0, prev.y0, max(prev.x1, l.x1), l.y1,
                              prev.size, prev.bold)
        else:
            merged.append(l)
    return merged


def _name(header: list[Line], all_lines: list[Line], page: Page) -> str:
    candidates = [l for l in _merge_wrapped(header or all_lines)
                  if l.y0 < 0.3 * page.height and not EMAIL.search(l.text)
                  and not any(c.isdigit() for c in l.text) and 2 <= len(l.text.split()) <= 4]
    return _clean(max(candidates, key=lambda l: l.size).text) if candidates else ""


def _entries(lines: list[Line]) -> list[tuple[str, list[str]]]:
    """(title line, [parts of the detail line]) for every line that carries a year."""
    entries = []
    for i, line in enumerate(lines):
        if YEAR.search(line.text) and i > 0:
            title = lines[i - 1].text
            if title.lstrip().startswith(("-", "•")) or YEAR.search(title):
                continue
            entries.append((_clean(title), [_clean(p) for p in line.text.split("|")]))
    return entries


def extract_from_lines(lines: list[Line], page: Page) -> Resume:
    sections = segment(lines)
    text = "\n".join(l.text for l in lines)
    header = sections.get("header", [])
    name = _name(header, lines, page)

    skills_lines = sections.get("skills", [])
    skills = [_clean(s) for l in skills_lines for s in SPLIT.split(l.text)]
    skills = [s for s in skills if 1 < len(s) <= 40]

    experience = [Experience(title=t, company=p[0], period=p[-1] if len(p) > 1 else "")
                  for t, p in _entries(sections.get("experience", []))]
    education = [Education(degree=t, school=p[0], year=p[-1] if len(p) > 1 else "")
                 for t, p in _entries(sections.get("education", []))]

    headline = ""
    for l in header:
        t = _clean(l.text)
        if t and t != name and not EMAIL.search(t) and not PHONE.search(t):
            headline = t
            break

    email = EMAIL.search(text)
    phone = PHONE.search(text)
    return Resume(name=name, email=email.group(0).lower() if email else "",
                  phone=_clean(phone.group(0)) if phone else "", headline=headline,
                  summary=" ".join(l.text for l in sections.get("summary", [])),
                  skills=skills, experience=experience, education=education)


def extract(path: str | Path) -> Resume:
    page = read_document(path)
    return extract_from_lines(reading_order(page), page)


def extract_plain(path: str | Path) -> Resume:
    """Baseline: same rules on the raw text layer, with no column or geometry information."""
    raw = [t for t in plain_text(path).splitlines() if t.strip()]
    lines = [Line(text=t, x0=0, y0=i, x1=1, y1=i + 1, size=0) for i, t in enumerate(raw)]
    page = Page(lines, 1, max(len(lines), 1) * 4)
    resume = extract_from_lines(lines, page)
    resume.name = _clean(raw[0]) if raw else ""
    return resume
