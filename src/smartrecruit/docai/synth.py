"""Synthetic resumes with ground truth, rendered as digital PDFs and simulated scans.

Two layouts are used: a single-column resume and a two-column resume with a
sidebar, where naive text extraction mixes the columns.
"""
import io
import json
import random
from pathlib import Path

import cv2
import fitz
import numpy as np
from faker import Faker
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from smartrecruit.schema import Education, Experience, Resume

PROFILES = {
    "Data Scientist": ["Python", "SQL", "Pandas", "Scikit-learn", "PyTorch", "Statistics",
                       "Machine Learning", "Spark", "Tableau", "A/B Testing", "NLP", "Docker"],
    "Software Engineer": ["Java", "Python", "C++", "Git", "Docker", "Kubernetes", "REST APIs",
                          "PostgreSQL", "AWS", "CI/CD", "Microservices", "Linux"],
    "Accountant": ["IFRS", "Bookkeeping", "Excel", "SAP", "Tax Preparation", "Auditing",
                   "Financial Reporting", "Payroll", "QuickBooks", "Budgeting"],
    "Sales Manager": ["CRM", "Salesforce", "Negotiation", "Lead Generation", "Forecasting",
                      "Key Accounts", "B2B Sales", "Team Leadership", "HubSpot", "Pricing"],
    "HR Specialist": ["Recruitment", "Onboarding", "Labor Law", "Workday", "Payroll",
                      "Employee Relations", "Talent Management", "HRIS", "Training"],
    "Civil Engineer": ["AutoCAD", "Revit", "Structural Analysis", "Project Management",
                       "Eurocodes", "Site Supervision", "Cost Estimation", "BIM", "Surveying"],
}
TITLES = {
    "Data Scientist": ["Data Scientist", "Data Analyst", "Machine Learning Engineer"],
    "Software Engineer": ["Software Engineer", "Backend Developer", "DevOps Engineer"],
    "Accountant": ["Accountant", "Junior Auditor", "Financial Analyst"],
    "Sales Manager": ["Sales Manager", "Account Executive", "Business Developer"],
    "HR Specialist": ["HR Specialist", "Recruiter", "HR Assistant"],
    "Civil Engineer": ["Civil Engineer", "Site Engineer", "Structural Engineer"],
}
DEGREES = {
    "Data Scientist": ["MSc Data Science", "BSc Mathematics", "MSc Artificial Intelligence"],
    "Software Engineer": ["BSc Computer Science", "MSc Software Engineering"],
    "Accountant": ["BSc Accounting", "Master in Finance"],
    "Sales Manager": ["Bachelor in Business", "MBA"],
    "HR Specialist": ["Bachelor in Psychology", "Master in Human Resources"],
    "Civil Engineer": ["BEng Civil Engineering", "MSc Structural Engineering"],
}
HEADINGS = {
    "summary": ["SUMMARY", "PROFILE", "About Me"],
    "experience": ["EXPERIENCE", "Work Experience", "PROFESSIONAL EXPERIENCE"],
    "education": ["EDUCATION", "Education", "ACADEMIC BACKGROUND"],
    "skills": ["SKILLS", "Technical Skills", "CORE SKILLS"],
    "contact": ["CONTACT", "Contact"],
}


def make_resume(rng: random.Random, fake: Faker) -> tuple[Resume, str]:
    job = rng.choice(list(PROFILES))
    first, last = fake.first_name(), fake.last_name()
    year = rng.randint(2014, 2020)
    n_jobs = rng.randint(2, 3)
    starts = sorted((year + 2 * k for k in range(n_jobs)), reverse=True)
    experience = [Experience(title=rng.choice(TITLES[job]),
                             company=fake.company().replace(",", ""),
                             period=f"{start} - {'Present' if k == 0 else start + 2}")
                  for k, start in enumerate(starts)]
    education = [Education(degree=d, school=f"University of {fake.city()}", year=str(year - 2 * k))
                 for k, d in enumerate(rng.sample(DEGREES[job], k=min(2, len(DEGREES[job]))))]
    resume = Resume(
        name=f"{first} {last}",
        email=f"{first}.{last}@{rng.choice(['gmail.com', 'outlook.com', 'mail.com'])}".lower(),
        phone=f"+33 6 {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}",
        headline=experience[0].title,
        summary=f"{experience[0].title} with {2026 - year} years of experience. " + fake.sentence(nb_words=14),
        skills=rng.sample(PROFILES[job], k=rng.randint(5, 8)),
        experience=experience,
        education=education,
    )
    return resume, job


class _Writer:
    def __init__(self, c, x, y, width):
        self.c, self.x, self.y, self.width = c, x, y, width

    def text(self, s, size=10, bold=False, color="#222222", gap=4):
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        self.c.setFillColor(HexColor(color))
        font = "Helvetica-Bold" if bold else "Helvetica"
        lines, line = [], ""
        for w in s.split():
            trial = f"{line} {w}".strip()
            if self.c.stringWidth(trial, font, size) > self.width and line:
                lines.append(line)
                line = w
            else:
                line = trial
        lines.append(line)
        for i, ln in enumerate(lines):
            self.c.drawString(self.x, self.y, ln)
            self.y -= size + (gap if i == len(lines) - 1 else 3)

    def heading(self, s, rule=True):
        self.y -= 6
        self.text(s, size=12, bold=True, color="#1d3557", gap=3)
        if rule:
            self.c.setStrokeColor(HexColor("#1d3557"))
            self.c.line(self.x, self.y + 9, self.x + self.width, self.y + 9)
        self.y -= 4


def render_pdf(resume: Resume, layout: str, rng: random.Random) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    h = {k: rng.choice(v) for k, v in HEADINGS.items()}
    bullet = rng.choice([", ", " | ", " · "])

    def experience_block(w):
        for e in resume.experience:
            w.text(e.title, size=10.5, bold=True, gap=2)
            w.text(f"{e.company} | {e.period}", size=9.5, color="#555555", gap=2)
            w.text("- " + Faker().sentence(nb_words=12), size=9.5, gap=8)

    def education_block(w):
        for e in resume.education:
            w.text(e.degree, size=10.5, bold=True, gap=2)
            w.text(f"{e.school} | {e.year}", size=9.5, color="#555555", gap=8)

    if layout == "single":
        w = _Writer(c, 50, H - 60, W - 100)
        w.text(resume.name, size=22, bold=True, color="#1d3557", gap=6)
        w.text(resume.headline, size=12, color="#457b9d", gap=6)
        w.text(f"{resume.email}  |  {resume.phone}", size=9.5, gap=10)
        w.heading(h["summary"])
        w.text(resume.summary, size=10)
        w.heading(h["experience"])
        experience_block(w)
        w.heading(h["education"])
        education_block(w)
        w.heading(h["skills"])
        w.text(bullet.join(resume.skills), size=10)
    else:
        side = 185
        c.setFillColor(HexColor("#eef2f7"))
        c.rect(0, 0, side, H, stroke=0, fill=1)
        left = _Writer(c, 25, H - 60, side - 45)
        left.text(resume.name, size=18, bold=True, color="#1d3557", gap=6)
        left.text(resume.headline, size=11, color="#457b9d", gap=12)
        left.heading(h["contact"], rule=False)
        left.text(resume.email, size=8.5, gap=4)
        left.text(resume.phone, size=8.5, gap=10)
        left.heading(h["skills"], rule=False)
        for s in resume.skills:
            left.text(s, size=9.5, gap=4)
        right = _Writer(c, side + 30, H - 60, W - side - 60)
        right.heading(h["summary"])
        right.text(resume.summary, size=10)
        right.heading(h["experience"])
        experience_block(right)
        right.heading(h["education"])
        education_block(right)
    c.showPage()
    c.save()
    return buf.getvalue()


def simulate_scan(pdf_bytes: bytes, rng: random.Random, dpi: int = 150) -> np.ndarray:
    """Rasterize the first page and degrade it like an office scan or phone photo."""
    page = fitz.open(stream=pdf_bytes, filetype="pdf")[0]
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    h, w = img.shape
    angle = rng.uniform(-3.0, 3.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    shade = np.linspace(rng.uniform(0.75, 0.95), 1.0, w)[None, :]
    img = (img.astype(np.float32) * shade).clip(0, 255)
    img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.4, 1.0))
    img = img + np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(6, 14), img.shape)
    img = img.clip(0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 60)])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


def generate(out_dir: Path, n: int = 50, seed: int = 0) -> None:
    rng = random.Random(seed)
    fake = Faker("en_US")
    Faker.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for i in range(n):
        resume, job = make_resume(rng, fake)
        layout = "single" if i % 2 == 0 else "two_column"
        pdf = render_pdf(resume, layout, rng)
        stem = f"cv_{i:03d}"
        (out_dir / f"{stem}.pdf").write_bytes(pdf)
        cv2.imwrite(str(out_dir / f"{stem}_scan.jpg"), simulate_scan(pdf, rng))
        (out_dir / f"{stem}.json").write_text(resume.model_dump_json(indent=2))
        meta.append({"id": stem, "layout": layout, "job": job})
    (out_dir / "index.json").write_text(json.dumps(meta, indent=2))
