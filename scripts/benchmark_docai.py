#!/usr/bin/env python3
"""Document AI benchmark on synthetic resumes with known ground truth.

Compares four reading pipelines with the same field extraction rules:
digital PDF + layout analysis, digital PDF raw text (no layout), scanned image + OCR
with OpenCV preprocessing, scanned image + OCR without preprocessing.
"""
import json
from pathlib import Path
import sys

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartrecruit.docai.evaluate import f1_table, field_counts  # noqa: E402
from smartrecruit.docai.extract import extract, extract_from_lines, extract_plain  # noqa: E402
from smartrecruit.docai.layout import reading_order  # noqa: E402
from smartrecruit.docai.ocr import lines_from_image  # noqa: E402
from smartrecruit.docai.synth import generate  # noqa: E402
from smartrecruit.schema import Resume  # noqa: E402

DATA = ROOT / "data" / "synthetic_resumes"
MODES = {
    "PDF + layout analysis": "pdf_layout",
    "PDF raw text (no layout)": "pdf_plain",
    "Scan + OpenCV preprocessing + OCR": "scan_clean",
    "Scan + OCR (no preprocessing)": "scan_raw",
}


def main(n: int = 50):
    if not (DATA / "index.json").exists():
        generate(DATA, n=n, seed=2026)
    index = json.loads((DATA / "index.json").read_text())
    counts = {m: {"all": [], "single": [], "two_column": []} for m in MODES.values()}
    for item in index:
        gold = Resume.model_validate_json((DATA / f"{item['id']}.json").read_text())
        pdf, scan = DATA / f"{item['id']}.pdf", cv2.imread(str(DATA / f"{item['id']}_scan.jpg"), 0)
        preds = {"pdf_layout": extract(pdf), "pdf_plain": extract_plain(pdf)}
        for mode, clean in (("scan_clean", True), ("scan_raw", False)):
            page = lines_from_image(scan, clean=clean)
            preds[mode] = extract_from_lines(reading_order(page), page)
        for mode, pred in preds.items():
            c = field_counts(pred, gold)
            counts[mode]["all"].append(c)
            counts[mode][item["layout"]].append(c)

    rows = {name: f1_table(counts[m]["all"]) for name, m in MODES.items()}
    table = pd.DataFrame(rows).T
    for layout in ("single", "two_column"):
        table[f"overall_{layout}"] = [f1_table(counts[m][layout])["overall"] for m in MODES.values()]
    table.round(3).to_csv(ROOT / "results" / "docai_benchmark.csv")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
