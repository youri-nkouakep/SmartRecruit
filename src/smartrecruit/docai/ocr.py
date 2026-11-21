"""Text lines with their geometry, from a digital PDF or from an image (Tesseract OCR)."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract

from smartrecruit.docai.preprocess import preprocess


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool = False


@dataclass
class Page:
    lines: list[Line]
    width: float
    height: float
    source: str = "text layer"


def lines_from_pdf(path: str | Path) -> Page:
    page = fitz.open(path)[0]
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for ln in block.get("lines", []):
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            lines.append(Line(text=" ".join(s["text"].strip() for s in spans),
                              x0=x0, y0=y0, x1=x1, y1=y1,
                              size=max(s["size"] for s in spans),
                              bold=any(s["flags"] & 16 or "Bold" in s["font"] for s in spans)))
    return Page(lines, page.rect.width, page.rect.height)


def pdf_to_image(path: str | Path, dpi: int = 200) -> np.ndarray:
    pix = fitz.open(path)[0].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


def lines_from_image(image: np.ndarray, clean: bool = True) -> Page:
    img = preprocess(image) if clean else image
    data = pytesseract.image_to_data(img, config="--oem 1 --psm 3",
                                     output_type=pytesseract.Output.DATAFRAME)
    data = data[(data["conf"] > 0) & data["text"].astype(str).str.strip().astype(bool)]
    lines = []
    for _, g in data.groupby(["block_num", "par_num", "line_num"], sort=False):
        lines.append(Line(text=" ".join(g["text"].astype(str)),
                          x0=g["left"].min(), y0=g["top"].min(),
                          x1=(g["left"] + g["width"]).max(), y1=(g["top"] + g["height"]).max(),
                          size=float(np.median(g["height"]))))
    return Page(lines, img.shape[1], img.shape[0], "OCR")


def read_document(path: str | Path) -> Page:
    """Digital PDFs use their text layer; images and scanned PDFs go through OCR."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        page = lines_from_pdf(path)
        if sum(len(l.text) for l in page.lines) > 50:
            return page
        return lines_from_image(pdf_to_image(path))
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return lines_from_image(image)


def plain_text(path: str | Path) -> str:
    """Raw text layer without any layout analysis (baseline)."""
    return fitz.open(path)[0].get_text("text", sort=True)

