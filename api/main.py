"""REST API. Run: uvicorn api.main:app --reload"""
import json
from pathlib import Path
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartrecruit.schema import Resume  # noqa: E402
from smartrecruit.service import SmartRecruit  # noqa: E402

app = FastAPI(title="SmartRecruit API", version="1.0.0",
              description="Resume parsing (OCR + layout), classification and offer matching.")
service = SmartRecruit()
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class TextIn(BaseModel):
    text: str = Field(..., min_length=50)


class MatchIn(BaseModel):
    offer: str = Field(..., min_length=30)
    k: int = Field(5, ge=1, le=20)


class ParseOut(BaseModel):
    resume: Resume
    source: str
    seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parse", response_model=ParseOut)
async def parse(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type {suffix}")
    return service.parse(await file.read(), file.filename)


@app.post("/classify")
def classify(body: TextIn):
    return service.classify(body.text)


@app.post("/match")
def match(body: MatchIn):
    return service.match(body.offer, body.k)


@app.get("/offers")
def offers():
    return json.loads((Path(__file__).resolve().parents[1] / "data" / "job_offers.json").read_text())
