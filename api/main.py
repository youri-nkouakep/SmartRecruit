"""SmartRecruit API. Run: uvicorn api.main:app --reload (docs at /docs).

The React frontend is served from web/dist when it has been built.
"""
import json
import os
from pathlib import Path
import sys

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from api.store import Store  # noqa: E402
from smartrecruit.schema import Resume  # noqa: E402
from smartrecruit.service import SmartRecruit  # noqa: E402

RESULTS = ROOT / "results"
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="SmartRecruit API", version="2.0.0",
              description="Resume reading (OCR + layout), classification, offer matching and grounded "
                          "LLM explanations.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"],
                   allow_headers=["*"])

service = SmartRecruit(llm_backend=os.getenv("SMARTRECRUIT_LLM", "transformers"),
                       llm_model=os.getenv("SMARTRECRUIT_LLM_MODEL"))
store = Store(Path(os.getenv("SMARTRECRUIT_DB", ROOT / "var" / "smartrecruit.db")))
store.seed_offers(json.loads((ROOT / "data" / "job_offers.json").read_text()))


class OfferIn(BaseModel):
    title: str = Field(..., min_length=3, max_length=120)
    text: str = Field(..., min_length=30)


class ExplainIn(BaseModel):
    offer_id: int
    evidence: list[dict] = Field(..., min_length=1, max_length=8)


def _process(cid: int, data: bytes, filename: str):
    try:
        parsed = service.parse(data, filename)
        prediction = service.classify(parsed["resume"].to_text())
        store.finish_candidate(cid, parsed, prediction, parsed["image"])
    except Exception as e:
        store.fail_candidate(cid, str(e))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/overview")
def overview():
    cands = store.candidates()
    ready = [c for c in cands if c["status"] == "ready"]
    families = {}
    for c in ready:
        label = c["prediction"]["top"][0]["label"]
        families[label] = families.get(label, 0) + 1
    return {"offers": len(store.offers()), "candidates": len(cands), "ready": len(ready),
            "processing": sum(c["status"] == "processing" for c in cands),
            "library": 2483, "families": families, "recent": ready[:5]}


@app.get("/api/benchmarks")
def benchmarks():
    import pandas as pd

    def csv(name):
        path = RESULTS / name
        return pd.read_csv(path, index_col=0).reset_index(names="row").to_dict("records") if path.exists() else []

    def js(name):
        path = RESULTS / name
        return json.loads(path.read_text()) if path.exists() else {}

    llm = js("llm_explanations.json")
    return {"docai": csv("docai_benchmark.csv"), "classification": js("classification.json"),
            "retrieval": csv("retrieval.csv"), "bias": js("bias_audit.json"),
            "llm": llm.get("summary", {}), "per_class": csv("tfidf_per_class.csv")}


@app.get("/api/offers")
def list_offers():
    return store.offers()


@app.post("/api/offers", status_code=201)
def create_offer(body: OfferIn):
    return store.offer(store.add_offer(body.title, body.text))


@app.get("/api/offers/{offer_id}")
def get_offer(offer_id: int):
    offer = store.offer(offer_id)
    if not offer:
        raise HTTPException(404, "Offer not found")
    return offer


@app.delete("/api/offers/{offer_id}", status_code=204)
def delete_offer(offer_id: int):
    store.delete_offer(offer_id)


@app.post("/api/candidates", status_code=202)
async def upload(background: BackgroundTasks, files: list[UploadFile] = File(...)):
    created = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in ALLOWED:
            raise HTTPException(415, f"Unsupported file type {suffix}")
        data = await f.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename} is larger than 10 MB")
        cid = store.new_candidate(f.filename)
        background.add_task(_process, cid, data, f.filename)
        created.append(cid)
    return {"ids": created}


@app.get("/api/candidates")
def list_candidates():
    return store.candidates()


@app.get("/api/candidates/{cid}")
def get_candidate(cid: int):
    c = store.candidate(cid)
    if not c:
        raise HTTPException(404, "Candidate not found")
    return c


@app.put("/api/candidates/{cid}/resume")
def update_resume(cid: int, resume: Resume):
    if not store.candidate(cid):
        raise HTTPException(404, "Candidate not found")
    store.update_resume(cid, resume.model_dump_json(), service.classify(resume.to_text()))
    return store.candidate(cid)


@app.get("/api/candidates/{cid}/image")
def candidate_image(cid: int):
    path = store.files / f"{cid}.png"
    if not path.exists():
        raise HTTPException(404, "No image")
    return FileResponse(path, media_type="image/png")


@app.delete("/api/candidates/{cid}", status_code=204)
def delete_candidate(cid: int):
    store.delete_candidate(cid)


@app.get("/api/offers/{offer_id}/match")
def match(offer_id: int, pool: str = "uploaded", k: int = 10):
    offer = get_offer(offer_id)
    if pool == "library":
        return service.match_library(offer["text"], k)
    ready = [c for c in store.candidates() if c["status"] == "ready"]
    items = [(f"candidate-{c['id']}", c["resume"].get("headline") or c["filename"],
              Resume.model_validate(c["resume"]).to_text()) for c in ready]
    hits = service.match_texts(offer["text"], items, k)
    by_ref = {f"candidate-{c['id']}": c for c in ready}
    for h in hits:
        c = by_ref[h["ref"]]
        h.update(candidate_id=c["id"], name=c["resume"].get("name"),
                 category=c["prediction"]["top"][0]["label"])
    return hits


@app.post("/api/explain")
def explain(body: ExplainIn):
    offer = get_offer(body.offer_id)

    def events():
        answer = ""
        try:
            for token in service.explain_stream(offer["text"], body.evidence):
                if not token:
                    continue
                answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True, **service.citation_check(answer, len(body.evidence))})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        file = DIST / path
        return FileResponse(file if path and file.is_file() else DIST / "index.html")
