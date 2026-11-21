#!/usr/bin/env python3
"""RAG explanations: for each offer, the top candidate is explained by a local LLM from
extractive evidence (closest resume sentence per requirement). We check that every bullet
cites a valid evidence id."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartrecruit.retrieval.explain import LLM_MODEL  # noqa: E402
from smartrecruit.service import SmartRecruit  # noqa: E402


def main():
    service = SmartRecruit()
    offers = json.loads((ROOT / "data" / "job_offers.json").read_text())
    hits = [(o, service.match(o["text"], k=1)[0]) for o in offers]
    del service.indexes
    rows, cited, total = [], 0, 0
    for offer, hit in hits:
        t = time.perf_counter()
        out = service.justify(offer["text"], hit["evidence"])
        cited += out["cited_bullets"]
        total += out["bullets"]
        rows.append({"offer": offer["category"], "resume_id": hit["id"], "resume_category": hit["category"],
                     "evidence": hit["evidence"], "answer": out["answer"],
                     "cited_bullets": out["cited_bullets"], "bullets": out["bullets"],
                     "seconds": round(time.perf_counter() - t, 1)})
        print(offer["category"], out["cited_bullets"], "/", out["bullets"], flush=True)
    summary = {"model": LLM_MODEL, "offers": len(rows), "bullets": total, "cited_bullets": cited,
               "citation_rate": cited / total if total else 0.0,
               "mean_seconds": sum(r["seconds"] for r in rows) / len(rows)}
    (ROOT / "results" / "llm_explanations.json").write_text(json.dumps({"summary": summary, "examples": rows},
                                                                       indent=2))
    print(summary)


if __name__ == "__main__":
    main()
