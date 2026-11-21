"""Single entry point used by the API and the Streamlit app. Models load lazily."""
import tempfile
import time
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np

from smartrecruit.docai.extract import extract_from_lines
from smartrecruit.docai.layout import reading_order
from smartrecruit.docai.ocr import read_document
from smartrecruit.nlp.classify import explain as explain_terms
from smartrecruit.nlp.data import load
from smartrecruit.retrieval.explain import LLM_MODEL, LocalLLM, OllamaLLM, build_prompt, evidence, grounding
from smartrecruit.retrieval.search import BI_ENCODER, BM25Index, DenseIndex, Reranker

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def _rrf(*rankings, k=60):
    scores = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + r + 1)
    return np.array(sorted(scores, key=scores.get, reverse=True))


class SmartRecruit:
    def __init__(self, llm_backend: str = "transformers", llm_model: str | None = None):
        self.llm_backend, self.llm_model = llm_backend, llm_model

    def parse(self, data: bytes, filename: str) -> dict:
        suffix = Path(filename).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            path = Path(f.name)
        t = time.perf_counter()
        page = read_document(path)
        resume = extract_from_lines(reading_order(page), page)
        path.unlink(missing_ok=True)
        return {"resume": resume, "seconds": time.perf_counter() - t, "source": page.source}

    @cached_property
    def tfidf(self):
        return joblib.load(MODELS / "tfidf_logreg.joblib")

    @cached_property
    def finetuned(self):
        path = MODELS / "distilbert_resume"
        if not path.exists():
            return None
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        return (AutoTokenizer.from_pretrained(path),
                AutoModelForSequenceClassification.from_pretrained(path).eval(),
                joblib.load(path / "classes.joblib"), torch)

    def classify(self, text: str, top: int = 3) -> dict:
        if self.finetuned:
            tok, model, classes, torch = self.finetuned
            with torch.no_grad():
                enc = tok([text], truncation=True, max_length=256, return_tensors="pt")
                proba = torch.softmax(model(**enc).logits, -1)[0].numpy()
            name = "DistilBERT (fine-tuned)"
        else:
            proba, classes, name = self.tfidf.predict_proba([text])[0], self.tfidf.classes_, "TF-IDF"
        order = np.argsort(proba)[::-1][:top]
        return {"model": name, "top": [(classes[i], float(proba[i])) for i in order],
                "key_terms": explain_terms(self.tfidf, text)}

    @cached_property
    def corpus(self):
        return load()

    @cached_property
    def encoder(self):
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(BI_ENCODER, device="cpu")
        enc.max_seq_length = 512
        return enc

    @cached_property
    def indexes(self):
        emb_path = MODELS / "resume_embeddings.npy"
        texts = list(self.corpus["text"])
        emb = np.load(emb_path) if emb_path.exists() else self.encoder.encode(texts, normalize_embeddings=True)
        return BM25Index(texts), DenseIndex(emb, self.encoder), Reranker()

    def match(self, offer: str, k: int = 5, candidates: int = 50) -> list[dict]:
        bm25, dense, reranker = self.indexes
        texts = list(self.corpus["text"])
        pool = _rrf(bm25.search(offer, candidates)[0], dense.search(offer, candidates)[0])[:candidates]
        idx, scores = reranker.rerank(offer, texts, pool, k)
        return [{"row": int(i), "id": int(self.corpus.at[i, "id"]), "category": self.corpus.at[i, "label"],
                 "score": float(s),
                 "evidence": self.evidence(offer, texts[i])}
                for i, s in zip(idx, scores)]

    @cached_property
    def llm(self):
        if self.llm_backend == "ollama":
            return OllamaLLM(self.llm_model or "qwen2.5:3b")
        return LocalLLM(self.llm_model or LLM_MODEL)

    def evidence(self, offer: str, resume_text: str) -> list[dict]:
        return evidence(offer, resume_text, self.encoder)

    def justify(self, offer: str, items: list[dict]) -> dict:
        t = time.perf_counter()
        answer = self.llm.generate(build_prompt(offer, items))
        cited, total = grounding(answer, len(items))
        return {"answer": answer, "cited_bullets": cited, "bullets": total,
                "seconds": time.perf_counter() - t}
