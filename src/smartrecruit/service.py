"""Single entry point used by the API. Models load lazily on first use."""
import tempfile
import threading
import time
from functools import cached_property
from pathlib import Path
from typing import Iterator

import cv2
import fitz
import joblib
import numpy as np

from smartrecruit.docai.extract import extract_from_lines
from smartrecruit.docai.layout import heading_type, reading_order, segment
from smartrecruit.docai.ocr import read_document
from smartrecruit.nlp.classify import explain as explain_terms
from smartrecruit.nlp.data import load
from smartrecruit.retrieval.explain import LLM_MODEL, LocalLLM, OllamaLLM, build_prompt, evidence, grounding
from smartrecruit.retrieval.search import BI_ENCODER, BM25Index, DenseIndex, Reranker

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def rrf(*rankings, k=60):
    scores = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + r + 1)
    return np.array(sorted(scores, key=scores.get, reverse=True))


def relevance(logit: float) -> float:
    return float(1.0 / (1.0 + np.exp(-logit)))


def _png(image: np.ndarray, max_width: int = 1100) -> bytes:
    if image.shape[1] > max_width:
        f = max_width / image.shape[1]
        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", image)[1].tobytes()


class SmartRecruit:
    def __init__(self, llm_backend: str = "transformers", llm_model: str | None = None):
        self.llm_backend, self.llm_model = llm_backend, llm_model
        self._lock = threading.Lock()

    def parse(self, data: bytes, filename: str) -> dict:
        """Structured resume, the page image and every text line with its box and section."""
        suffix = Path(filename).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            path = Path(f.name)
        t = time.perf_counter()
        page = read_document(path)
        ordered = reading_order(page)
        resume = extract_from_lines(ordered, page)
        section_of = {id(l): name for name, ls in segment(ordered).items() for l in ls}
        if page.image is not None:
            image = page.image
        else:
            pix = fitz.open(path)[0].get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csGRAY)
            image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        path.unlink(missing_ok=True)
        lines = [{"text": l.text,
                  "box": [l.x0 / page.width, l.y0 / page.height, l.x1 / page.width, l.y1 / page.height],
                  "section": "heading" if heading_type(l.text) else section_of.get(id(l), "header")}
                 for l in ordered]
        return {"resume": resume, "source": page.source, "seconds": time.perf_counter() - t,
                "image": _png(image), "lines": lines}

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
        return {"model": name, "top": [{"label": classes[i], "probability": float(proba[i])} for i in order],
                "key_terms": [{"term": t, "weight": w} for t, w in explain_terms(self.tfidf, text)]}

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
    def reranker(self):
        return Reranker()

    @cached_property
    def library(self):
        emb_path = MODELS / "resume_embeddings.npy"
        texts = list(self.corpus["text"])
        emb = np.load(emb_path) if emb_path.exists() else self.encoder.encode(texts, normalize_embeddings=True)
        return BM25Index(texts), DenseIndex(emb, self.encoder)

    def _rank(self, offer: str, texts: list[str], bm25, dense, k: int, candidates: int):
        pool = rrf(bm25.search(offer, candidates)[0], dense.search(offer, candidates)[0])[:candidates]
        return self.reranker.rerank(offer, texts, pool, min(k, len(pool)))

    def match_library(self, offer: str, k: int = 10, candidates: int = 50) -> list[dict]:
        with self._lock:
            bm25, dense = self.library
            texts = list(self.corpus["text"])
            idx, scores = self._rank(offer, texts, bm25, dense, k, candidates)
            return [{"ref": f"library-{int(self.corpus.at[i, 'id'])}", "title": self.corpus.at[i, "label"],
                     "category": self.corpus.at[i, "label"], "score": float(s), "relevance": relevance(s),
                     "evidence": evidence(offer, texts[i], self.encoder)} for i, s in zip(idx, scores)]

    def match_texts(self, offer: str, items: list[tuple[str, str, str]], k: int = 10) -> list[dict]:
        """Rank uploaded candidates given as (ref, title, text). Identity fields are not in the text."""
        if not items:
            return []
        with self._lock:
            texts = [t for _, _, t in items]
            emb = self.encoder.encode(texts, normalize_embeddings=True)
            idx, scores = self._rank(offer, texts, BM25Index(texts), DenseIndex(emb, self.encoder),
                                     k, len(texts))
            return [{"ref": items[i][0], "title": items[i][1], "category": None, "score": float(s), "relevance": relevance(s),
                     "evidence": evidence(offer, texts[i], self.encoder)} for i, s in zip(idx, scores)]

    @cached_property
    def llm(self):
        if self.llm_backend == "ollama":
            return OllamaLLM(self.llm_model or "qwen2.5:3b")
        return LocalLLM(self.llm_model or LLM_MODEL)

    def explain_stream(self, offer: str, items: list[dict]) -> Iterator[str]:
        yield from self.llm.stream(build_prompt(offer, items))

    def citation_check(self, answer: str, n_evidence: int) -> dict:
        cited, total = grounding(answer, n_evidence)
        return {"cited_bullets": cited, "bullets": total}
