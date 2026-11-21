#!/usr/bin/env python3
"""Counterfactual audit: 100 test resumes x 18 names (2 genders x 3 origins x 3 names).

For each model we measure how often the decision changes and how much the score moves
when only the name and pronoun change. Ideally both are zero."""
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartrecruit.fairness.audit import summarize, variants  # noqa: E402
from smartrecruit.nlp.data import load, split  # noqa: E402
from smartrecruit.retrieval.search import BI_ENCODER, QUERY_PREFIX, Reranker  # noqa: E402

N = 100


def classifier_audit(predict_proba, classes, base_texts, var_texts, keys):
    base = predict_proba(base_texts)
    base_pred = base.argmax(1)
    flips, deltas = 0, {k: [] for k in keys}
    probs = predict_proba([t for vs in var_texts for t in vs]).reshape(len(base_texts), len(keys), -1)
    for i in range(len(base_texts)):
        for j, k in enumerate(keys):
            flips += int(probs[i, j].argmax() != base_pred[i])
            deltas[k].append(probs[i, j, base_pred[i]])
    return flips / (len(base_texts) * len(keys)), {k: np.array(v) for k, v in deltas.items()}


def main():
    df = load()
    _, test = split(df)
    sample = test.groupby("label").head(5).head(N)
    offers = {o["category"]: o["text"] for o in json.loads((ROOT / "data" / "job_offers.json").read_text())}
    texts = list(sample["text"])
    var = [variants(t) for t in texts]
    keys = list(var[0])
    var_texts = [[v[k] for k in keys] for v in var]
    report = {}

    tfidf = joblib.load(ROOT / "models" / "tfidf_logreg.joblib")
    flip, scores = classifier_audit(tfidf.predict_proba, tfidf.classes_, texts, var_texts, keys)
    report["TF-IDF classifier"] = {"decision_flip_rate": flip, **summarize(scores)}

    ft_path = ROOT / "models" / "distilbert_resume"
    if ft_path.exists():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ft_path)
        model = AutoModelForSequenceClassification.from_pretrained(ft_path).eval()

        @torch.no_grad()
        def proba(batch):
            out = []
            for i in range(0, len(batch), 32):
                enc = tok(batch[i:i + 32], truncation=True, max_length=256, padding=True, return_tensors="pt")
                out.append(torch.softmax(model(**enc).logits, -1))
            return torch.cat(out).numpy()

        flip, scores = classifier_audit(proba, None, texts, var_texts, keys)
        report["Fine-tuned DistilBERT classifier"] = {"decision_flip_rate": flip, **summarize(scores)}

    encoder = SentenceTransformer(BI_ENCODER, device="cpu")
    encoder.max_seq_length = 512
    q = {c: encoder.encode([QUERY_PREFIX + t], normalize_embeddings=True)[0] for c, t in offers.items()}
    base = encoder.encode(texts, batch_size=16, normalize_embeddings=True)
    allv = encoder.encode([t for vs in var_texts for t in vs], batch_size=16,
                          normalize_embeddings=True).reshape(len(texts), len(keys), -1)
    qs = np.stack([q[c] for c in sample["label"]])
    base_s = (base * qs).sum(1)
    dense = {k: (allv[:, j] * qs).sum(1) - base_s for j, k in enumerate(keys)}
    report["Dense retrieval score (cosine)"] = summarize(dense)

    rr = Reranker()
    pairs_base = rr.model.predict([(offers[c], t) for c, t in zip(sample["label"], texts)], batch_size=16)
    pairs = rr.model.predict([(offers[c], vs[j]) for c, vs in zip(sample["label"], var_texts)
                              for j in range(len(keys))], batch_size=16).reshape(len(texts), len(keys))
    report["Cross-encoder score"] = summarize({k: pairs[:, j] - pairs_base for j, k in enumerate(keys)})

    (ROOT / "results" / "bias_audit.json").write_text(json.dumps(report, indent=2))
    print(pd.DataFrame(report).T.round(4).to_string())


if __name__ == "__main__":
    main()
