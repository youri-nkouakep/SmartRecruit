#!/usr/bin/env python3
"""Train and compare the three classifiers on the same split. Results go to results/."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartrecruit.nlp.classify import (EMBEDDING_MODEL, FINETUNE_MODEL, EmbeddingClassifier,  # noqa: E402
                                       FineTunedClassifier, Result, tfidf_model, timed, top_terms)
from smartrecruit.nlp.data import load, split  # noqa: E402

RESULTS, MODELS = ROOT / "results", ROOT / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["tfidf", "embeddings", "finetune"], nargs="*")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()
    steps = args.only or ["tfidf", "embeddings", "finetune"]
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    df = load()
    train, test = split(df)
    test[["id", "label"]].to_csv(RESULTS / "test_ids.csv", index=False)
    scores_path = RESULTS / "classification.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}

    if "tfidf" in steps:
        pipe = tfidf_model()
        _, t_fit = timed(pipe.fit, train["text"], train["label"])
        pred, t_pred = timed(pipe.predict, test["text"])
        scores["TF-IDF + logistic regression"] = vars(Result.of("tfidf", test["label"], pred, t_fit, t_pred, len(test)))
        joblib.dump(pipe, MODELS / "tfidf_logreg.joblib")
        (RESULTS / "top_terms.json").write_text(json.dumps(top_terms(pipe), indent=1))
        report = classification_report(test["label"], pred, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.round(3).to_csv(RESULTS / "tfidf_per_class.csv")

    if "embeddings" in steps:
        clf = EmbeddingClassifier()
        emb_path, timing_path = MODELS / "resume_embeddings.npy", MODELS / "encode_seconds.txt"
        if emb_path.exists() and timing_path.exists():
            emb, t_enc = np.load(emb_path), float(timing_path.read_text())
        else:
            emb, t_enc = timed(clf.encode, df["text"])
            np.save(emb_path, emb)
            timing_path.write_text(str(t_enc))
        df[["id", "label"]].to_csv(MODELS / "resume_index.csv", index=False)
        _, t_fit = timed(clf.fit, None, train["label"], embeddings=emb[train.index])
        pred, t_pred = timed(clf.predict, embeddings=emb[test.index])
        per_doc = t_enc / len(df)
        r = Result.of("emb", test["label"], pred, t_fit + per_doc * len(train), t_pred + per_doc * len(test), len(test))
        scores[f"Embeddings ({EMBEDDING_MODEL.split('/')[-1]}) + logistic regression"] = vars(r)
        joblib.dump(clf.clf, MODELS / "embedding_logreg.joblib")

    if "finetune" in steps:
        ft = FineTunedClassifier()
        _, t_fit = timed(ft.fit, list(train["text"]), list(train["label"]), epochs=args.epochs)
        pred, t_pred = timed(ft.predict, list(test["text"]))
        scores[f"Fine-tuned {FINETUNE_MODEL} (PyTorch)"] = vars(Result.of("ft", test["label"], pred, t_fit, t_pred, len(test)))
        ft.save(MODELS / "distilbert_resume")
        labels = sorted(test["label"].unique())
        cm = confusion_matrix(test["label"], pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(RESULTS / "finetune_confusion.csv")

    scores_path.write_text(json.dumps(scores, indent=2))
    print(pd.DataFrame(scores).T[["accuracy", "f1_macro", "train_seconds", "predict_ms"]].round(3))


if __name__ == "__main__":
    main()
