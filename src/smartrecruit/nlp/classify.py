"""Resume classification: TF-IDF baseline, frozen embeddings, fine-tuned transformer."""
import time
from dataclasses import dataclass

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FINETUNE_MODEL = "distilbert-base-uncased"


@dataclass
class Result:
    model: str
    accuracy: float
    f1_macro: float
    train_seconds: float
    predict_ms: float

    @staticmethod
    def of(model, y_true, y_pred, train_s, predict_s, n):
        return Result(model, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro"),
                      train_s, 1000 * predict_s / n)


def tfidf_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.8,
                                  stop_words="english", max_features=50000)),
        ("clf", LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
    ])


def top_terms(pipe: Pipeline, n: int = 8) -> dict[str, list[str]]:
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    clf = pipe.named_steps["clf"]
    return {c: vocab[np.argsort(w)[::-1][:n]].tolist() for c, w in zip(clf.classes_, clf.coef_)}


def explain(pipe: Pipeline, text: str, n: int = 8) -> list[tuple[str, float]]:
    """Exact per-term contributions to the predicted class logit (linear model, so this is
    what SHAP's LinearExplainer computes up to the baseline term)."""
    x = pipe.named_steps["tfidf"].transform([text])
    clf = pipe.named_steps["clf"]
    k = int(np.argmax(clf.decision_function(x)))
    vocab = pipe.named_steps["tfidf"].get_feature_names_out()
    contrib = x.multiply(clf.coef_[k]).tocsr()
    order = np.argsort(contrib.data)[::-1][:n]
    return [(vocab[contrib.indices[i]], float(contrib.data[i])) for i in order]


class EmbeddingClassifier:
    def __init__(self, model_name: str = EMBEDDING_MODEL, max_seq_length: int = 512):
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(model_name, device="cpu")
        self.encoder.max_seq_length = max_seq_length
        self.clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced")

    def encode(self, texts):
        return self.encoder.encode(list(texts), batch_size=16, normalize_embeddings=True,
                                   show_progress_bar=False)

    def fit(self, texts, labels, embeddings=None):
        self.clf.fit(self.encode(texts) if embeddings is None else embeddings, labels)
        return self

    def predict(self, texts=None, embeddings=None):
        return self.clf.predict(self.encode(texts) if embeddings is None else embeddings)


class FineTunedClassifier:
    def __init__(self, model_name: str = FINETUNE_MODEL, max_length: int = 256, seed: int = 42):
        self.model_name, self.max_length, self.seed = model_name, max_length, seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _batches(self, texts, labels=None, batch_size=16, shuffle=False):
        idx = list(range(len(texts)))
        loader = DataLoader(idx, batch_size=batch_size, shuffle=shuffle)
        for b in loader:
            enc = self.tokenizer([texts[i] for i in b], truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt")
            if labels is not None:
                enc["labels"] = torch.tensor([labels[i] for i in b])
            yield enc

    def fit(self, texts, labels, epochs: int = 4, lr: float = 5e-5, batch_size: int = 16, log=print):
        torch.manual_seed(self.seed)
        self.classes_ = sorted(set(labels))
        y = [self.classes_.index(l) for l in labels]
        texts = list(texts)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.classes_))
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * ((len(texts) + batch_size - 1) // batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        self.model.train()
        for epoch in range(epochs):
            total = 0.0
            for batch in self._batches(texts, y, batch_size, shuffle=True):
                loss = self.model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                total += loss.item()
            log(f"epoch {epoch + 1}/{epochs}  loss {total / (steps / epochs):.3f}")
        return self

    @torch.no_grad()
    def predict_proba(self, texts, batch_size: int = 32):
        self.model.eval()
        out = [torch.softmax(self.model(**b).logits, -1) for b in self._batches(list(texts), batch_size=batch_size)]
        return torch.cat(out).numpy()

    def predict(self, texts):
        return np.array(self.classes_)[self.predict_proba(texts).argmax(1)]

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        joblib.dump(self.classes_, f"{path}/classes.joblib")


def timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t
