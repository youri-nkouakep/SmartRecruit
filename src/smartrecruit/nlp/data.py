"""Public resume dataset (2,484 resumes, 24 job categories, scraped from livecareer.com).

The first line of each resume is the job title, which often names the category
directly. It is removed so that classifiers learn from the content, not the title.
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HF_REPO = "Divyaamith/Kaggle-Resume"
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
SEED = 42

_SEGMENTS = re.compile(r"\s{3,}|\n+")


def strip_title(text: str) -> str:
    segments = [s.strip() for s in _SEGMENTS.split(text) if s.strip()]
    while segments and segments[0].upper() == segments[0] and len(segments) > 1:
        segments.pop(0)
    return " ".join(segments)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def download(path: Path = DATA_DIR / "resumes.csv") -> Path:
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(HF_REPO, "Resume.csv", repo_type="dataset")
    df = pd.read_csv(src)[["ID", "Resume_str", "Category"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load(path: Path = DATA_DIR / "resumes.csv", keep_title: bool = False) -> pd.DataFrame:
    if not path.exists():
        download(path)
    df = pd.read_csv(path).dropna(subset=["Resume_str"])
    df["title"] = df["Resume_str"].map(lambda t: clean_text(_SEGMENTS.split(t.strip())[0]))
    df["text"] = df["Resume_str"].map(clean_text if keep_title else lambda t: clean_text(strip_title(t)))
    df = df[df["text"].str.len() > 200].rename(columns={"Category": "label", "ID": "id"})
    df["label"] = df["label"].str.replace("-", " ").str.title().replace({"Hr": "HR", "Bpo": "BPO"})
    return df[["id", "title", "text", "label"]].reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, stratify=df["label"], random_state=SEED)
