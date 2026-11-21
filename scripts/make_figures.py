#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    doc = pd.read_csv(RES / "docai_benchmark.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    cols = ["overall_single", "overall_two_column"]
    doc[cols].rename(columns={"overall_single": "single column", "overall_two_column": "two columns"}) \
        .plot.barh(ax=ax, color=["#457b9d", "#e9c46a"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("field extraction F1")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("Document AI on 50 synthetic resumes")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIG / "docai_benchmark.png", dpi=130)
    plt.close(fig)

    cls = pd.DataFrame(json.loads((RES / "classification.json").read_text())).T
    cls = cls.loc[[i for i in cls.index if "TF-IDF" in i] + [i for i in cls.index if "Embeddings" in i]
                  + [i for i in cls.index if "Fine-tuned" in i]]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    labels = ["TF-IDF\n+ log. reg.", "bge-small embeddings\n+ log. reg.", "DistilBERT\nfine-tuned"]
    ax.bar(labels, cls["f1_macro"].astype(float), color=["#bbbbbb", "#457b9d", "#2a9d8f"])
    for i, v in enumerate(cls["f1_macro"].astype(float)):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("F1 macro (24 classes)")
    ax.set_title("Resume classification, 497 test resumes")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "classification.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
