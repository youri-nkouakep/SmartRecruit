"""SmartRecruit demo. Run: streamlit run app/streamlit_app.py"""
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartrecruit.service import SmartRecruit  # noqa: E402

RESULTS = ROOT / "results"
SAMPLES = ROOT / "data" / "synthetic_resumes"

st.set_page_config(page_title="SmartRecruit", layout="wide")


@st.cache_resource
def get_service():
    return SmartRecruit()


@st.cache_data
def offers():
    return json.loads((ROOT / "data" / "job_offers.json").read_text())


service = get_service()
st.title("SmartRecruit")
st.caption("Document AI, NLP and retrieval for resume screening: OCR and layout analysis, "
           "fine-tuned transformer classification, semantic search with re-ranking, "
           "LLM explanations grounded in the resume, and a bias audit.")

tab_parse, tab_match, tab_bench = st.tabs(["Read a resume", "Find candidates", "Benchmarks"])

with tab_parse:
    left, right = st.columns([2, 3])
    with left:
        upload = st.file_uploader("Resume (PDF, scanned image or photo)",
                                  type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
        samples = sorted(p.name for p in SAMPLES.glob("cv_*_scan.jpg"))[:10] if SAMPLES.exists() else []
        sample = st.selectbox("...or a synthetic sample", ["-"] + samples)
        data, name = None, None
        if upload is not None:
            data, name = upload.getvalue(), upload.name
        elif sample != "-":
            data, name = (SAMPLES / sample).read_bytes(), sample
        if data and Path(name).suffix.lower() != ".pdf":
            st.image(data, width="stretch")
    with right:
        if data:
            with st.spinner("Reading the document..."):
                out = service.parse(data, name)
            r = out["resume"]
            st.success(f"Read with {out['source']} in {out['seconds']:.1f} s")
            c1, c2, c3 = st.columns(3)
            c1.metric("Name", r.name or "-")
            c2.metric("Email", r.email or "-")
            c3.metric("Phone", r.phone or "-")
            st.markdown("**Skills:** " + (", ".join(r.skills) or "-"))
            st.dataframe(pd.DataFrame([e.model_dump() for e in r.experience]), hide_index=True)
            st.dataframe(pd.DataFrame([e.model_dump() for e in r.education]), hide_index=True)
            text = r.to_text()
            if len(text) > 50:
                pred = service.classify(text)
                st.markdown(f"**Predicted job family** ({pred['model']}): " +
                            ", ".join(f"{c} ({p:.0%})" for c, p in pred["top"]))
                st.caption("Terms that weigh most (TF-IDF model): " +
                           ", ".join(t for t, _ in pred["key_terms"]))
            with st.expander("Structured JSON"):
                st.json(r.model_dump())

with tab_match:
    items = offers()
    choice = st.selectbox("Job offer", [o["category"] for o in items])
    offer = st.text_area("Offer text", next(o["text"] for o in items if o["category"] == choice), height=120)
    k = st.slider("Candidates", 3, 10, 5)
    if st.button("Search", type="primary"):
        with st.spinner("BM25 + dense retrieval, then cross-encoder re-ranking..."):
            st.session_state["hits"] = service.match(offer, k)
            st.session_state["offer"] = offer
    for i, hit in enumerate(st.session_state.get("hits", [])):
        with st.container(border=True):
            st.markdown(f"**#{i + 1}** - resume {hit['id']} - category *{hit['category']}* - "
                        f"relevance score {hit['score']:.2f}")
            ev = pd.DataFrame(hit["evidence"])
            if not ev.empty:
                ev.index = [f"E{j + 1}" for j in range(len(ev))]
                st.dataframe(ev.round(2), width="stretch")
            if st.button("Explain this match with the LLM", key=f"llm{i}"):
                with st.spinner("Generating (a few seconds to a minute on CPU)..."):
                    ans = service.justify(st.session_state["offer"], hit["evidence"])
                st.markdown(ans["answer"])
                st.caption(f"{ans['cited_bullets']}/{ans['bullets']} bullets cite valid evidence "
                           f"- {ans['seconds']:.0f} s")

with tab_bench:
    def show(title, file, note=""):
        path = RESULTS / file
        if path.exists():
            st.subheader(title)
            if note:
                st.caption(note)
            st.dataframe(pd.read_csv(path, index_col=0).round(3), width="stretch")

    show("Document AI: field extraction F1", "docai_benchmark.csv",
         "50 synthetic resumes (single and two-column layouts), digital and scanned.")
    if (RESULTS / "classification.json").exists():
        st.subheader("Resume classification (24 job families, 497 test resumes)")
        st.dataframe(pd.DataFrame(json.loads((RESULTS / "classification.json").read_text())).T
                     .drop(columns="model").round(3), width="stretch")
    show("Offer to resume retrieval", "retrieval.csv", "24 offers, 2,483 resumes.")
    if (RESULTS / "bias_audit.json").exists():
        st.subheader("Bias audit (same resume, different name)")
        st.dataframe(pd.DataFrame(json.loads((RESULTS / "bias_audit.json").read_text())).T.round(4),
                     width="stretch")
