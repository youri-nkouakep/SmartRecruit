"""Retrieval-augmented explanations: a local LLM justifies a match by quoting the resume."""
import json
import re
import threading
import urllib.request
from typing import Iterator

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT = """You help a recruiter screen a resume against a job offer.
Below, each requirement of the offer is paired with the closest sentence found in the resume (evidence E1, E2, ...).
Write exactly 3 short bullet points saying which requirements are met and which are missing.
Use only the evidence. End every bullet with the evidence id it relies on, like [E2].

Job offer:
{offer}

Evidence from the resume:
{passages}

Bullets:"""


class LocalLLM:
    """Small instruction-tuned model run with transformers on CPU or GPU."""

    def __init__(self, model_name: str = LLM_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=self.device).eval()

    def _inputs(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                  return_tensors="pt", return_dict=True).to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 180) -> str:
        enc = self._inputs(prompt)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def stream(self, prompt: str, max_new_tokens: int = 180) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**self._inputs(prompt), max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer)
        thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        yield from streamer
        thread.join()


class OllamaLLM:
    """Same interface, using a model served by Ollama (faster on a laptop)."""

    def __init__(self, model_name: str = "qwen2.5:3b", url: str = "http://localhost:11434"):
        self.model_name, self.url = model_name, url

    def _request(self, prompt: str, max_new_tokens: int, stream: bool):
        body = json.dumps({"model": self.model_name, "prompt": prompt, "stream": stream,
                           "options": {"num_predict": max_new_tokens, "temperature": 0}}).encode()
        req = urllib.request.Request(f"{self.url}/api/generate", body, {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=300)

    def generate(self, prompt: str, max_new_tokens: int = 220) -> str:
        with self._request(prompt, max_new_tokens, False) as r:
            return json.loads(r.read())["response"].strip()

    def stream(self, prompt: str, max_new_tokens: int = 220) -> Iterator[str]:
        with self._request(prompt, max_new_tokens, True) as r:
            for line in r:
                if line.strip():
                    yield json.loads(line).get("response", "")


def requirements(offer: str) -> list[str]:
    parts = re.split(r"(?<=[.;])\s+|,\s+(?=[A-Za-z])", offer)
    return [p.strip(" .;") for p in parts if len(p.split()) >= 2][:8]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\s{2,}|\s(?=[A-Z][a-z]+ [A-Z][a-z]+:)", text)
    return [p.strip() for p in parts if 4 <= len(p.split()) <= 60]


def evidence(offer: str, resume: str, encoder, n: int = 5) -> list[dict]:
    """Closest resume sentence for each requirement of the offer (extractive, always grounded)."""
    reqs, sents = requirements(offer), sentences(resume)
    if not reqs or not sents:
        return []
    r = encoder.encode(reqs, normalize_embeddings=True)
    s = encoder.encode(sents, normalize_embeddings=True)
    sim = r @ s.T
    rows = [{"requirement": q, "evidence": sents[int(sim[i].argmax())], "similarity": float(sim[i].max())}
            for i, q in enumerate(reqs)]
    return sorted(rows, key=lambda x: -x["similarity"])[:n]


def build_prompt(offer: str, items: list[dict]) -> str:
    lines = [f"E{i + 1}. requirement: {e['requirement']} | resume: \"{e['evidence']}\""
             for i, e in enumerate(items)]
    return PROMPT.format(offer=offer, passages="\n".join(lines))


def bullets(answer: str) -> list[str]:
    """Top-level bullets only; indented sub-points are attached to their parent bullet."""
    out: list[str] = []
    for line in answer.splitlines():
        if re.match(r"^(?:[-*•]|\d+[.)])\s+", line):
            out.append(line.strip())
        elif out and line.strip():
            out[-1] += " " + line.strip()
    return out


def grounding(answer: str, n_evidence: int) -> tuple[int, int]:
    """(bullets citing at least one valid evidence id, total bullets)."""
    bs = bullets(answer)
    ok = 0
    for b in bs:
        ids = [int(i) for i in re.findall(r"\[E(\d+)\]", b)]
        ok += bool(ids) and all(1 <= i <= n_evidence for i in ids)
    return ok, len(bs)
