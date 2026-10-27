# Vistaar

**Citation-grounded document QA with calibrated abstention.**

Vistaar answers questions about your documents and tells you exactly which
passages support each answer. When the document does not contain the answer,
it says so instead of guessing — and that abstention decision is driven by a
calibrated confidence score you can tune against measured risk.

This repository contains a working service **and** the evaluation harness used
to produce its published numbers.

---

## Why the abstention part matters

Retrieval-augmented generation fails in a specific, measurable way: when the
retrieved context does not contain the answer, the model still produces fluent
text. The usual mitigation is a prompt telling the model not to hallucinate,
which does not work reliably.

Vistaar instead separates the problem into two decisions:

1. **Retrieve** candidate passages (dense + lexical fusion, optional reranking).
2. **Read** a span and report a confidence comparing the answer hypothesis
   against an explicit no-answer hypothesis.

The confidence score is then thresholded. Because the raw score is retained for
every evaluated question, the threshold can be swept offline to trace the
**risk–coverage curve** — accuracy as a function of how often the system is
allowed to answer.

---

## Architecture

```
Upload ──► validate (size, type) ──► extract text (PDF/TXT, page offsets)
       ──► chunk (boundary-aware, provenance retained)
       ──► embed (MiniLM, L2-normalised) ──► per-namespace FAISS + BM25
                                          │
Question ─────────────────────────────────┴──► hybrid retrieval (RRF)
                                             ──► cross-encoder rerank (optional)
                                             ──► span reader + null-hypothesis
                                                 confidence
                                             ──► answer + cited passages
```

Every chunk carries `doc_id`, `chunk_id`, page number and ordinal, so citations
point at a real location rather than a model assertion.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # only needed for the hosted reader
uvicorn app:app --reload  # http://127.0.0.1:8000
```

Two readers are supported:

| `VISTAAR_READER` | Backend | Requires |
|---|---|---|
| `local` (default) | `deepset/roberta-base-squad2` on CPU | nothing — runs offline |
| `gemini` | Gemini via REST | a `GEMINI_API_KEY` with quota |

The local reader is the default because it is deterministic and reproducible,
which is what the evaluation needs. The hosted reader produces more fluent
answers but is non-deterministic and quota-bound.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Upload + ask UI |
| `GET` | `/health` | Liveness, reader config, active namespaces |
| `POST` | `/upload` | `multipart/form-data`: `file`, `namespace` |
| `POST` | `/ask` | `{"question": "...", "namespace": "..."}` |
| `DELETE` | `/namespace/{ns}` | Drop a namespace and its index |

`/ask` returns:

```json
{
  "answer": "42.5 F1",
  "abstained": false,
  "citations": [{"id": "P1", "chunk_id": "paper-abc:0003", "page": 4, "text": "..."}],
  "scores": {"confidence": 0.87, "retrieval": 0.61, "passages": 4},
  "latency_s": 0.53,
  "model": "deepset/roberta-base-squad2"
}
```

---

## Evaluation

The harness scores the system on **QASPER** (`allenai/qasper`, validation
split): 281 papers, gold answers, gold evidence spans, and native
`unanswerable` labels. No new annotation is required, and the abstention
experiment gets real negatives for free.

```bash
# download the eval data (~4.5 MB)
curl -sL "https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/validation/0.parquet" \
  -o data/raw/qasper_validation.parquet

python -m eval.run_eval --papers 40 --workers 2
python -m eval.sweep --config E
```

Results are written to `results/`:

- `summary.json` — all metrics with bootstrap confidence intervals
- `table.md` — the ablation table
- `records.jsonl` — per-question predictions, scores and retrieved passages
- `sweep_E.json` / `.md` — risk–coverage curve and confidence AUC

### Evaluation matrix

| Config | Retrieval |
|---|---|
| A | dense only |
| B | dense + abstention |
| C | hybrid (dense + BM25, reciprocal rank fusion) |
| D | hybrid + cross-encoder rerank |
| E | hybrid + rerank + abstention (full system) |

### Metrics

| Metric | Definition |
|---|---|
| EM / F1 | SQuAD-style exact match and token F1 against gold answers, computed on the reader's raw span (so abstention does not double-penalise answer quality) |
| Ret-R | fraction of gold-evidence vocabulary covered by the retrieved passages (union, depth 4) |
| Cite-P | fraction of cited passages drawn ≥50% from gold evidence |
| Cite-R | fraction of gold-evidence vocabulary covered by cited passages |
| Abst-Acc | share of questions where answering/abstaining matched gold answerability |
| AUC | confidence score's ability to separate answerable from unanswerable questions |
| Latency | wall-clock per question |

All headline numbers are reported with bootstrap 95% confidence intervals
(1000 resamples). Configurations are compared with a paired bootstrap test whose
differences are centred under the null.

### Results (QASPER validation, 40 papers, 127 questions, 7,391 chunks)

| Config | F1 [95% CI] | Cite-P | Cite-R | Ret-R | Lat (s) |
|---|---|---|---|---|---|
| A: dense | 0.0403 [0.024, 0.060] | 0.087 | 0.104 | 0.184 | 0.59 |
| C: hybrid | 0.0721 [0.046, 0.103] | 0.095 | 0.129 | **0.267** | 0.66 |
| D: hybrid+rerank | 0.0795 [0.051, 0.114] | 0.102 | 0.139 | 0.247 | 0.86 |
| E: full | 0.0795 [0.051, 0.114] | 0.102 | 0.139 | 0.247 | 0.88 |

Two findings worth knowing before you read the numbers as a ranking:

- **Hybrid retrieval significantly improves evidence recall** over dense-only
  (0.184 → 0.267, p < 0.001).
- **The cross-encoder reranker significantly *reduces* evidence recall**
  (0.267 → 0.247, p = 0.002) even though it raises answer F1. A reranker trained
  on MS MARCO web passages does not transfer to reference-dense scientific text.
  Do not adopt one without evaluating it on your own domain.

Abstention: confidence AUC = **0.696**. Thresholding raises answer F1 from
0.019 at full coverage to **0.083 at 15% coverage** (4.4×). See
`results/sweep_E.md`.

### Known limitations

- F1 understates quality: QASPER gold answers are abstractive while the local
  reader is extractive, so correct-but-shorter answers score low.
- The evaluation is single-domain (NLP papers) and English-only.
- Only 5 of 127 questions are unanswerable, so the abstention AUC is noisy.
  A class-balanced run over the full 281-paper split is the next step.
- Citation precision uses vocabulary overlap against gold evidence — strict,
  but it cannot credit correct paraphrase.
- No human evaluation.


---

## Project layout

```
vistaar/
  config.py        settings; env normalisation
  text.py          extraction + boundary-aware chunking with provenance
  index.py         embeddings, per-namespace FAISS/BM25 store, RRF, reranker
  reader.py        local span reader with null-hypothesis confidence
  generate.py      Gemini REST client (thinkingBudget=0)
  pipeline.py      end-to-end pipeline; ablations as RetrievalConfig
  api.py           FastAPI application
  templates/       single-page UI
eval/
  qasper.py        dataset loader
  metrics.py       EM, F1, citation P/R, bootstrap CI, paired test
  run_eval.py      evaluation matrix runner
  sweep.py         abstention threshold sweep / risk-coverage
tests/
  test_core.py     12 tests, no model downloads
```

---

## Design notes

**Per-namespace isolation.** FAISS indices are namespaced so one user's
documents are never retrievable by another. The original single global index
was a data-leak bug.

**Why chunking preserves offsets.** Without page numbers and chunk ids a
citation cannot be verified, and an unauditable citation is worse than none.

**Why the reader runs on CPU.** The MPS backend aborts on this workload
(Metal command-buffer assertion). A base-size encoder is fast enough on
Apple silicon and CPU results are deterministic across machines.

**Why `thinkingBudget: 0`.** Gemini 3.x spends its output budget on
`thoughtsTokenCount` by default and can return an empty candidate. The client
sets it to zero explicitly; without this the app looks permanently "busy".

---

## License

MIT. QASPER is used under its own license (CC BY 4.0).
