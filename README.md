# Vistaar

**Citation-grounded document QA with thresholded abstention.**

Vistaar answers questions about your documents and tells you exactly which
passages support each answer. When the document does not contain the answer,
it says so instead of guessing — and that abstention decision is driven by a
reader confidence score whose threshold can be tuned against measured risk.
We report the full risk–coverage curve rather than claiming probability
calibration.

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

NATURAL=results/runs/natural-clean
ENRICHED=results/runs/negative-enriched-clean

python -m eval.run_eval --dataset-partition validation \
  --papers 40 --workers 2 --out "$NATURAL"
python -m eval.sweep --records "$NATURAL/records.jsonl" --config E \
  --out "$NATURAL"

python -m eval.run_eval --dataset-partition validation \
  --papers 281 --only E --balanced --workers 2 \
  --out "$ENRICHED"
python -m eval.sweep --records "$ENRICHED/records.jsonl" --config E \
  --out "$ENRICHED"
python -m eval.threshold_protocol --records "$ENRICHED/records.jsonl" \
  --config E --out "$ENRICHED/threshold_protocol"
```

An explicit local-only shadow mode collects versioned confidence diagnostics in
a dedicated cache/output namespace. Keep smoke runs bounded unless a larger
inference budget has been approved:

```bash
python -m eval.run_eval --shadow --dataset-partition validation \
  --papers 1 --limit 2 --only E --workers 1
```

Shadow mode rejects hosted readers, diagnostic-free cache rows, the ordinary
evaluation cache root, and nonempty output directories. Its development and
partition protocol is `docs/MILESTONE5B_SHADOW_PROTOCOL.md`.
The protocol freezes settings for a future calibrator experiment; preprocessing,
fitting, repeated-CV model selection, and final refitting are not implemented or
executed in this checkpoint.

Every evaluation has an isolated index/cache fingerprint and refuses to replace
a non-empty output directory. Outputs include:

- `summary.json` — all metrics with bootstrap confidence intervals
- `table.md` — the ablation table
- `records.jsonl` — predictions, all gold answers, scores and chunk provenance
- optional confidence diagnostics follow `docs/CONFIDENCE_DIAGNOSTICS.md`
- `run_manifest.json` — dataset/code/model/config fingerprints and index checks
- `sweep_E.json` / `.md` — risk–coverage curve and confidence AUC
- `threshold_protocol/` — document-disjoint calibration/test results

The original Milestone 1 artifacts remain under `results/` and
`results/balanced/`. The clean audited artifacts are under
`results/milestone2/`; see `docs/MILESTONE2_REPORT.md`.

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
| Balanced Acc | mean answerable recall and unanswerable recall |
| Conditional F1 | first-gold F1 among covered answerable questions |
| Covered F1 | F1 over every covered question, assigning zero to answered unanswerable questions |
| Latency | wall-clock per question |

Answer-F1 summaries use 1,000 bootstrap resamples. Held-out abstention intervals
use 2,000 document-cluster resamples. Configurations are compared with a paired
bootstrap test whose differences are centred under the null.

### Clean results (QASPER validation, 40 papers, 127 questions, 2,987 unique chunks)

| Config | F1 [95% CI] | Cite-P | Cite-R | Ret-R | Lat (s) |
|---|---|---|---|---|---|
| A: dense | 0.0604 [0.038, 0.087] | 0.079 | 0.114 | 0.227 | 0.64 |
| C: hybrid | 0.0719 [0.045, 0.103] | 0.095 | 0.133 | **0.267** | 0.98 |
| D: hybrid+rerank | 0.0695 [0.043, 0.101] | 0.118 | 0.143 | 0.245 | 1.16 |
| E: full | 0.0695 [0.043, 0.101] | 0.118 | 0.143 | 0.245 | 1.15 |

Two findings worth knowing before you read the numbers as a ranking:

- **Hybrid retrieval significantly improves evidence recall** over dense-only
  (0.227 → 0.267, paired bootstrap p < 0.0001).
- **The cross-encoder reranker significantly *reduces* evidence recall**
  (0.267 → 0.245, p = 0.0014). It also slightly reduces answer F1
  (0.0719 → 0.0695), but not significantly (p = 0.8371). Do not adopt a
  reranker without evaluating it on your own domain.

The clean natural-sample AUC is **0.6459**, but it rests on only five negatives.
The historical 0.019 selective-F1 baseline and 4.4× gain were invalid and are
not retained as findings.

### Negative-enriched abstention (config E)

Because the natural sample has only 5 negatives, the primary abstention analysis
uses a negative-enriched slice: all 51 documents in the 281-paper split that contain
at least one unanswerable question → **231 questions (171 answerable, 60
unanswerable), 3,784 unique chunks**. This is 74% answerable, not class-balanced.

The document-disjoint protocol selects threshold **0.25** on 31 calibration
documents (101 answerable / 40 unanswerable), then evaluates once on 20 held-out
documents (70/20):

| Held-out quantity | Result |
|---|---:|
| AUC | 0.6443 [0.5322, 0.7467] |
| Balanced accuracy @ 0.25 | 0.5179 [0.4289, 0.5938] |
| Coverage | 0.1778 |
| Conditional F1 | 0.0115 (always answer: 0.0107) |
| Covered F1 | 0.0094 (always answer: 0.0083) |
| Answered unanswerable | 3 / 20 |

The balanced-accuracy interval includes chance. The score does not yet support a
useful abstention claim. On all 231 questions, signal AUCs are confidence 0.6181,
retrieval 0.5830, and confidence + z(retrieval) 0.5827.

### Known limitations

- F1 understates quality: QASPER gold answers are abstractive while the local
  reader is extractive, so correct-but-shorter answers score low.
- The evaluation is single-domain (NLP papers) and English-only.
- Abstention remains weak: held-out balanced accuracy is statistically
  compatible with chance, and the natural held-out partition has only two
  unanswerable questions.
- The negative-enriched stress test is not representative of natural prevalence.
- Conditional F1 excludes answered-unanswerable questions; covered F1 is also
  reported so those errors receive zero utility.
- Citation precision uses vocabulary overlap against gold evidence — strict,
  but it cannot credit correct paraphrase.
- No human evaluation.


---

## Project layout

```
vistaar/
  config.py        settings; env normalisation
  diagnostics.py   optional inference-time confidence diagnostics schema
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
  abstention_signals.py  offline comparison of abstention signals
  provenance.py    stable run/cache fingerprints
  threshold_protocol.py document-grouped calibration and held-out evaluation
  compare_results.py historical-to-clean record verification
tests/
  ...              45 tests, no model downloads
```

---

## Design notes

**Per-namespace isolation.** FAISS indices are namespaced so one user's
documents are never retrievable by another. The original single global index
was a data-leak bug.

**Why chunking preserves offsets.** Without page numbers and chunk ids a
citation cannot be verified, and an unauditable citation is worse than none.

**Why the reader runs on CPU.** The MPS backend aborts on this workload
(Metal command-buffer assertion). Exact model revisions, package versions and
platform are fingerprinted; identical local smoke runs reproduced all scientific
fields, while latency is treated as environment-dependent.

**Why `thinkingBudget: 0`.** Gemini 3.x spends its output budget on
`thoughtsTokenCount` by default and can return an empty candidate. The client
sets it to zero explicitly; without this the app looks permanently "busy".

---

## License

MIT. QASPER is used under its own license (CC BY 4.0).
