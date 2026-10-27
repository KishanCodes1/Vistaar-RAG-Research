# PROJECT_STATE.md

**Project:** Vistaar — citation-grounded RAG with thresholded abstention
**Repository:** `/Users/kishansingh/Documents/STUDY/Research Paper/Vistaar` (git repo, branch `main`; milestone-1 commit on top of the initial import)
**State captured:** 2026-10-27, 21:15 local (Milestone 1 reconciliation applied)
**Status:** working system + completed evaluation + paper draft v1.1. Nothing is running.

---

## 1. Current Architecture & What Was Executed

### 1.1 Architecture (as built)

```
Upload ─► validate (type, 20 MB cap) ─► extract text (pypdf / UTF-8, page offsets)
       ─► boundary-aware chunking (800 chars, 150 overlap, provenance retained)
       ─► embed (all-MiniLM-L6-v2, L2-normalised) ─► per-namespace FAISS (IndexFlatIP)
                                                      + per-namespace BM25 index
Question ─────────────────────────────────────────────┴─► hybrid retrieval (RRF, k=60)
                                                        ─► optional cross-encoder rerank
                                                        ─► span reader + null-hypothesis confidence
                                                        ─► answer + cited passages (page, chunk_id)
```

Two interchangeable readers, selected by `VISTAAR_READER`:

| Value | Backend | Status |
|---|---|---|
| `local` (default) | `deepset/roberta-base-squad2`, CPU | working — used for **all** reported metrics |
| `gemini` | Gemini REST (`gemini-3.5-flash`) | code correct, **not evaluable** — free-tier quota (see §2.1) |

Ablations are expressed as `RetrievalConfig` instances passed per call, so all five
configurations share one index and one embedding pass:

| Config | Dense | BM25 | Rerank | Abstention |
|---|---|---|---|---|
| A | ✓ | – | – | – |
| B | ✓ | – | – | ✓ |
| C | ✓ | ✓ | – | – |
| D | ✓ | ✓ | ✓ | – |
| E | ✓ | ✓ | ✓ | ✓ |

### 1.2 Datasets

| Dataset | Location | Size | Role |
|---|---|---|---|
| QASPER validation (`allenai/qasper`) | `data/raw/qasper_validation.parquet` | 281 papers, 4.5 MB | **primary evaluation** — gold answers, gold evidence spans, native `unanswerable` labels |
| CUAD v1 (`theatticusproject/cuad`) | `data/raw/cuad_v1.json` | 510 contracts, 20,910 QA pairs (14,208 `is_impossible`), 38 MB | **downloaded, not yet wired in** — intended legal/contract domain |

No hand annotation was required; both datasets ship gold labels. QASPER questions whose
gold answer consists only of pointer tokens (`BIBREF19`, `TABREF13`, `FIGREF4`) are
detected by regex and treated as having no extractable answer text.

### 1.3 Metrics (implemented in `eval/metrics.py`)

| Metric | Definition |
|---|---|
| EM / F1 | SQuAD normalisation; F1 computed on the reader's **raw span** (`raw_prediction`), so abstention does not double-penalise answer quality |
| Ret-R | fraction of gold-evidence vocabulary covered by the **union** of retrieved passages at depth 4 |
| Cite-P | fraction of cited passages drawn ≥50% from gold evidence (containment) |
| Cite-R | fraction of gold-evidence vocabulary covered by cited passages |
| Abst-Acc | share of questions where answer/abstain matched gold answerability |
| Balanced Acc | mean of per-class correct rates (added because the answerable base rate is 96%) |
| AUC | confidence score's ability to separate answerable from unanswerable (rank-based, tie-corrected) |
| Selective F1 | mean raw-answer F1 over questions the system chose to answer at a given threshold |
| Latency | wall-clock seconds per question, CPU only |

Statistics: 95% percentile bootstrap intervals (1,000 resamples, seed 13); paired
bootstrap tests with differences **centred under the null**.

### 1.4 What was executed

**Run 1 — main ablation** (`--papers 40`): 40 papers, **127 questions (122 answerable,
5 unanswerable)**, **7,391 chunks** indexed once and shared across configs.
Output: `results/summary.json`, `results/table.md`, `results/records.jsonl`.

| Config | F1 [95% CI] | EM | Cite-P | Cite-R | Ret-R | Abst-Acc | Over-abst | Lat (s) |
|---|---|---|---|---|---|---|---|---|
| A: dense | 0.0403 [0.024, 0.060] | 0.0000 | 0.087 | 0.104 | 0.184 | 0.961 | 0.000 | 0.59 |
| B: dense+abstain | 0.0410 [0.025, 0.061] | 0.0000 | 0.087 | 0.104 | 0.184 | 0.189 | 0.844 | 0.59 |
| C: hybrid | 0.0721 [0.046, 0.103] | 0.0082 | 0.095 | 0.129 | **0.267** | 0.961 | 0.000 | 0.66 |
| D: hybrid+rerank | 0.0795 [0.051, 0.114] | 0.0164 | 0.102 | 0.139 | 0.247 | 0.961 | 0.000 | 0.86 |
| E: full | 0.0795 [0.051, 0.114] | 0.0164 | 0.102 | 0.139 | 0.247 | 0.189 | 0.844 | 0.88 |

**Paired bootstrap significance** (main run):

| Comparison | Metric | Δ | p | Verdict |
|---|---|---|---|---|
| A → C | Ret-R | **+0.0830** | <0.001 | **significant win** for hybrid |
| C → D | Ret-R | **−0.0203** | **0.002** | **significant loss** from reranking |
| A → E | Ret-R | +0.0627 | <0.001 | significant |
| A → C | F1 | +0.0142 | 0.204 | not significant |
| C → D | F1 | +0.0091 | 0.402 | not significant |
| A → E | F1 | +0.0233 | 0.108 | not significant |

**Run 2 — abstention sweep, main run** (`eval/sweep.py`, config E, n=127):
AUC **0.6959**. Selective F1 rises from **0.019 at 100% coverage → 0.083 at 15.0%
coverage** (threshold 0.50), a 4.4× gain. Max balanced accuracy **0.639 at
threshold 0.10** (coverage 0.268, selective F1 0.049). The `best` field in
`sweep_E.json` maximises selective F1, not balanced accuracy.
Output: `results/sweep_E.json`, `results/sweep_E.md`.

**Run 3 — class-balanced abstention run** (`--papers 281 --only E --balanced`):
restricted to the 51 documents that contain at least one unanswerable question →
**231 questions (171 answerable, 60 unanswerable)**, 4,645 chunks.
Output: `results/balanced/summary.json`, `records.jsonl`, `table.md`, `sweep_E.*`.

Config E on this slice: F1 0.0559 [0.0388, 0.0763], EM 0.0058, Cite-P 0.026,
Cite-R 0.100, Ret-R 0.183, Abst-Acc 0.325, over-abstention 0.877,
missed-unanswerable 0.10, latency 0.86 s.

**Balanced sweep:** AUC **0.6172** (n=231). Peak **balanced accuracy 0.566 at
threshold 0.10** (coverage 0.281, selective F1 0.040). Best selective F1 **0.055 at
threshold 0.40** (coverage 0.156, balanced accuracy 0.515). At full coverage,
selective F1 = 0.0385. The default threshold 0.50 yields selective F1 0.0373,
slightly below full coverage, and balanced accuracy 0.511.

Alternative abstention signals measured offline from the same records (n=231),
reproduced by `python -m eval.abstention_signals`: retrieval score alone AUC
0.5830; `confidence + z(retrieval)` AUC 0.5832. **No combination beat confidence
alone (0.6172).** Output: `results/balanced/signals.json`.

**Verification — unit tests:** `python -m pytest tests/ -q` → **12 passed**
(env whitespace regression, namespace isolation, chunking bounds, RRF dedupe,
metric known-values, bootstrap interval sanity, rejected file types).

**Verification — API end-to-end** via `TestClient`: `/health` 200, `/` 200,
`/upload` 200, `/ask` returned `0.0795` with confidence 0.992 and 1 citation,
nonsense question correctly abstained, `DELETE /namespace/{ns}` 200,
`.exe` upload → 400.

---

## 2. Key Findings & Constraints

### 2.1 Gemini free-tier quota is the binding constraint on the hosted path

- Quota error: `generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash`
  → **20 requests per day**, already exhausted, reset in ~15h49m at the time of the run
  (i.e. usable again roughly midday 2026-10-28).
- Therefore **no hosted-model evaluation was possible**; every reported number comes
  from the deterministic local reader. This is a limitation to state, not to hide.

### 2.2 Live model names differ from the code in the original project

`GET /v1beta/models` returns `gemini-2.5-flash` as *listed* but generation returns
`404 … no longer available to new users`. Live and generating: `gemini-3.5-flash`
(reports `modelVersion: gemini-3.6-flash`), `gemini-3.6-flash`, `gemini-3.7-flash`,
`gemini-3.8-flash`. `gemini-3.8-flash` returned **503 high demand** repeatedly, so
`gemini-3.5-flash` is primary and `gemini-3.8-flash` the only fallback.

### 2.3 `thinkingBudget: 0` is mandatory for Gemini 3.x

With the default thinking budget, `maxOutputTokens=20` produced
`finishReason: MAX_TOKENS`, `content: {}`, and `thoughtsTokenCount: 16` — i.e. the
entire output budget consumed by reasoning tokens. The original app saw empty
`response.text`, fell through its error path and reported "AI service is busy"
permanently. Setting `thinkingConfig.thinkingBudget = 0` makes it return `OK` in ~2.3 s.

### 2.4 The original `.env` had a leading space in the value

Byte-level inspection: `key='GEMINI_API_KEY'`, `valuelen=54`, value starts with a space
→ real key is 53 chars. `source .env` therefore assigned an empty variable and direct
`curl` calls returned `403 PERMISSION_DENIED (unregistered callers)`.
`config._clean()` strips and there is a regression test for it.

### 2.5 transformers 5.21.0 removed the `question-answering` pipeline

`pipeline("question-answering", …)` now raises
`KeyError: The task question-answering has been removed`. The reader is therefore
implemented directly on `AutoModelForQuestionAnswering` + `AutoTokenizer` with a
sliding window (384 tokens, stride 128), `padding=True`, and explicit
`offset_mapping` handling. This is also why confidence is computed from raw logits.

### 2.6 MPS backend hard-crashes on this workload

`SentenceTransformer` / `CrossEncoder` on MPS abort with
`failed assertion _status < MTLCommandBufferStatusCommitted`. Both are pinned to
`device="cpu"`. CPU inference is also what makes the numbers reproducible.

### 2.7 Confidence formulation was the single largest correctness risk

The first implementation used a joint softmax ratio
(`p_span / (p_span + p_null)`). Because the softmax is spread over hundreds of
candidate positions, every span looked improbable → **76% over-abstention** and
nonsense AUC. Replaced with the logit margin against the SQuAD2 null candidate:

```
margin     = (start_logit[i] + end_logit[j]) − (start_logit[0] + end_logit[0])
confidence = sigmoid(margin)
```

Confidence now spans 0.002–0.946 on real questions, with `0.5` interpretable as
"span and null equally likely". Default `reader_threshold = 0.5`.

### 2.8 QASPER parquet text is partly character-spaced

In 40/40 sampled papers, `full_text` contains long runs of single-character tokens
(`A l t h o u g h N e u r a l …`). Consequences:

- Exact-substring matching against `full_text` finds **0/435** evidence spans.
- Joining tokens without spaces finds **381/435** (87%).
- De-spacing is lossy (words merge: `AlthoughNeuralMachineTranslationhas…`), so it was
  rejected.

All citation/retrieval metrics therefore use **token-vocabulary overlap**, never
substring matching. This is documented in the paper as a limitation.

### 2.9 Two metric bugs found and fixed during the session

1. **Overlap direction.** The first citation/recall metric asked "does this chunk cover
   the whole gold paragraph", which produced a nonsensical 2.4% recall. Fixed to
   vocabulary coverage over the **union** of retrieved passages, with containment-based
   citation precision.
2. **Uncentred paired bootstrap.** Resampling raw paired differences compares the
   observed mean against itself and returns **p ≈ 0.5 by construction**. Fixed by
   centring differences under the null; the corrected test is what produced the
   p < 0.001 and p = 0.002 results above.

### 2.10 Substantive findings

- **Hybrid retrieval significantly improves evidence recall** over dense-only
  (0.184 → 0.267, p < 0.001). Largest, most reliable effect in the study.
- **The cross-encoder reranker significantly *hurts* evidence recall**
  (0.267 → 0.247, p = 0.002) while raising answer F1 (not significant). Domain
  mismatch: MS MARCO web passages vs. reference-dense scientific paragraphs
  containing `BIBREFnn` placeholders. This is the paper's negative result.
- **F1 0.04–0.08 is a metric mismatch, not a broken system.** QASPER gold answers are
  abstractive (`multilingual NMT (MNMT) BIBREF19`); the reader is extractive and
  returns the shorter span (`multilingual NMT approaches`). The same system reaches
  0.696 AUC and a 4.4× selective-F1 gain on the natural sample, but only 0.617 AUC
  and a 1.4× gain on the class-balanced slice — the latter is the primary result.
- **Abstention is real but weak, and the natural-sample AUC is optimistic.** With 5
  negatives, AUC = 0.696; on 60 balanced negatives, AUC = **0.617** and peak balanced
  accuracy is only **0.566**. The paper reports both, with 0.696 labelled as an
  optimistic small-sample estimate; the 0.696 is not presented alone.
- **A fixed threshold of 0.5 over-abstains** (0.844) because the answerable prior is
  96%; the threshold has to be selected from measured balanced accuracy, not assumed.

### 2.11 Environment constraints

- Python 3.13.9, Apple M4, 16 GB RAM, no `ollama`, no `llama.cpp` → no local
  generative LLM available; extractive reader was the only offline option.
- Versions are pinned in `requirements.txt`: `transformers==5.21.0`, `torch==2.14.1`,
  `faiss-cpu==1.15.1`, `sentence-transformers==6.2.0`, `fastapi==0.162.0`,
  `numpy==2.5.4`, `pyarrow==26.0.0`, `pytest==9.0.1`.
- The original project at
  `/Users/kishansingh/Documents/STUDY/Research Paper/Vistaar-main/` was **not modified**;
  it was read only. The API key exists only in
  `…/VISTAAR/Vistaar/backend/.env` and was copied into the (gitignored) local `.env`.

---

## 3. Files Added/Modified

The repository started empty (`.git` only, zero commits), so every file below is
**new in this session**. Nothing from the original project was edited.

### Application package — `vistaar/`
| File | Purpose |
|---|---|
| `vistaar/__init__.py` | package exports (`Settings`, `RAGPipeline`, `RetrievalConfig`) |
| `vistaar/config.py` | `Settings` (frozen dataclass), env loading with whitespace/quote normalisation, reader switch |
| `vistaar/text.py` | PDF/text extraction, boundary-aware recursive chunking with provenance (`doc_id`, `page`, `chunk_id`, `ordinal`) |
| `vistaar/index.py` | `Embedder` (CPU), namespaced `Namespace` (FAISS + BM25, `RLock`), `Store` with save/load/`reset_namespace`, `reciprocal_rank_fusion`, `Reranker` (CPU) |
| `vistaar/reader.py` | `LocalReader`: RoBERTa-SQuAD2 span head, sliding window, sigmoid logit-margin confidence, span-offset citations |
| `vistaar/generate.py` | Gemini REST client: `thinkingBudget: 0`, ordered fallbacks, retry/backoff, token-bucket rate limiter, tolerant JSON parsing |
| `vistaar/pipeline.py` | `RAGPipeline` + `RetrievalConfig` + `Answer` (incl. `raw_answer`); per-call config so ablations share one index |
| `vistaar/api.py` | FastAPI app: `/`, `/health`, `/upload`, `/ask`, `DELETE /namespace/{ns}`; slowapi limits, size/type validation |
| `vistaar/templates/index.html` | single-page UI with abstention badge, confidence, citations |

### Evaluation harness — `eval/`
| File | Purpose |
|---|---|
| `eval/__init__.py` | package marker |
| `eval/qasper.py` | QASPER parquet loader; pointer-token filter; gold answers + evidence + answerability |
| `eval/metrics.py` | EM, token F1, `evidence_coverage`, citation P/R, `bootstrap_ci`, `paired_bootstrap_p` (null-centred), `summarize`, `to_markdown` |
| `eval/run_eval.py` | matrix runner: `--papers --limit --workers --out --only --balanced`; per-config JSONL cache; `READER_PASSAGES = 4` |
| `eval/sweep.py` | threshold sweep → risk–coverage curve, AUC, balanced accuracy, selective F1 |
| `eval/abstention_signals.py` | offline comparison of confidence, retrieval, and combined abstention signals |

### Tests, data, results
| File | Purpose |
|---|---|
| `tests/test_core.py` | 12 unit tests, no model downloads |
| `data/raw/qasper_validation.parquet` | QASPER validation (gitignored) |
| `data/raw/cuad_v1.json` | CUAD v1, 510 contracts (gitignored, **not yet used**) |
| `results/summary.json`, `results/table.md`, `results/records.jsonl` | main ablation outputs |
| `results/sweep_E.json`, `results/sweep_E.md` | main-run abstention sweep (AUC 0.6959) |
| `results/balanced/summary.json`, `table.md`, `records.jsonl` | balanced run outputs |
| `results/balanced/sweep_E.json`, `sweep_E.md` | balanced sweep (AUC 0.6172) |
| `results/balanced/signals.json` | signal comparison (confidence 0.6172 vs retrieval 0.5830 vs combined 0.5832) |
| `results/cache/{A,B,C,D,E}.jsonl` | per-question resume cache (gitignored) |

### Documentation and delivery
| File | Purpose |
|---|---|
| `paper/PAPER.md` | full draft v1.1: balanced abstention as primary, natural-sample AUC labelled optimistic, verified signal comparison |
| `docs/AUDIT.md` | 23 defects in the original codebase with fixed/open status |
| `README.md` | usage, architecture, API, metrics definitions, results table, limitations |
| `app.py` | ASGI entrypoint (`uvicorn app:app`) |
| `requirements.txt` | fully pinned dependency set |
| `Dockerfile` | image with models pre-downloaded, healthcheck |
| `Makefile` | `install / test / serve / data / eval / sweep / clean` |
| `.gitignore` | excludes `.venv/`, `.env`, `data/raw/`, `data/store/`, `results/cache/` |
| `.env`, `.env.example` | local key (gitignored) and documented template |
| `.github/workflows/ci.yml` | unit tests + 8-paper eval quality gate (F1 ≥ 0.01, Ret-R ≥ 0.10) |

### Deleted
| File | Reason |
|---|---|
| `paper/fill.py` | obsolete placeholder-injection script; `PAPER.md` is fully populated with real numbers, so it produced a stale duplicate |

---

## 4. Milestones

### Milestone 1 — Reconcile the paper with the balanced abstention result, then commit — ✅ completed

Completed in this milestone:

- `paper/PAPER.md` §5.4 now reports the class-balanced run (AUC 0.6172, n=231,
  60 unanswerable) as the **primary** result, with the natural-prevalence AUC
  0.696 explicitly labelled an optimistic small-sample estimate.
- Added the verified three-way signal comparison: confidence 0.6172, retrieval
  0.5830, combined 0.5832. Reproduced by `eval/abstention_signals.py` and stored
  in `results/balanced/signals.json`.
- Corrected the natural-run balanced-accuracy values (0.639 at threshold 0.10,
  0.594 at 0.30) directly from `results/sweep_E.json`; the earlier "0.603 at
  0.20" claim was wrong.
- Softened the abstract claim to "modestly above chance"; changed the title and
  tagline from "Calibrated Abstention" to "Thresholded Abstention", since no
  probability-calibration or ECE analysis was run.
- Extended §7 Limitations with weak abstention, non-independence, post-hoc
  threshold selection, and the first-gold vs best-over-gold selective-F1 caveat.
- Applied the same reconciled numbers to `README.md` and this file.
- Created the milestone commit:
  `feat(milestone-1): reconcile paper and readme with balanced abstention metrics and signal verification`.

Milestone 2 remains next.

### Milestone 2 — Add the legal/contract domain (CUAD), or scope the paper explicitly

`data/raw/cuad_v1.json` is already downloaded (510 contracts, 20,910 QA pairs, 14,208
`is_impossible`) but unused. Add `eval/cuad.py` (SQuAD-format loader using
`answer_start` offsets, `is_impossible` → unanswerable) and a retrieval-only ablation
(dense vs hybrid vs hybrid+rerank, depth 4) over ~40–60 contracts, plus a
class-balanced abstention subset (~100 answerable / 100 impossible). This yields a
genuine second-domain result and tests whether the reranker's negative effect
reproduces on short-clause legal text — the most interesting open question in the
project. If it cannot be completed cleanly, state in the paper that evaluation is
single-domain and drop the cross-domain claim.

### Milestone 3 — Scale the evaluation and harden for production

Rerun the ablation on the **full 281-paper split** with two bootstrap seeds instead of
the current 40-paper sample, so the confidence intervals (currently ±0.03 on F1) and
the reranker result are stable enough to defend in review. In parallel close the
production gaps already listed in `docs/AUDIT.md`: proxy-aware rate limiting
(`X-Forwarded-For`), authentication/tenant isolation beyond session namespaces,
OCR for scanned PDFs, and a Docker build + CI run to prove `requirements.txt`
resolves from scratch.

---

## 5. How to resume

```bash
cd "/Users/kishansingh/Documents/STUDY/Research Paper/Vistaar"

./.venv/bin/python -m pytest tests/ -q            # 12 tests, ~0.3s
./.venv/bin/python -m eval.run_eval --papers 40 --workers 2   # cached: near-instant
./.venv/bin/python -m eval.sweep --config E
./.venv/bin/python -m eval.sweep --records results/balanced/records.jsonl --config E --out results/balanced
./.venv/bin/python -m eval.abstention_signals --records results/balanced/records.jsonl --config E --out results/balanced/signals.json
./.venv/bin/uvicorn app:app --reload              # http://127.0.0.1:8000
```

Notes for whoever resumes:

- The per-question cache in `results/cache/` is keyed by `config → question_id` and is
  **shared across runs**; `results/balanced/records.jsonl` therefore reuses config-E
  rows produced during the main run. That is intentional (identical model and config),
  but it means `results/cache/E.jsonl` holds more rows (324) than either run alone.
- `rm -rf results/cache` forces a full re-evaluation (~4 min per config on 127
  questions, ~0.9 s/question CPU-bound).
- If the Gemini path is ever needed: the quota resets daily at 20 requests. Verify
  `thinkingConfig.thinkingBudget = 0` is still present before debugging anything else.
