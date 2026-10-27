# PROJECT_STATE.md

**Project:** Vistaar — citation-grounded RAG with thresholded abstention
**Repository:** `/Users/kishansingh/Documents/STUDY/Research Paper/Vistaar` (git repo, branch `main`; milestone-1 commit on top of the initial import)
**State captured:** 2026-10-28 (Milestone 3 full-QASPER verification complete)
**Status:** full 281-paper QASPER A–E evaluation independently verified. CUAD
has not started. Nothing is running; no Milestone 2/3 changes are committed.

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
| Conditional F1 | first-gold raw-answer F1 over covered questions known to be answerable |
| Covered F1 | F1 over all covered questions, assigning zero to answered-unanswerable questions |
| Latency | wall-clock seconds per question, CPU only |

Statistics: answer F1 uses 1,000 percentile-bootstrap resamples; held-out
abstention uses 2,000 document-cluster resamples; paired bootstrap differences
are **centred under the null**.

### 1.4 Historical Milestone 1 evaluations — preserved, not current

The values in this subsection are retained only as an audit trail. Their shared
index accumulated duplicate chunks, their cache lacked fingerprints, and the
natural 0.019/4.4× selective-F1 claim used a stale metric artifact. Do not cite
them as current results; see §1.5 and `docs/ABSTENTION_AUDIT.md`.

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

### 1.5 Current verified evaluations

**Clean natural ablation** (`results/milestone2/natural-clean-final/`): 40 papers,
127 questions (122 answerable, 5 unanswerable), **2,987 stored / 2,987 unique
chunks**, zero errors. Run fingerprint `dab15af5d3ab...`.

| Config | F1 [95% CI] | EM | Cite-P | Cite-R | Ret-R | Abst-Acc | Over-abst | Lat (s) |
|---|---|---|---|---|---|---|---|---|---|
| A: dense | 0.0604 [0.038, 0.087] | 0.0000 | 0.079 | 0.114 | 0.227 | 0.961 | 0.000 | 0.64 |
| B: dense+abstain | 0.0604 [0.038, 0.087] | 0.0000 | 0.079 | 0.114 | 0.227 | 0.158 | 0.877 | 0.74 |
| C: hybrid | 0.0719 [0.045, 0.103] | 0.0082 | 0.095 | 0.133 | **0.267** | 0.961 | 0.000 | 0.98 |
| D: hybrid+rerank | 0.0695 [0.043, 0.101] | 0.0164 | 0.118 | 0.143 | 0.245 | 0.961 | 0.000 | 1.16 |
| E: full | 0.0695 [0.043, 0.101] | 0.0164 | 0.118 | 0.143 | 0.245 | 0.173 | 0.861 | 1.15 |

Paired retrieval comparisons: A→C +0.0400 (`p < 0.0001`); C→D −0.0221
(`p = 0.0014`); A→E +0.0179 (`p = 0.0218`). F1 changes are not significant.

**Clean negative-enriched config E**
(`results/milestone2/balanced-clean-final/`): 51 documents, 231 questions
(171/60), **3,784/3,784 unique chunks**, zero errors. F1 0.0553, EM 0.0058,
Cite-P 0.0346, Cite-R 0.1029, Ret-R 0.1830. Run fingerprint
`5b6f98b8586c...`. The CLI retains `--balanced` for compatibility, but this
sample is 74% answerable and is now called negative-enriched.

**Descriptive sweeps:** natural AUC 0.6459; negative-enriched AUC 0.6181 and
in-sample peak balanced accuracy 0.5662 at threshold 0.10. These do not select a
test operating point.

**Document-disjoint threshold protocol:** threshold 0.25 is selected on 31
negative-enriched calibration documents (101/40) and locked for 20 held-out
documents (70/20). Test AUC is 0.6443 [0.5322, 0.7467], balanced accuracy is
0.5179 [0.4289, 0.5938], and coverage is 0.1778. The interval includes chance.
The natural held-out partition has only two negatives and is diagnostic only.

**Verification:** `python -m pytest tests/ -q` → **35 passed**.
`results/milestone2/integrity_verification.json` confirms record counts,
manifests, unique chunks, cache metadata, record-only summary/sweep reproduction,
and disjoint calibration/test documents.

### 1.6 Full-QASPER evaluation — current primary result

Authoritative output: `results/milestone3/full-qasper-validation-ae-v1/`.
Run fingerprint: `77f3ef35739c...`. The validation split contains 281 papers and
1,003 evaluable questions (943 answerable / 60 unanswerable; 2 dropped), indexed
as **21,516 stored / 21,516 unique chunks**. All 5,015 records completed with
zero errors in 3,222.5 seconds; peak RSS was 1,874 MiB.

| Config | F1 [document-cluster 95% interval] | EM | Cite-P | Cite-R | Ret-R |
|---|---:|---:|---:|---:|---:|
| A: dense | 0.0562 [0.0472, 0.0651] | 0.0032 | 0.0409 | 0.1195 | 0.2283 |
| B: dense+abstain | 0.0562 [0.0472, 0.0651] | 0.0032 | 0.0409 | 0.1195 | 0.2283 |
| C: hybrid | 0.0624 [0.0530, 0.0728] | 0.0064 | 0.0568 | 0.1409 | **0.2671** |
| D: hybrid+rerank | 0.0625 [0.0523, 0.0729] | 0.0095 | 0.0608 | 0.1352 | 0.2392 |
| E: full | 0.0625 [0.0523, 0.0729] | 0.0095 | 0.0608 | 0.1352 | 0.2392 |

Paired document bootstrap: hybrid improves retrieval recall by +0.0388
[0.0332, 0.0448]; reranking reduces it by −0.0279 [−0.0349, −0.0212]. These
effects persist on the 241 papers outside the prior 40-paper run.

The predeclared split uses 169 calibration papers (600 questions, 560/40) and
112 held-out papers (403 questions, 383/20). Calibration selects threshold 0.10.
Held-out AUC is 0.6636 [0.5286, 0.7912], balanced accuracy 0.5686
[0.4773, 0.6441], and coverage 0.2804. The balanced-accuracy, conditional-F1 and
covered-F1 differences versus always-answer all have document-bootstrap
intervals crossing zero. The system answers 3/20 unanswerable questions and
declines 273/383 answerable questions.

All independent checks pass: exact counts, unique index, cache/source/model
fingerprints, chunk provenance, summary/sweep reproduction, predeclared split,
2,000 bootstrap intervals, and 635/635 scientifically identical records on the
Milestone 2 overlap. Final test suite: **38 passed**. See
`docs/FULL_QASPER_EVALUATION.md`.

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
   clean paired results p < 0.0001 and p = 0.0014.

### 2.10 Substantive findings

- **Hybrid retrieval significantly improves evidence recall** over dense-only
  (0.227 → 0.267, p < 0.0001). Largest, most reliable effect in the study.
- **The cross-encoder reranker significantly *hurts* evidence recall**
  (0.267 → 0.245, p = 0.0014) without improving answer F1. Domain
  mismatch: MS MARCO web passages vs. reference-dense scientific paragraphs
  containing `BIBREFnn` placeholders. This is the paper's negative result.
- **F1 0.06–0.07 is partly a metric mismatch.** QASPER gold answers are
  abstractive (`multilingual NMT (MNMT) BIBREF19`); the reader is extractive and
  returns the shorter span (`multilingual NMT approaches`).
- **Held-out abstention is weak.** The negative-enriched locked threshold reaches
  balanced accuracy 0.5179 [0.4289, 0.5938], statistically compatible with chance.
  The historical 0.019 baseline and 4.4× gain are invalid and retained only in the
  audit trail.
- **A fixed threshold of 0.5 over-abstains** (0.861 natural; 0.883 enriched).
  Thresholds must be selected on calibration documents and locked before test.

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
| `vistaar/config.py` | frozen settings, env normalisation, reader switch and exact model revisions |
| `vistaar/text.py` | PDF/text extraction, boundary-aware recursive chunking with provenance (`doc_id`, `page`, `chunk_id`, `ordinal`) |
| `vistaar/index.py` | namespaced FAISS/BM25, strict duplicate rejection, document upsert, validated snapshot persistence, RRF and reranker |
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
| `eval/run_eval.py` | matrix runner with isolated indices, run manifests, fingerprinted caches, integrity checks and resource metadata |
| `eval/sweep.py` | threshold sweep → risk–coverage curve, AUC, balanced accuracy, selective F1 |
| `eval/abstention_signals.py` | offline comparison of confidence, retrieval, and combined abstention signals |
| `eval/provenance.py` | canonical SHA-256 run/config/item fingerprints and fail-closed cache validation |
| `eval/threshold_protocol.py` | document-disjoint calibration/test and grouped-CV diagnostics |
| `eval/compare_results.py` | record-only reproduction and historical-to-clean comparison |
| `eval/full_qasper_analysis.py` | full-run cluster intervals, overlap verification and failure extraction |

### Tests, data, results
| File | Purpose |
|---|---|
| `tests/` | 38 unit/regression tests, no model downloads |
| `data/raw/qasper_validation.parquet` | QASPER validation (gitignored) |
| `data/raw/cuad_v1.json` | CUAD v1, 510 contracts (gitignored, **not yet used**) |
| `results/`, `results/balanced/` | preserved historical Milestone 1 outputs; not current claims |
| `results/milestone2/natural-clean-final/` | authoritative clean natural A–E run and sweep |
| `results/milestone2/balanced-clean-final/` | authoritative negative-enriched E run, sweep, signals and threshold protocol |
| `results/milestone2/comparison-final/` | historical-to-clean JSON/Markdown comparison |
| `results/milestone2/integrity_verification.json` | machine verification of final artifacts |
| `results/cache/eval/<run fingerprint>/` | isolated indices and validated per-config caches (gitignored) |
| `results/milestone3/full-qasper-validation-ae-v1/` | full 281-paper run, sweep, threshold protocol and independent analyses |

### Documentation and delivery
| File | Purpose |
|---|---|
| `paper/PAPER.md` | draft v1.2 with clean results and held-out threshold protocol |
| `docs/AUDIT.md` | 23 defects in the original codebase with fixed/open status |
| `docs/ABSTENTION_AUDIT.md` | independent Milestone 1 scientific audit |
| `docs/MILESTONE2_REPORT.md` | integrity fixes, clean comparisons, protocol and scaling recommendation |
| `docs/FULL_QASPER_EVALUATION.md` | independently verified full-run report and next-step recommendation |
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

Milestone 1's numerical claims were subsequently audited; §1.4 preserves them as
history rather than current evidence.

### Milestone 2 — Evaluation integrity and reproducibility — ✅ completed

- Fixed duplicate index accumulation with strict append and document upsert.
- Added exact model revisions and run/config/item fingerprints.
- Isolated evaluation indices/caches and preserved every historical artifact.
- Added smoke rerun, cache invalidation and cross-run isolation tests.
- Reran natural and negative-enriched samples from fresh indices.
- Added document-disjoint threshold calibration/test and grouped CV.
- Corrected the stale 0.019/4.4× claim and updated paper/README/state.
- Final integrity verification passes; see `docs/MILESTONE2_REPORT.md`.

No commit was created automatically.

### Milestone 3 — Controlled full-QASPER evaluation — ✅ completed

- Completed all 5,015 A–E records over 281 papers and verified them independently.
- Preserved the predeclared paper split and selected threshold only on calibration.
- Added document-cluster aggregate and paired uncertainty estimates.
- Confirmed the hybrid retrieval gain and reranker retrieval loss at full scale.
- Found no robust held-out abstention utility improvement over always-answer.
- Added failure and label-sensitivity analyses and reconciled the final report.
- No commit or push was performed.

### Milestone 4 — Add the legal/contract domain (CUAD), or scope explicitly

`data/raw/cuad_v1.json` is downloaded but unused. After the controlled QASPER run,
add a CUAD loader and independently fingerprinted retrieval/abstention evaluation,
or retain the explicit single-domain limitation.

In parallel close the
production gaps already listed in `docs/AUDIT.md`: proxy-aware rate limiting
(`X-Forwarded-For`), authentication/tenant isolation beyond session namespaces,
OCR for scanned PDFs, and a Docker build + CI run to prove `requirements.txt`
resolves from scratch.

---

## 5. How to resume

```bash
cd "/Users/kishansingh/Documents/STUDY/Research Paper/Vistaar"

./.venv/bin/python -m pytest tests/ -q            # 38 tests
./.venv/bin/python -m eval.run_eval --papers 40 --workers 2 --out <new-output-dir>
./.venv/bin/python -m eval.sweep --records <new-output-dir>/records.jsonl --config E --out <new-output-dir>
./.venv/bin/python -m eval.threshold_protocol --records <new-output-dir>/records.jsonl --config E --out <new-output-dir>/threshold_protocol
./.venv/bin/uvicorn app:app --reload              # http://127.0.0.1:8000
```

Notes for whoever resumes:

- Never reuse or overwrite `results/` or `results/balanced/`; they are the
  checksummed historical audit trail.
- New caches live under `results/cache/eval/<run fingerprint>/`. Any relevant
  dataset, document, code, model, prompt or config change creates a new run path.
- `run_eval` refuses non-empty output directories; choose a new path for every
  experiment. Identical inputs can safely reuse their fingerprinted cache.
- If the Gemini path is ever needed: the quota resets daily at 20 requests. Verify
  `thinkingConfig.thinkingBudget = 0` is still present before debugging anything else.
