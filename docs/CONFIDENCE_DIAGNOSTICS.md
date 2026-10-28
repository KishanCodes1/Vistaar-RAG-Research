# Confidence Diagnostics Schema

**Schema:** `vistaar-confidence-diagnostics-v1`

Vistaar can collect confidence diagnostics as an optional side channel. Existing
callers that do not pass a diagnostics dictionary receive the same retrieval and
answer payload as before. The evaluation runner opts in and stores the resulting
object in each record's additive `diagnostics` field.

Diagnostics contain only values available during inference. Gold answers,
answerability labels, gold evidence, F1, citation correctness, and evidence
coverage are never diagnostic features.

## Collection API

```python
diagnostics = {}
hits = pipeline.retrieve(question, namespace, config, diagnostics=diagnostics)
answer = pipeline.answer(
    question,
    namespace,
    config,
    hits=hits,
    diagnostics=diagnostics,
)
```

`Answer.to_dict()` includes `diagnostics` only when collection was requested.
The evaluator records diagnostics and fingerprints the schema/version and
`vistaar/diagnostics.py` implementation in the run manifest. This prevents old
caches from being mistaken for instrumented records.

## Opt-in computation boundary

Diagnostic collection is disabled by default. With no diagnostics dictionary:

- retrieval does not copy the pre-rerank candidate list for logging and does not
  serialize candidates, upstream ranks, score summaries, or overlap features;
- the reader does not instantiate diagnostic state, count candidate starts/spans,
  track the second-best span, calculate null-score ranges, or construct
  `ReaderDiagnostics`;
- `Answer.to_dict()` retains its legacy keys and payload shape.

Ordinary inference still performs the calculations required for its existing
behavior: dense/BM25 retrieval, RRF, optional reranking and truncation, tokenizer
overflow windows, start-probability pruning, best-span search, each window's null
score, the selected window's span/null margin, sigmoid confidence, citations, and
the existing abstention comparison. Diagnostics reuse those required values
rather than recomputing them.

When a diagnostics dictionary is supplied, the pipeline passes
`collect_diagnostics=True` to the local reader and enables only the additional
state described above.

## Top-level fields

| Field | Type | Source | Inference-time? |
|---|---|---|---:|
| `schema_version` | string | `vistaar.diagnostics.CONFIDENCE_DIAGNOSTICS_SCHEMA` | Yes |
| `retrieval` | object | `RAGPipeline.retrieve` | Yes |
| `reader` | object or null | local-reader instrumentation or explicit unavailable reason | Yes |
| `decision` | object or null | `RAGPipeline.answer` | Yes |

## Retrieval object

### Configuration and limits

`retrieval.namespace` identifies the searched namespace. `retrieval.config`
records the name, `use_bm25`, `use_rerank`, `use_abstention`, `top_k`, and
`rerank_top_n`. `retrieval.limits` records the requested dense, BM25, RRF, and
reranker-output limits. These values describe the current query, not global
defaults inferred later.

### Stages

`retrieval.stages` contains `dense`, `bm25`, `rrf`, `reranker`, and
`selected_context`. Every optional stage has:

| Field | Meaning |
|---|---|
| `requested` | Whether the retrieval config requested this stage. |
| `available` | Whether scores were produced. |
| `reason` | `not_requested`, `model_unavailable`, `empty_namespace`, `externally_supplied_hits`, or null. |
| `candidates` | Ranked candidate records, or null when unavailable. An available empty stage uses `[]`. |
| `score_summary` | Count, top score, top1–top2 margin, mean, standard deviation, minimum, maximum, and range; null for no scored candidates. |

Candidate records contain:

| Field | Meaning |
|---|---|
| `rank` | Rank within this stage, starting at one. |
| `chunk_id`, `doc_id`, `page`, `section` | Deterministic chunk provenance. `page` is a positive, one-based integer; no negative sentinel is used. |
| `source`, `score` | Stage label and native stage score. |
| `dense_rank`, `bm25_rank`, `fused_rank` | Upstream rank where known; otherwise null. |

The dense stage exposes up to `top_k * 3` candidates. BM25 exposes up to the
same limit but only positive-score candidates. RRF exposes up to `top_k * 2`
deduplicated candidates. When reranking is available, its diagnostic stage
contains the complete scored RRF pool before `rerank_top_n` truncation.

`retrieval.features` currently contains `dense_bm25_overlap_count` and
`selected_chunk_count`. It intentionally does not contain any gold-evidence
measurement.

### Score limitations

- Dense scores are normalized-vector inner products.
- BM25 scores have paper/query-dependent scale.
- RRF scores are rank-fusion values rather than relevance probabilities.
- Reranker scores are raw cross-encoder outputs and are not probabilities.
- Comparing raw magnitudes across stages or queries requires a train-only
  normalization/model; diagnostics do not perform that transformation.

## Reader object

For the local reader, `reader` contains:

| Field | Type | Source / interpretation |
|---|---|---|
| `available` | bool | True when local span/null diagnostics were computed. |
| `found` | bool | Existing local-reader span-found status. |
| `passage_count` | int | Number of selected passages presented to the reader. |
| `context_characters` | int | Length of the concatenated labelled context. |
| `window_count` | int | Number of tokenizer overflow windows. |
| `candidate_start_count` | int | Context starts surviving the existing 0.02 start-probability filter, summed over windows. |
| `candidate_span_count` | int | Spans evaluated after start filtering and answer-length constraints. |
| `best_span_score` | float or null | Selected start-logit plus end-logit score. |
| `second_best_span_score` | float or null | Runner-up among spans already evaluated by the unchanged search loop. |
| `selected_window` | int or null | Overflow window containing the selected span. |
| `selected_start_token`, `selected_end_token` | int or null | Selected token indices in that window. |
| `selected_null_score` | float or null | CLS start/end score from the selected span's window. |
| `null_score_min`, `null_score_max` | float or null | Range of per-window CLS null scores. Descriptive only. |
| `span_null_margin` | float or null | `best_span_score - selected_null_score`, before sigmoid. |
| `selected_passage` | int or null | One-based passage containing the answer start. |
| `no_span_reason` | string or null | `no_retrieved_passages`, `no_candidate_span`, or `empty_decoded_span`. |

The recorded second span and null range are diagnostics only. They do not alter
the selected span or current confidence. Candidate counts reflect the existing
start-probability pruning, not all mathematically possible token spans.

The hosted reader explicitly reports local reader diagnostics as unavailable;
it records only the hosted `found` decision and existing binary confidence.

## Decision object

| Field | Meaning |
|---|---|
| `backend` | `local` or `gemini`. |
| `abstention_enabled` | Retrieval config's existing abstention flag. |
| `configured_threshold` | Existing reader or dense-retrieval threshold where applicable; null when no scalar hosted threshold is used. |
| `found` | Reader/model found status; null if hosted inference was skipped. |
| `confidence` | Existing decision confidence, never recomputed from diagnostics. |
| `abstained` | Existing final abstention decision. |
| `triggered_reasons` | Conditions that caused the actual abstention. |
| `counterfactual_conditions` | Local-reader conditions that would trigger abstention if enabled. |

Diagnostics never select a threshold or override `abstained`.

## Behavior-equivalence contract

With collection enabled or disabled, the following must remain identical:

- dense/BM25/RRF/reranker ordering and selected chunk IDs;
- cross-encoder scores and top-k truncation;
- reader answer span and citations;
- confidence, configured threshold, and abstention decision;
- final answer and raw answer.

Focused tests run instrumented and uninstrumented calls against the same
deterministic fixture and compare these business outputs. The reranker test also
proves that exposing all scored candidates does not change its selected hits.

A one-question, in-memory QASPER fixture with the pinned real embedding,
reranker, and reader models also produced identical selected chunk IDs/scores,
answer, raw answer, citations, confidence, and abstention decision with
collection disabled versus enabled. This is a smoke equivalence check, not proof
over the full input distribution; no full-QASPER rerun was performed.

## Known limitations

- Instrumentation adds score bookkeeping, candidate-list copying,
  serialization, and memory overhead only when enabled. This overhead has **not
  been benchmarked**.
- Candidate text is not duplicated inside every diagnostic stage; chunk IDs map
  to the indexed provenance, while final selected texts remain in the existing
  record fields.
- No diagnostic is evidence that a score is probabilistically calibrated.
- Gold-based metrics may be joined during offline evaluation, but must never be
  passed to an inference-time gate or learned feature pipeline.
- Existing Milestone 3 records predate this schema and legitimately lack the
  `diagnostics` field. Consumers must treat absence as an older schema, not fill
  missing values with zeros.
- Full-QASPER behavior equivalence has not been run. Real-model equivalence has
  been checked only on the one-question fixture described above.
