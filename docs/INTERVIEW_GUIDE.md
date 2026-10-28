# Interview Guide

Use the cited reports as the source of truth. Be ready to distinguish a verified result from a documented rationale and from a reconstructed explanation. Related records: [decisions](DECISION_LOG.md), [challenges](CHALLENGE_LOG.md), [trade-offs](TRADEOFFS.md), and [milestones](MILESTONES.md).

> **Authorship boundary:** Git history proves project changes, not who personally performed each task. Use first-person wording only for work you personally did and can explain under follow-up. Where authorship is unknown, use project-neutral wording such as “the implementation” or “the project.”

## Project in one minute

“Vistaar is a citation-grounded retrieval-augmented QA system with a thresholded answer/abstain decision. Its QASPER evaluation shows why evaluation integrity matters as much as model choice: persistent index state and incomplete cache identity distorted early results. After run isolation and a document-disjoint protocol, hybrid retrieval improved evidence recall, while the general-domain reranker reduced it through top-four evidence displacement. The current evidence does not support a useful abstention claim. Recent work instruments confidence signals and freezes a future experiment protocol; it has not implemented or fitted a calibrator.”

## Architecture walkthrough

1. **Ingest and preserve provenance.** The API validates type and size, extracts PDF pages or UTF-8 text, and creates boundary-aware chunks of up to 800 characters with 150-character overlap. Each chunk retains `doc_id`, `chunk_id`, one-based page, section, and ordinal; persistent document character offsets are not stored.
2. **Build isolated retrieval indexes.** Normalized `all-MiniLM-L6-v2` embeddings go to a per-namespace FAISS `IndexFlatIP`; the same chunks form a per-namespace BM25 index. Namespaces prevent accidental session mixing but do not provide authentication or authorization.
3. **Retrieve and fuse.** With the standard `top_k=8`, dense and BM25 each expose up to 24 candidates. Hybrid mode deduplicates and keeps up to 16 RRF candidates. A reranking configuration cross-encodes that pool and keeps four; without reranking, the reader still consumes at most the first four passages.
4. **Extract and score an answer.** The local SQuAD2 reader concatenates labelled passages, tokenizes 384-token overflow windows with stride 128, searches spans up to 48 tokens, and computes `sigmoid(best_span_score - selected_window_null_score)`. This is a threshold score, not a calibrated probability.
5. **Map the answer to a citation.** The reader records each labelled passage's character range in the concatenated context. The selected token offsets produce a context character start, which identifies the containing passage; the pipeline then emits that chunk's ID, page, and text. It does not claim persistent original-document character offsets.
6. **Apply the decision rule.** In local abstention configurations, Vistaar declines when no span is found or confidence falls below the configured threshold. Retrieval and citation metrics are evaluation outputs, not decision inputs.

## Stories to prepare

### 1. A result that changed after an audit

- **Situation:** Early natural-sample abstention results appeared stronger than warranted.
- **Evidence/root cause:** `docs/ABSTENTION_AUDIT.md` found duplicate chunks from persistent indexing, cache keys without full provenance, a stale sweep using post-abstention predictions, and thresholds selected on evaluation records. Clean indexing changed natural AUC from 0.6959 to 0.6459.
- **Action:** The project added strict duplicate handling, run-scoped stores/caches, input fingerprints, raw-span sweep logic, explicit empty-span handling, and document-disjoint calibration/test.
- **Result/limit:** The **Milestone 2 negative-enriched 20-paper holdout** reaches balanced accuracy 0.5179 [0.4289, 0.5938]. The later authoritative Milestone 3 full-QASPER holdout uses 112 papers / 403 questions and reaches 0.5686 [0.4773, 0.6441] at 0.2804 coverage; it answers 3/20 unanswerable questions and declines 273/383 answerable questions. Both intervals include chance, and exact provenance for every historical cached prediction cannot be recovered.
- **Likely follow-ups:** What was the root cause? How did duplicate candidates affect ranking? Which old claims were withdrawn? How did you prevent a similar issue?

### 2. A useful negative result

- **Situation:** Compare dense, hybrid, and reranked retrieval on scientific papers.
- **Evidence:** Clean 40-paper results report Ret-R 0.227 for dense, 0.267 for hybrid, and 0.245 after reranking. The full-QASPER report confirms the hybrid gain and reranker loss; answer-F1 impact from reranking is not significant.
- **Confirmed mechanism:** Reranking displaces evidence-bearing RRF candidates at the top-four cutoff more often than it adds useful evidence. MS MARCO/scientific-domain mismatch is a plausible but unconfirmed hypothesis; no scientific-domain reranker or manual relevance study isolates it.
- **Likely follow-ups:** Define Ret-R; explain why retrieving more gold evidence may not improve extractive F1; describe how D-anchor would test a recall-preserving approach.

### 3. Choosing reproducibility over hosted inference

- **Situation:** Evaluation needed a controlled path, while hosted requests were quota-bound and MPS failed in the recorded environment.
- **Action:** The reported metrics use the local CPU extractive reader; Gemini remains an optional application path.
- **Trade-off/limit:** Offline repeatability versus speed and abstractive answer quality. No hosted-reader comparison is part of the reported results.
- **Likely follow-ups:** How would you design a controlled reader comparison? Which factors must be pinned? Why does extractive output mismatch QASPER references?

### 4. Instrumentation without changing the system

- **Situation:** Future confidence research needs inference-time retrieval and reader signals, but ordinary callers and records should remain stable.
- **Action/evidence:** Milestone 4B adds opt-in diagnostics; current shadow work uses explicit local-only mode, a distinct cache namespace, schema validation, and tests for behavior preservation and deterministic protocol calculations.
- **Limit:** Fixture-level tests do not prove equivalence on a full dataset. Do not say a calibrator was trained or that shadow results improved performance.
- **Likely follow-ups:** What is inference-time versus label-derived? Why encode unavailability separately from zero? What data must exist before confirmation?

## Metrics and claims to know

| Metric | What it measures here | What it does **not** establish |
|---|---|---|
| EM / token F1 | Normalized answer-string overlap with one or more gold answers. | Semantic equivalence, evidence retrieval, citation correctness, or whether answering was appropriate. |
| Ret-R@4 | Fraction of gold-evidence vocabulary covered by the union of the four passages presented to the reader. | Semantic sufficiency, entailment, or causal responsibility for the answer. |
| Cite-P | Fraction of cited chunks for which at least 50% of cited-chunk vocabulary appears in gold evidence. | Human attribution correctness; broad chunks can contain the answer yet fail the containment threshold. |
| Cite-R | Fraction of gold-evidence vocabulary covered by cited chunks. | That every cited statement is supported or that paraphrases are handled correctly. |
| Confidence AUC | How well the score ranks answerable above unanswerable questions. | Probability calibration or quality at a chosen threshold. |
| Balanced accuracy | Mean answerable recall and unanswerable rejection rate at one threshold. | Deployment utility without coverage, prevalence, and error costs. |
| Coverage | Fraction of all questions answered at a threshold. | Correctness of those answers. |
| Conditional F1 | First-gold raw-answer F1 among covered questions known to be answerable. | Total selective-system utility; answered-unanswerable cases are outside its denominator. |
| Covered F1 | F1 over all covered questions, assigning zero to answered-unanswerable cases. | Semantic correctness beyond lexical overlap. |

**Calibration versus thresholding:** selecting a threshold does not make the
underlying score a calibrated probability. No ECE, Brier, or reliability claim
has been established for the current reader score.

## Provenance and cache explanation

- `stable_fingerprint` hashes canonical JSON with SHA-256.
- A run fingerprint covers dataset bytes and partition role, selected question order, document content and labels/gold data, chunking, exact embedding/reranker/reader revisions, retrieval and reader settings, prompt/source hashes, worker count, packages, platform, diagnostics schema, and relevant implementation files.
- Each evaluation item separately fingerprints the question, document content, title, gold answers/evidence, and answerability label. Retrieval configurations have their own fingerprints.
- A cached row is reusable only when its run, config, item, and record schema contracts match. A mismatch is a cache miss; malformed or rejected rows are not silently accepted.
- Run-scoped index/cache directories prevent the earlier hidden shared state from contaminating a new run. Relevant code or data changes intentionally invalidate reuse.

## What to study before the interview

- Trace one question through chunk provenance, dense/BM25 retrieval, RRF, optional reranking, reader span/null scoring, passage-range citation mapping, and threshold decision in `vistaar/`.
- Explain run and item fingerprints in `eval/provenance.py`; identify which dataset, document, model, code, config, and environment inputs invalidate a cache.
- Reproduce the logic (not the experiment) behind document-level splitting and threshold locking in `eval/threshold_protocol.py`.
- Know why the 40-paper sample had only five negatives, why the negative-enriched sample is not natural prevalence, and why paper-level bootstrap is used.
- Read the limitations in `docs/FULL_QASPER_EVALUATION.md`, `docs/MILESTONE2_REPORT.md`, and `docs/MILESTONE5B_SHADOW_PROTOCOL.md` before making claims about generalization or future models.

## Gaps to state plainly

- Git contains six commits and provides a short, dated checkpoint trail, not a full engineering diary. It cannot recover every abandoned approach, discussion, experiment command, or personal contribution unless preserved in committed documentation or artifacts.
- Test counts are checkpoint-specific: 12 at the first audit, 35 at Milestone 2, 38 after Milestone 3. Current Milestone 5B test files are uncommitted; this audit did not run them.
- The exact historical cache provenance for every early prediction is unrecoverable.
- The published evidence is English QASPER only; CUAD is recorded as unused, and no human evaluation is documented.
- Official QASPER train/test artifacts and disjointness remain unverified locally. The future calibrator workflow is protocol-only, not implemented or fitted.
- The repository documents technical work but does not establish which tasks were personally performed by the user versus collaborators. Attribute only work you can personally defend and explain.

## Practice questions

1. What failure most changed your conclusion, and how did you find it?
2. Why did you use document-disjoint rather than question-level splits?
3. What does the reranker negative result tell you, and what does it not tell you?
4. Why is confidence AUC not evidence of calibrated probabilities?
5. What was the exact fix for cache contamination, and how do fingerprints prevent recurrence?
6. How would you validate D-anchor without tuning on the final test set?
7. Which current limitations would block production deployment?
