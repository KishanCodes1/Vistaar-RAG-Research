# Decision Log

This log records decisions supported by repository evidence. “Documented” means an existing project record states the rationale; “reconstructed” means the choice follows from code or results but was not recorded as an explicit decision at the time. See [milestones](MILESTONES.md), [challenges](CHALLENGE_LOG.md), [trade-offs](TRADEOFFS.md), and the [interview guide](INTERVIEW_GUIDE.md) for related context.

## D1 — Namespace document indexes

- **When:** Initial implementation; present in commit `130398e` (2026-10-27).
- **Problem/evidence:** A single module-global FAISS index mixed every user's documents in one retrieval collection. The original-code audit lists this as a critical defect.
- **Decision/status:** Use separate namespace indexes and synchronize store operations to prevent accidental cross-session mixing. Documented in `docs/AUDIT.md`; code is in `vistaar/index.py` and `vistaar/api.py`. Namespace names are caller-supplied capabilities, not an authorization boundary.
- **Alternatives:** No contemporaneous alternatives are recorded. Per-user durable identity/authentication was not implemented; session namespaces are not a substitute for authentication or full tenant management.
- **Verification/limits:** Initial tests cover namespace isolation; concurrent production behavior and real multi-tenant authorization are not established by that test.
- **Interview framing:** “The original global index mixed sessions. Separate namespace indexes and synchronized access prevent that accidental mixing, but the API still lacks authentication and authorization.”
- **Likely follow-ups:** How is a namespace chosen and expired? What changes are needed for authenticated tenants? What concurrency guarantee does the lock provide?

## D2 — Make citations trace to source chunks

- **When:** Initial implementation; commit `130398e` (2026-10-27).
- **Problem/evidence:** The audited original had no page/chunk provenance, so citations could not be checked. Fixed-length character chunking also split text at arbitrary boundaries.
- **Decision/status:** Preserve document, page, ordinal, and chunk identifiers through boundary-aware chunking and retrieval. Documented in `docs/AUDIT.md`; implementation is in `vistaar/text.py`, `vistaar/index.py`, and `vistaar/pipeline.py`.
- **Alternatives:** No explicit alternatives are recorded. “Boundary-aware” was chosen over the audited fixed character split; a comparative chunking experiment is not documented.
- **Trade-off/verification:** More metadata and boundary-aware splitting support traceability but do not establish semantic correctness of a citation. Tests cover chunk bounds and retrieval deduplication; see `tests/test_core.py` and `tests/test_index_lifecycle.py`.
- **Interview framing:** “A citation is useful only if it maps back to a real source location, so provenance travels with each chunk through retrieval.”
- **Likely follow-ups:** What happens when a PDF has no usable page text? How does the reader map a selected span in its concatenated context back to the source passage and page?

## D3 — Use a local extractive reader for reproducible evaluation

- **When:** Initial implementation; commit `130398e` (2026-10-27); reaffirmed in `PROJECT_STATE.md`.
- **Problem/evidence:** The hosted Gemini path was quota-bound and unsuitable for the controlled reported evaluation. The project state records that MPS inference aborted on this workload; no local generative model was available in the recorded environment.
- **Decision/status:** Use the local CPU RoBERTa SQuAD2 span reader for reported metrics; retain Gemini as an optional application backend. Rationale is documented in `PROJECT_STATE.md` and `README.md`; code is in `vistaar/reader.py` and `vistaar/generate.py`.
- **Alternatives:** Hosted generation was tried but could not support the evaluation then; GPU/MPS was ruled out by the recorded crash. No controlled quality comparison of local and hosted readers is documented.
- **Trade-off/verification:** A pinned local CPU path supports controlled, repeatable evaluation, while extractive answers mismatch abstractive QASPER gold answers and score low on exact-match/F1. All reported metrics use the local reader.
- **Interview framing:** “The reported evaluation uses the pinned local CPU reader because hosted quotas and the recorded MPS failure prevented a controlled alternative run. Gemini remains a separate, unverified product path.”
- **Likely follow-ups:** How does extractive evaluation bias F1? What would be required to compare readers fairly?

## D4 — Replace informal abstention claims with measured thresholding

- **When:** Milestone 1, commit `e87fa0c` (2026-10-27), then revised after the audit in Milestone 2.
- **Problem/evidence:** Early natural-sample results had only five negatives, the “calibrated” wording lacked calibration metrics, and a stale sweep overstated selective-F1 improvement. `docs/ABSTENTION_AUDIT.md` documents the reproducibility and threshold-selection defects.
- **Decision/status:** Describe the system as thresholded abstention, report natural and negative-enriched results with limits, and use raw reader spans for counterfactual sweeps. Then add document-disjoint calibration/test evaluation. See `docs/MILESTONE2_REPORT.md`, `paper/PAPER.md`, and `eval/threshold_protocol.py`.
- **Alternatives:** Reporting the natural sample alone and calling the score calibrated were rejected in the committed paper revision. The audit recommends nested grouped CV where a fixed holdout is too small; the later report also includes grouped-CV diagnostics.
- **Trade-off/verification:** A held-out split reduces direct threshold leakage but has limited negative support. On the **Milestone 2 negative-enriched 20-paper holdout**, locked balanced accuracy is 0.5179 [0.4289, 0.5938], compatible with chance. The later Milestone 3 full-QASPER result is also unresolved: 0.5686 [0.4773, 0.6441] at 0.2804 coverage on 112 held-out papers / 403 questions; it answers 3/20 unanswerable questions and declines 273/383 answerable questions.
- **Interview framing:** “The audit showed that the early natural-sample result depended on five negatives, contaminated state, and post-hoc threshold choice. The project replaced that claim with locked, paper-grouped evaluation, whose Milestone 2 and Milestone 3 intervals still include chance.”
- **Likely follow-ups:** Why is AUC not calibration? Why report coverage and answered-unanswerable rate? What does the interval permit you to claim?

## D5 — Isolate runs and fail closed on provenance

- **When:** Milestone 2, documented in `docs/MILESTONE2_REPORT.md`; published in commit `28c9b3f` (2026-10-28).
- **Problem/evidence:** The independent audit found persistent index append duplicates and caches keyed without full input fingerprints. Retrieval changed on a clean rerun; the historical natural AUC changed from 0.6959 to 0.6459.
- **Decision/status:** Reject duplicate chunk IDs, scope indexes and caches to a fingerprinted run, include code/data/model/config provenance, preserve outputs, and refuse implicit overwrite. See `eval/provenance.py`, `eval/run_eval.py`, `vistaar/index.py`, and `tests/test_eval_cache.py`.
- **Alternatives:** Repairing the historical shared store in place was not chosen; the report says historical state was preserved and clean run-scoped stores used.
- **Trade-off/verification:** Stronger provenance costs storage and invalidates caches when relevant inputs change. The report records 35 passing tests and repeat-run smoke checks; the current project state later records 38 tests after Milestone 3.
- **Interview framing:** “The root cause was hidden state: repeated indexing and under-specified cache keys made metrics depend on prior runs. Run identity is now explicit, and reuse is conditional on matching fingerprints.”
- **Likely follow-ups:** Which inputs enter the fingerprint? How do you preserve useful cache reuse? How did the clean rerun change the scientific conclusion?

## D6 — Keep the full-QASPER holdout grouped by paper

- **When:** Milestone 3, commit `28c9b3f` (2026-10-28).
- **Problem/evidence:** Questions from the same paper share topic and evidence; splitting questions independently could leak paper-specific information between calibration and test. The protocol and report specify paper-disjoint partitions.
- **Decision/status:** Select the threshold using calibration papers, lock it, then evaluate once on held-out papers; use paper-cluster uncertainty estimates. See `docs/FULL_QASPER_EVALUATION.md` and `eval/threshold_protocol.py`.
- **Alternatives:** Question-level splitting is explicitly disallowed by the protocol. The exact selection objective was balanced accuracy for the published operating point; later plans propose AURC for future research.
- **Trade-off/verification:** Grouping better reflects generalization across papers. Milestone 3 uses 169 calibration papers / 600 questions and **112 held-out papers / 403 questions (383 answerable, 20 unanswerable)**. The 20-paper holdout belongs to Milestone 2's negative-enriched analysis, not full QASPER. Milestone 3 held-out balanced accuracy is 0.5686 [0.4773, 0.6441], still compatible with chance.
- **Interview framing:** “The protocol groups by the unit that can carry correlated content—the paper—and locks the threshold before scoring the held-out partition.”
- **Likely follow-ups:** Why group by paper? How sensitive is the result to mixed annotations? What would a larger independent test set change?

## D7 — Collect diagnostics as an opt-in shadow side channel

- **When:** Milestone 4B, commit `8c400eb` (2026-10-28); Milestone 5B work is currently uncommitted.
- **Problem/evidence:** The Milestone 4 research plan identifies retrieval and reader signals for future confidence research, but ordinary records did not preserve all inference-time signals. The diagnostics schema explicitly excludes gold-derived data.
- **Decision/status:** Make collection opt-in and additive, preserve ordinary business outputs, version the schema, and isolate shadow caches. The committed baseline is `8c400eb`; current working-tree files add shadow workflow guards, cache validation, deterministic protocol helpers, and tests.
- **Alternatives:** No explicit contemporaneous alternative list is present. Reconstructed rationale: collecting every signal in ordinary production/evaluation could add overhead and complicate legacy records, so collection is scoped to an explicit local-only run.
- **Trade-off/verification:** The schema and fixtures test invariance and fail-closed validation, but fixture-level agreement does not demonstrate full-distribution equivalence. Current shadow instrumentation has not been run as an experiment in this task.
- **Interview framing:** “Observability is explicit and versioned, and diagnostics use an additive local shadow path so the ordinary inference contract stays stable in tested fixtures.”
- **Likely follow-ups:** Which values are truly inference-time? How are unavailable signals represented? What proves enabled diagnostics do not alter answers?

## D8 — Freeze a future calibrator protocol without claiming a trained model

- **When:** Milestone 5B working-tree protocol, `docs/MILESTONE5B_SHADOW_PROTOCOL.md`.
- **Problem/evidence:** Only the QASPER validation artifact is documented as locally available; official train/test artifacts, hashes, paper IDs, and disjointness are unverified.
- **Decision/status:** Specify paper-level train/validation/test roles, grouped CV, metrics, preprocessing, and stopping rules before outcomes; do not fit or claim a calibrator in the instrumentation checkpoint. The protocol explicitly says fitting and final refitting are not implemented.
- **Alternatives:** If official test artifacts cannot be verified, nested grouped CV may be reported only as exploratory; validation cannot be relabeled as confirmatory test data.
- **Trade-off/verification:** A frozen protocol reduces outcome-driven choices but provides no model-performance evidence until required data and implementation exist. The D-anchor selection helper has deterministic tests; the future model workflow does not yet exist.
- **Interview framing:** “The experiment design is frozen before fitting, but the claim boundary remains explicit: instrumentation and protocol code are not a fitted calibrator or a confirmatory result.”
- **Likely follow-ups:** Why require official splits? What is fitted inside each CV fold? What would make the final test confirmatory?

## D9 — Replace the joint-softmax confidence ratio with a span-versus-null logit margin

- **When:** Initial implementation; documented retrospectively in `PROJECT_STATE.md` and `paper/PAPER.md`.
- **Problem/evidence:** The first confidence formulation compared softmax probabilities across hundreds of candidate positions. The probability mass was spread so thin that spans appeared improbable, producing roughly 76% over-abstention on answerable questions and an unusable score distribution.
- **Decision/status:** Compare the selected start/end logit sum with the SQuAD2 null logit sum from that window, then apply a sigmoid: `sigmoid(best_span_score - selected_null_score)`. The current implementation is in `vistaar/reader.py`.
- **Alternatives:** The joint-softmax ratio is the only earlier formulation preserved in the project record. No controlled comparison with a learned calibrator was performed.
- **Trade-off/verification:** The margin is better scaled for thresholding and makes 0.5 the equal-logit point, but it is still a model score—not a probability-calibrated confidence estimate. Later held-out results remain weak.
- **Interview framing:** “The original normalization spread probability across hundreds of token positions and drove over-abstention. The replacement compares the best span directly with the model's null hypothesis, while retaining the explicit caveat that a sigmoid margin is not calibrated probability.”
- **Likely follow-ups:** Why compare with the null score from the selected window? Why does applying a sigmoid not prove calibration? What diagnostics would support a better confidence model?
