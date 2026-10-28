# Project Milestones

This is a navigation and status record, not a replacement for the detailed reports. Status labels distinguish committed history from current working-tree work. See [decision log](DECISION_LOG.md), [challenge log](CHALLENGE_LOG.md), [trade-offs](TRADEOFFS.md), and [interview guide](INTERVIEW_GUIDE.md).

## Timeline

| Milestone | Date/status | Evidence and checkpoint | Main result or unresolved point |
|---|---|---|---|
| Initial Vistaar implementation | 2026-10-27; committed | `130398e` — initial RAG pipeline, evaluation harness, and project state. | Added namespaced document QA, provenance, retrieval variants, reader, tests, and initial QASPER records. Original-code defects and limitations are catalogued in `docs/AUDIT.md`. |
| Milestone 1 — reconcile paper and abstention claims | 2026-10-27; committed | `e87fa0c`. | Replaced “calibrated” language with thresholded abstention and reported negative-enriched results. Later independent audit found historical index/cache and sweep defects; preserve this as history, not current evidence. |
| Milestone 2 — evaluation integrity | 2026-10-28; documented in `docs/MILESTONE2_REPORT.md`, published alongside Milestone 3 | Main published checkpoint `28c9b3f`. | Run-scoped indexes/caches, strict duplicate checks, input fingerprints, document-disjoint threshold selection. Report records 35 tests passing at that checkpoint. The negative-enriched 20-paper holdout reaches balanced accuracy 0.5179 [0.4289, 0.5938], compatible with chance. |
| Milestone 3 — full QASPER | 2026-10-28; committed | `28c9b3f`; report `docs/FULL_QASPER_EVALUATION.md`. | 281 papers, 1,003 questions, 5,015 A–E records; independently verified. Hybrid retrieval improves evidence recall; reranking reduces it. The 112-paper / 403-question holdout reaches balanced accuracy 0.5686 [0.4773, 0.6441] at 0.2804 coverage, answers 3/20 unanswerable questions, and declines 273/383 answerable questions; the gain over always-answer is unresolved. Project state records 38 tests passing after this milestone. |
| Milestone 4 — research plan | 2026-10-28; committed | `4f01f71`; `docs/MILESTONE4_RESEARCH_PLAN.md`. | Plans confidence/abstention research. Existing validation analyses are hypothesis-generating; they cannot serve as independent confirmation. |
| Milestone 4B — confidence diagnostics | 2026-10-28; committed | `8c400eb`; `docs/CONFIDENCE_DIAGNOSTICS.md`. | Adds opt-in inference-time diagnostic schema and tests. It preserves ordinary behavior in tested fixtures; no new calibrated abstention result follows from instrumentation alone. |
| Milestone 5 — diagnostic failure attribution | Completed offline analysis; working tree, uncommitted | `docs/MILESTONE5_DIAGNOSTIC_ANALYSIS.md`. | Post hoc exploratory analysis of existing Milestone 3 records; **no inference run or production change**. On held-out answerable questions, complete retrieval failure is 13/383, the reader-failure proxy is 268/383 (235/383 with Ret-R@4 ≥ 0.1), reranker degradation is 161/383, and the locked threshold declines 273/383. Categories overlap and are operational proxies, not causal assignments. The analysis motivates shadow instrumentation rather than a production change. |
| Milestone 5B — shadow protocol and instrumentation | 2026-10-28; working tree, uncommitted | `docs/MILESTONE5B_SHADOW_PROTOCOL.md`, `eval/shadow.py`, `eval/shadow_protocol.py`, `eval/run_eval.py`, and related tests are present as local changes. | Defines local-only diagnostics, cache/provenance guards, deterministic protocol calculations, and D-anchor selection. The calibrator preprocessing/fitting/CV/refit pipeline is not implemented; no experiment or model fit is claimed. Official local train/test artifacts and split verification remain prerequisites. |

## Checkpoint and test history

The Git history contains six commits, from the initial import on 2026-10-27 through the Milestone 4B checkpoint on 2026-10-28. The repository records 12 tests at the independent Milestone 1 audit, 35 at Milestone 2, and 38 after Milestone 3. The current uncommitted Milestone 5B changes add focused diagnostics, cache, shadow, and protocol tests, but this documentation audit did not execute tests or experiments.

Commit chronology and file summaries can be checked with `git log`; the precise published report remains authoritative for each result. No commit, push, or staging action is part of this record.

## Open work

- Verify official QASPER train/test files, hashes, paper-ID disjointness, and annotation counts before any confirmatory use.
- Implement the frozen calibrator workflow only after a separate review; fit preprocessing and models inside training folds, select using the frozen rule, and keep the final test untouched.
- Evaluate D-anchor against D on the predeclared data and metrics; it is not yet a measured result.
- Address OCR, authentication/tenant management, proxy-aware rate limiting, and other production issues listed in `docs/AUDIT.md`.
- Add a second domain or retain the explicit single-domain scope. CUAD is recorded as downloaded but unused in `PROJECT_STATE.md`.
- Add human or validated semantic evidence/answer evaluation; current lexical-overlap metrics have known limits.
- Recover no more historical detail than existing Git commits, reports, and artifacts support. See the gaps in [the interview guide](INTERVIEW_GUIDE.md).
