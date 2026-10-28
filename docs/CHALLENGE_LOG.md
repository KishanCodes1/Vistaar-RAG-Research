# Challenge Log

Entries summarize documented engineering or research failures. “Confirmed” refers to evidence in the repository; explanations called “interpretation” are not additional measurements. See [the original-code audit](AUDIT.md), [the abstention audit](ABSTENTION_AUDIT.md), [Milestone 2](MILESTONE2_REPORT.md), and [the full-QASPER report](FULL_QASPER_EVALUATION.md) for primary detail.

## C1 — Cross-user index exposure

- **When:** Original code, before the initial commit (`130398e`, 2026-10-27).
- **Symptom/root cause:** A module-global FAISS index mixed every session's documents in one retrieval collection. `docs/AUDIT.md` classifies this as a critical data-leak defect.
- **Resolution/evidence:** Separate namespace indexes and locking prevent accidental cross-session mixing; implementation is in `vistaar/index.py` and `vistaar/api.py`, with namespace-isolation tests.
- **Limit:** The caller supplies the namespace. There is no authentication or authorization, so namespaces are not secure tenant boundaries or durable tenant management.
- **Interview answer:** “The failure was shared mutable retrieval state. Separate namespace indexes and synchronized access prevent accidental session mixing; production tenancy still needs identity and authorization.”
- **Follow-ups:** How are namespaces generated, protected, and expired? What concurrent operations are covered?

## C2 — Hosted reader could not support the reported evaluation

- **When:** Initial development, recorded in `PROJECT_STATE.md` and `docs/AUDIT.md`.
- **Symptom/root cause:** Gemini requests hit free-tier quota; the project record says no hosted evaluation was possible. Gemini 3.x also returned empty output when its thinking budget consumed the token budget.
- **Resolution/evidence:** Use the local CPU extractive reader for reported metrics; set `thinkingBudget: 0` in the hosted path. See `vistaar/generate.py` and `vistaar/reader.py`.
- **Limit:** The hosted path was not evaluated in the published metric runs; local extractive outputs mismatch abstractive references.
- **Interview answer:** “The quota and hosted-service constraints made Gemini unsuitable for the controlled reported run, so the evaluation uses the local reader and treats hosted behavior as a separate, unverified path.”
- **Follow-ups:** How would you compare readers without changing the retrieval corpus or evaluation protocol?

## C3 — MPS inference aborted

- **When:** Initial development, recorded in `PROJECT_STATE.md`.
- **Symptom/root cause:** SentenceTransformer/CrossEncoder on MPS aborted with a Metal command-buffer assertion in the recorded environment.
- **Resolution/evidence:** Pin those paths to CPU; the project states CPU inference was used for reported metrics.
- **Limit:** This is an environment-specific observation, not evidence that MPS fails on every device or version.
- **Interview answer:** “The recorded backend abort led the project to use CPU, trading throughput for a stable evaluation path in that environment.”
- **Follow-ups:** What benchmark would justify revisiting device choice?

## C4 — Abstention results were inflated by evaluation defects

- **When:** Discovered by the independent audit dated 2026-10-28; addressed in Milestone 2.
- **Symptom/root cause:** The audit confirmed duplicate chunks from persistent append-only indexing, incomplete cache identity, stale natural sweep values based on post-abstention predictions, and post-hoc threshold selection. A separate bug treated a valid empty `raw_prediction` as if the field were missing, falling back to the abstention message and making otherwise identical dense configurations diverge. Clean re-indexing moved natural AUC from 0.6959 to 0.6459.
- **Resolution/evidence:** Run-scoped indexes/caches, strict duplicate checks, full fingerprints, raw-span sweep behavior, explicit preservation of empty raw spans, and document-disjoint calibration/test; see `docs/MILESTONE2_REPORT.md`, `eval/provenance.py`, `eval/run_eval.py`, and tests. The key lesson is that field absence and a valid empty string have different semantics.
- **Limit:** The **Milestone 2 negative-enriched 20-paper holdout** remains weak (balanced accuracy 0.5179 [0.4289, 0.5938]). The later Milestone 3 full-QASPER holdout is 112 papers / 403 questions, with balanced accuracy 0.5686 [0.4773, 0.6441], coverage 0.2804, 3/20 unanswerable questions answered, and 273/383 answerable questions declined. Neither supports a robust abstention claim. The original cache state for every historical prediction cannot be reconstructed exactly.
- **Interview answer:** “The surprising score was partly a state-management and metric-generation problem. Clean run identity, strict cache reuse, explicit empty-span handling, and locked paper-level threshold selection replaced the contaminated workflow.”
- **Follow-ups:** Why did duplicate chunks affect top-k retrieval? Which historical facts remain unrecoverable? What is the corrected claim?

## C5 — Scientific evidence recall improved with hybrid retrieval, but reranking hurt

- **When:** First measured in the 40-paper evaluation; confirmed at full QASPER scale in Milestone 3.
- **Evidence:** The final report gives full-run Ret-R 0.2283 for dense, 0.2671 for hybrid, and 0.2392 after reranking. Paired paper-cluster differences are +0.0388 [0.0332, 0.0448] for dense-to-hybrid and −0.0279 [−0.0349, −0.0212] for hybrid-to-rerank. The reranker answer-F1 difference is not significant.
- **Confirmed mechanism and hypothesis:** The verified proximate mechanism is **top-four evidence displacement**: reranking promotes lower-RRF candidates and removes evidence-bearing C-top4 chunks more often than it adds useful evidence. MS MARCO/scientific-domain mismatch is a plausible explanation, but it has not been isolated with a scientific-domain reranker or manual relevance judgments and is not a proven root cause.
- **Resolution/status:** Preserve the negative result and do not assume a general-domain reranker transfers. D-anchor is a future recall-preserving experiment, not a demonstrated fix.
- **Interview answer:** “The reranker made a plausible ranking change but reduced the target metric—evidence recall. The full-scale result made that a useful negative finding.”
- **Follow-ups:** What exactly is Ret-R? Why can answer F1 and evidence recall move differently? What would validate D-anchor?

## C6 — Extractive reader and QASPER gold answers disagree in form

- **When:** Identified during initial evaluation and retained in later reports.
- **Symptom/root cause:** QASPER references can be abstractive while the local reader selects a source span; exact match/token F1 can penalize a semantically correct but shorter span.
- **Resolution/evidence:** Report retrieval, citation, answerability, and raw-span metrics alongside F1; document the limitation in `README.md` and `paper/PAPER.md`.
- **Limit:** No human evaluation or semantic answer judge is reported; this does not establish that low F1 is only a metric artifact.
- **Interview answer:** “Extraction quality, retrieval, and abstention are reported separately, but the metric mismatch does not excuse low F1; human or validated semantic evaluation remains missing.”
- **Follow-ups:** What complementary metric would you add, and how would you validate it?

## C7 — QASPER text spacing defeats exact evidence matching

- **When:** Initial research, recorded in `PROJECT_STATE.md`.
- **Symptom/root cause:** Sampled `full_text` contains character-spaced runs; exact substring matching found 0/435 evidence spans, while de-spacing joined words and was lossy.
- **Resolution/evidence:** Use token-vocabulary overlap for retrieval/citation metrics; document it as a limitation.
- **Limit:** Vocabulary overlap can miss paraphrases and can reward lexical overlap without semantic entailment.
- **Interview answer:** “The project rejected lossy de-spacing and uses vocabulary coverage, while explicitly limiting the claim to lexical overlap rather than semantic entailment.”
- **Follow-ups:** How would you validate semantic evidence alignment without leaking labels into inference?

## C8 — Instrumentation must preserve inference behavior

- **When:** Milestone 4B diagnostics and current Milestone 5B work.
- **Symptom/risk:** Collecting candidate traces and reader internals could accidentally change ordinary outputs or create misleading missing-value defaults.
- **Resolution/evidence:** Opt-in diagnostics, versioned schemas, explicit availability reasons, additive records, fail-closed cache validators, and fixture-level equality tests. See `docs/CONFIDENCE_DIAGNOSTICS.md`, `eval/shadow.py`, and the relevant diagnostics/cache tests.
- **Limit:** Fixture tests establish the tested path only. No full shadow experiment or end-to-end performance result is documented in the current task.
- **Interview answer:** “Observability is a versioned side channel with fixture-level behavior checks; that evidence must not be generalized to the full data distribution without an approved run.”
- **Follow-ups:** Why is unavailable different from zero? What fields are not inference-time? What is the next validation gate?

## C9 — The first confidence formula drove over-abstention

- **When:** Initial development, documented in `PROJECT_STATE.md` and `paper/PAPER.md`.
- **Symptom/root cause:** A joint-softmax ratio spread probability mass across hundreds of start/end candidate positions, making plausible spans look improbable. The recorded symptom was roughly 76% over-abstention on answerable questions and an unusable confidence distribution.
- **Resolution/evidence:** Replace the ratio with a direct best-span versus SQuAD2 null-logit margin and apply a sigmoid. `vistaar/reader.py` computes `sigmoid(best_span_score - selected_null_score)`; project records document the before/after rationale.
- **Limit:** Better scaling does not establish probability calibration. Milestone 2 and Milestone 3 held-out results remain weak, and no learned calibrator has been implemented or fitted.
- **Interview answer:** “The original normalization compared one span with probability mass spread over hundreds of positions. A direct span-versus-null logit margin fixed that scaling failure, but the resulting sigmoid remains a threshold score, not a calibrated probability.”
- **Follow-ups:** Why does 0.5 mean equal span/null logits? Why is that not calibration? Which train-only diagnostics could support a future calibrator?

## Unresolved failures and evidence gaps

- OCR for scanned PDFs, real authentication/tenant management, proxy-aware rate limiting, and streaming remain open in `docs/AUDIT.md`.
- No human evaluation is documented; single-domain English QASPER limits external validity.
- Official QASPER train/test artifacts and split disjointness are unverified locally according to `docs/MILESTONE5B_SHADOW_PROTOCOL.md`.
- The cache/index defect's exact effect on every historical output cannot be recovered from the old cache provenance, according to `docs/ABSTENTION_AUDIT.md`.
