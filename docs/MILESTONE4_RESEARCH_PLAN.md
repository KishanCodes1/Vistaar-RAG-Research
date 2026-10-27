# Milestone 4 — Confidence and Abstention Research Plan

**Status:** diagnostic and predeclared planning milestone  
**Baseline:** verified Milestone 3 full-QASPER run  
**Decision:** ready for a controlled instrumentation/development experiment; not ready for a new abstention claim

## 1. Executive summary

Vistaar's current local abstention gate uses the reader's no-span `found` flag
plus one scalar signal: a sigmoid of the SQuAD2 best-span versus null logit
margin. Retrieval and reranker scores are recorded but do not affect the local
decision. On the 112-paper Milestone 3
held-out split, this score ranks answerability modestly (AUC 0.6636), but the
calibration-selected operating point answers only 110/383 answerable questions,
answers 3/20 unanswerable questions, and has no statistically resolved utility
gain over always answering.

The saved records do not support one causal explanation. They do establish four
actionable facts:

1. Reader confidence and answerability overlap heavily; some unanswerable cases
   receive confidence near one, while all six held-out exact-answer spans are
   rejected at the locked threshold.
2. Retrieval-derived signals contain additional but imperfect information. The
   final reranker score has held-out answerability AUC 0.6392 and only rank
   correlation 0.309 with reader confidence. This is post-hoc diagnostic evidence,
   not evidence that combining the signals will improve a fresh test result.
3. The present records omit important inference-time features: dense/BM25/RRF
   score distributions, the full pre-rerank pool, all 16 reranker scores, and raw
   per-window span/null statistics.
4. The reranker is not failing because of reversed sorting or corpus mismatch in
   the evaluator. It correctly sorts descending, but its top-four truncation
   displaces evidence-bearing RRF chunks often enough to lower recall.

The recommended next controlled experiment is therefore **instrumentation plus a
frozen, regularized answerability calibrator** using only inference-time features.
Development should use official QASPER train papers, threshold calibration should
use validation papers, and the official 416-paper test split must remain unopened
until all feature, model, and operating-point choices are frozen. The existing
validation split is exhausted for confirmatory testing.

## 2. Verified baseline

The report and machine-verification artifacts agree on the following baseline:

| Quantity | Verified value |
|---|---:|
| Papers | 281 |
| Evaluable questions | 1,003 (943 answerable / 60 unanswerable) |
| Evaluation records | 5,015 |
| Stored / unique chunks | 21,516 / 21,516 |
| Dense retrieval recall | 0.2283 |
| Hybrid retrieval recall | 0.2671 |
| Hybrid + reranker retrieval recall | 0.2392 |
| Held-out confidence AUC | 0.6636 [0.5286, 0.7912] |
| Locked threshold | 0.10, selected on calibration papers only |
| Held-out coverage | 0.2804 |
| Held-out balanced accuracy | 0.5686 [0.4773, 0.6441] |
| Answered unanswerable | 3 / 20 |
| Unnecessarily abstained answerable | 273 / 383 |

The document-bootstrap differences versus always-answer cross zero: balanced
accuracy −0.0214 to 0.1444, conditional-F1 −0.0092 to 0.0226, and covered-F1
−0.0081 to 0.0229. Milestone 3 therefore supports the hybrid-retrieval gain and
reranker-recall loss, but not a robust abstention utility improvement.

Integrity is not an open concern for this diagnosis: the final verifier reports
5,015/5,015 records, zero run errors, unique indices and retrieved chunks,
matching cache/config/source hashes, exact protocol-plan agreement, and 38
passing tests.

Repository state is `main` synchronized with `origin/main` at commit `5d2e042`.
The only pre-existing untracked path is `results/milestone2/`. One documentation
detail is stale: `PROJECT_STATE.md:7` says Milestones 2/3 are uncommitted, while
Git history contains the published Milestone 3 commit `28c9b3f`. This planning
task does not modify that existing report.

## 3. Evidence versus hypotheses

### 3.1 Verified from code and records

| Finding | Evidence |
|---|---|
| Local abstention ignores retrieval scores | `RAGPipeline._answer_local` gates only on `output.found` and `output.confidence` (`vistaar/pipeline.py:173-208`). `min_retrieval_score` is used only in the Gemini dense-hit branch (`pipeline.py:210-221`). |
| Confidence is not a calibrated probability | `LocalReader.answer` takes `sigmoid(best_span_score - null_score)` (`vistaar/reader.py:99-143`); no probability-calibration objective or reliability fit is applied. |
| Null comparison is local to one overflow window | The best span is selected globally, then compared with `null_by_window[window]` for that span's window (`reader.py:99-140`), not a separately selected global null candidate. |
| Candidate pruning can produce a zero-confidence no-span result | Start positions with softmax probability below 0.02 are skipped; no surviving span returns `found=False`, confidence 0 (`reader.py:111-124`). |
| Online and research thresholds differ | Runtime default is 0.5 (`vistaar/config.py`); Milestone 3 selected 0.10 offline. The selected threshold is not deployed automatically. |
| Gold evidence metrics cannot be inference features | Gold answers/evidence enter records and evaluation in `eval/run_eval.py:375-466`; they are unavailable for an unseen user query. |
| Final record fields are incomplete for feature research | E records preserve final four reranker hits and scores, confidence, raw answer and citations, but not dense/BM25 lists, RRF ranks 1–16, all reranker candidates, raw span/null logits, second-best spans or window counts (`run_eval.py:372-407`). |

### 3.2 Measured diagnostic evidence

On the held-out split:

- Answerable confidence median is 0.0150; unanswerable median is 0.0025. The
  distributions remain broad (standard deviations 0.269 and 0.255; maxima 0.995
  and 0.994).
- 190/403 questions have confidence below 0.01, including 176 answerable and 14
  unanswerable questions. At threshold 0.10, 273/383 answerable questions are
  rejected.
- Confidence has Spearman correlation only 0.196 with best-gold F1 among
  answerable held-out questions (0.239 on the full split).
- All six held-out answerable questions with best-gold F1 1.0 are below threshold
  0.10; the full split has 9/9 such cases below the same threshold.
- Final reranker top score has held-out answerability AUC 0.6392 (full: 0.6062),
  versus confidence AUC 0.6636 (full: 0.6227). Their held-out rank correlation is
  0.309 (full: 0.358), so they are related but not redundant.
- Top-1/top-2 reranker margin has held-out AUC 0.6102 but full-split AUC only
  0.5215. This instability argues against choosing it from the held-out result.
- Thirty-six of 112 test papers have zero covered answerable questions at
  threshold 0.10. Median per-paper answerable coverage is 0.25; the lower
  quartile is zero.

A heuristic question-prefix analysis is heterogeneous: held-out AUC is 0.819 for
"how" questions, 0.543 for "what", and 0.543 for yes/no-form questions; the
full-split yes/no-form AUC is 0.488. These groups contain only 5, 8 and 5 held-out
negatives respectively, so this is a hypothesis about question-type shift, not a
subgroup performance claim.

The three held-out false answers are consensus-unanswerable in the source
annotations. Their confidences are 0.9936, 0.6028 and 0.4296. One output quotes a
topical sentence without answering the requested distinction; another answers a
related reward question without establishing "at each step"; the third extracts
reported WER values from a paper whose annotators marked the question
unanswerable. This mixture warns against equating either lexical relevance or
reader margin with evidence sufficiency.

### 3.3 Hypotheses requiring experiments

The following are plausible, not established causes:

- **Domain-shifted reader confidence.** The reader is trained on SQuAD2, while
  QASPER contains scientific, list, yes/no and abstractive questions. Prior work
  finds softmax confidence unreliable under QA domain shift, but Vistaar has not
  isolated training-domain effects from retrieval and annotation effects.
- **Window-local null comparison.** Comparing the global best span with the null
  score from only that window may distort the margin. Logging global and
  per-window alternatives is required before changing the formula.
- **Question-type mismatch.** Extractive span selection may be especially weak
  for yes/no and list questions. Current subgroup negative counts are too small
  for a reliable conclusion.
- **Retrieval sufficiency adds information.** Retrieval scores separate labels
  somewhat, but no combination was predeclared and tested on untouched data.
- **Annotation noise contributes to apparent false answers.** Seventy-five full
  validation questions have mixed answerability annotations, yet all three
  held-out false answers above are consensus-unanswerable. Noise is therefore a
  limitation, not a complete explanation.

## 4. Confidence-signal diagnosis

### 4.1 Actual inference path

1. Dense search returns up to 24 chunks (`top_k * 3`).
2. Hybrid mode independently returns up to 24 positive-score BM25 chunks.
3. RRF deduplicates by deterministic chunk ID and keeps up to 16 (`top_k * 2`).
4. The cross-encoder, when enabled, sorts descending and keeps four.
5. The local reader concatenates those four chunks, creates 384-token overflow
   windows with stride 128, finds one global best span, and computes a sigmoid
   margin against the same window's null logits.
6. The local response abstains when no span is found or confidence is below the
   configured threshold. Retrieval scores and citation/evidence metrics do not
   participate.

### 4.2 Signal availability and leakage boundary

| Signal | Available online now? | Saved in E records? | Safe candidate feature? |
|---|---:|---:|---:|
| Reader confidence / raw answer / answer length | Yes | Yes | Yes |
| Final four reranker scores and margins | Yes | Yes | Yes, after train-only scaling |
| Dense top-24 scores and margins | Computed | No | Yes, requires logging |
| BM25 score/rank distribution | Computed | No | Yes, requires logging |
| RRF ranks, dense/BM25 agreement, candidate count | Computed | No | Yes, requires logging |
| All 16 cross-encoder scores | Computed | No | Yes, requires logging |
| Context diversity/redundancy and answer passage rank | Computable | Partly | Yes |
| Per-window span/null logits, second span, global null | Computed internally | No | Yes, requires reader instrumentation |
| Gold answer, answerability, evidence coverage, citation correctness | No | Evaluation only | **No — leakage** |

Retrieval scores need semantic care: in config E, `scores.retrieval` is the
maximum cross-encoder logit (`pipeline.py:173-204`), not a dense cosine or RRF
score. Cross-query magnitudes are not probabilities and require train-only
normalization or a model that can learn their scale.

### 4.3 Why the current threshold over-abstains

The selected score has some ranking value, but most answerable examples lie near
zero. Threshold 0.10 covers only 28.7% of held-out answerable questions. Lowering
the threshold increases coverage but also admits negatives; raising it quickly
removes correct spans. This is a ranking/representation problem, not merely a bad
choice between 0.10 and 0.50. The current balanced-accuracy objective also values
one of 20 unanswerable test errors as much as roughly 19 answerable errors; that
is appropriate for class balance but does not encode deployment utility.

## 5. Reranker diagnosis

### 5.1 Verified mechanics

- Configs C and D use the same 1,003 questions, paper namespaces, chunk corpus,
  dense/BM25 retrieval and vocabulary-coverage recall definition.
- C records eight RRF hits; the reader and Ret-R metric use the first four.
- D sends up to 16 RRF candidates to the cross-encoder and keeps four
  (`pipeline.py:132-149`, `index.py:342-353`). Thus reranking is intended to
  reorder **and then truncate**, not only annotate scores.
- All 1,003 D records contain four `source="rerank"` hits sorted in descending
  cross-encoder score. Score direction and final sorting are correct.
- RRF deduplicates by chunk ID before reranking (`index.py:295-307`). No duplicate
  top-four lists were found.

### 5.2 Measured displacement

| Diagnostic | Count/value |
|---|---:|
| C Ret-R@4 | 0.2671 |
| C Ret-R@8 | 0.2878 |
| D Ret-R@4 | 0.2392 |
| D lower / higher / equal than C@4 | 413 / 149 / 441 questions |
| C positive recall → D zero | 13 questions |
| C zero → D positive recall | 2 questions |
| Mean C-top4/D-top4 chunk overlap | 2.30 / 4 |
| No top-four overlap | 49 questions |
| D final slots outside C's recorded top eight | 961 slots across 598 questions |
| Questions where C@8 exceeds C@4 | 541 |
| Degraded questions dropping a C-top4 chunk with ≥0.1 individual gold-evidence coverage | 273 |

The verified proximate mechanism is therefore **evidence displacement at the
top-four cutoff**. Reranking promotes many lower-RRF candidates and removes
evidence-bearing C-top4 chunks more often than it adds useful evidence.

### 5.3 What is not yet verified

The complete 16-candidate pre-rerank pool and its dense/BM25/RRF ranks were not
saved, so the records cannot reveal the original rank of every promoted D chunk
or the cross-encoder scores of discarded candidates. Gold-overlap diagnostics
also measure lexical correspondence, not semantic relevance. Training-domain
mismatch is plausible—the cross-encoder is MS MARCO-based, and heterogeneous IR
benchmarks document variable zero-shot transfer—but Vistaar has not compared a
scientific-domain reranker or manually judged candidate relevance. Do not call
domain mismatch the proven root cause.

## 6. Candidate research approaches

### A. Non-parametric retrieval-sufficiency gate

**Hypothesis.** A predeclared two-signal gate using reader margin plus retrieval
sufficiency can reject flat, weak or internally inconsistent candidate sets that
the reader margin misses.

**Changes.** Log dense/BM25/RRF top-k scores and ranks, all 16 reranker scores,
top-score margins, dense/BM25 rank agreement, unique-section count and embedding
diversity. Evaluate individual signals and a small calibration-only rule grid;
do not choose rules from test.

**Data/compute.** Retrieval/reranker replay is required because current E records
lack pre-rerank features. Reader reruns are avoidable if records are joined by
fingerprinted question ID. CPU cost is low to moderate.

**Risks/failures.** Cross-encoder logits are query-scale dependent; BM25 scales
vary by paper; a two-dimensional threshold grid can still overfit 40 validation
negatives. Topical relevance may remain high on unanswerable questions.

**Baselines/metrics.** Existing reader confidence, retrieval top score, margin,
always-answer, default 0.5, AURC, answerability ROC/PR AUC, and matched-coverage
error counts. Complexity: **low–medium**.

### B. Reader confidence correction and explicit support verification

**Hypothesis.** The current same-window span/null margin is an incomplete proxy
for whether the retrieved set supports the answer. Global-null/second-span
statistics or a question–answer–evidence support verifier may better identify
plausible but unsupported spans.

**Changes.** First instrument each overflow window's null score, best and
second-best spans, global null, number of candidate starts, answer passage/rank
and span stability. Compare margin formulas without changing the reader. A later
variant can score whether the selected passages entail or sufficiently answer
the question. Mere answer-string containment is invalid here because the reader
is extractive and every nonempty answer already comes from context.

**Data/compute.** Reader instrumentation requires a fresh, fingerprinted run.
An NLI/support model adds one or more inference passes per answer and must have a
pinned revision. Human support labels may be required because QASPER evidence is
not a direct label for whether a predicted answer is entailed.

**Risks/failures.** Another generic verifier may repeat the current domain-shift
problem; using gold evidence as an input leaks; answerable does not imply the
predicted span is correct; added latency may dominate. Complexity: **medium–high**.

**Baselines/metrics.** Existing margin, global-null variants, support verifier
alone and combined, answerability AURC, covered F1, attribution judgments, latency
and verifier ablations.

### C. Learned leakage-safe answerability calibrator

**Hypothesis.** A regularized classifier over retrieval and reader diagnostics
can learn interactions that a scalar threshold misses. Kamath et al. report that
feature calibrators can improve selective QA under domain shift, but this is a
literature motivation, not a Vistaar result.

**Changes.** Train logistic regression as the primary model; permit one
predeclared tree baseline. Features may include only those available before the
answer decision: A/B features, question length/type, answer length, passage rank
and no-span indicator. Fit imputation, scaling, feature selection and model on
training papers only. Probability calibration, if attempted, uses calibration
papers only and is evaluated with Brier/ECE separately from selection utility.

**Data/compute.** The repository currently has only QASPER validation and CUAD.
Acquire and hash official QASPER train (888 papers, 2,593 questions) and test
(416 papers, 1,451 questions) splits. Negative counts must be audited before
power claims. Compute is modest after feature extraction.

**Risks/failures.** Few negatives, mixed annotations, document/topic shortcuts,
high feature-to-negative ratio, post-hoc feature fishing and cross-domain drift.
Class weighting and hyperparameters selected from final test are prohibited.
Complexity: **medium**.

**Baselines/metrics.** Reader confidence, approach A, retrieval-only model,
question-only control, all-answer/all-abstain, AURC, matched-coverage errors,
covered F1, ROC/PR AUC and document-cluster paired intervals.

### D. Explicit selective-risk or risk-control policy

**Hypothesis.** Optimizing a coverage-risk objective or calibrating a risk bound
is more aligned with abstention than maximizing balanced accuracy or conditional
F1 independently.

**Changes.** Rank by a frozen score from A–C. On calibration papers, either
select maximum coverage under a predeclared answered-unanswerable risk bound or
use a learn-then-test/conformal risk-control procedure. Report the complete
risk–coverage curve; do not market a guarantee under distribution shift unless
its assumptions and effective calibration sample are satisfied.

**Data/compute.** Little additional inference, but enough calibration negatives
are essential. With only 40 validation negatives in Milestone 3 calibration,
finite-sample upper bounds will be wide. Complexity: **medium**.

**Risks/failures.** Marginal guarantees can hide subgroup failures; covariate
shift invalidates naive guarantees; choosing the risk target after seeing test
is leakage; extremely low coverage can satisfy risk without useful service.

**Baselines/metrics.** Existing calibration-selected threshold, fixed-coverage
comparisons, AURC, risk-bound coverage, subgroup diagnostics and plain utility.

## 7. Predeclared experiment protocol

### 7.1 Research question and primary outcome

**Question.** Does a frozen leakage-safe answerability score using inference-time
retrieval and reader diagnostics improve selective answerability ranking over the
current reader margin on previously untouched QASPER papers?

**Primary metric.** Paired document-bootstrap difference in answerability AURC
(lower is better). At each coverage, risk is the fraction of covered questions
whose gold label is unanswerable. AURC evaluates ranking without selecting a test
threshold. The definition, coverage grid and tie handling must be fixed in code
and unit-tested before any final-test predictions are read.

### 7.2 Data partitions

1. **Development/training:** official QASPER train, 888 papers / 2,593 questions.
   Use document-grouped inner CV for feature and hyperparameter decisions.
2. **Calibration:** the already-inspected 281-paper validation split. It may set
   scaling, probability calibration and operating thresholds, but cannot provide
   a new confirmatory result.
3. **Final test:** official QASPER test, 416 papers / 1,451 questions. Download,
   hash and freeze it, but do not inspect labels, distributions or examples until
   the feature schema, model, hyperparameters, missing-data policy, baselines,
   metrics and go/no-go rules are committed in a protocol manifest.

The repository currently contains only validation; therefore no confirmatory
experiment may start yet. Verify that the available test artifact includes the
required annotations and audit answerability counts only through a sealed summary
step after the protocol is frozen. If the official test cannot be used, run
nested document-grouped CV as exploratory evidence and require a separate domain
(for example CUAD with its own frozen protocol) for confirmation.

### 7.3 Candidate methods and baselines

The first experiment should compare:

1. Always answer and always abstain.
2. Current reader confidence ranking and default threshold 0.5.
3. Retrieval top score and each predeclared retrieval feature individually.
4. A predeclared non-parametric reader-plus-retrieval gate (Approach A).
5. Regularized logistic calibrator (Approach C, primary candidate).
6. One tree-based calibrator as a secondary capacity check.
7. Question-only control to detect dataset/topic shortcuts.

Support verification (B) and risk control (D) follow only after the instrumentation
run establishes feature completeness and the simple calibrator baseline.

### 7.4 Threshold and test rules

- Choose thresholds on calibration only.
- Report one operating point matched to the baseline's calibration coverage
  (target 0.28, tolerance ±0.02) and one maximum-coverage point under a risk
  target fixed before calibration results are opened.
- Apply each locked threshold once to final test; no retuning, subgroup-specific
  thresholding or test-driven feature removal.
- Report test always-answer, always-abstain and default-0.5 outcomes even if they
  are unfavorable.

### 7.5 Metrics and uncertainty

- **Primary:** answerability AURC and paired candidate-minus-baseline difference.
- **Secondary abstention:** answerability ROC AUC; unanswerable-class PR AUC;
  balanced accuracy; coverage; answered-unanswerable rate; answerable abstention
  rate; conditional and covered F1; Brier/ECE only for probabilistic claims.
- **Retrieval (separate):** Ret-R@4/@8, candidate recall before reranking, top-k
  displacement and dense/BM25/RRF/reranker ablations.
- **Answer quality (separate):** best-gold F1/EM, first-gold selective F1,
  citation precision/recall and a small blinded human support sample if B is run.
- **Uncertainty:** 5,000 paired paper-cluster bootstrap resamples for metric
  differences; report point, interval and effective answerable/unanswerable paper
  counts. Subgroups are descriptive unless a minimum count is predeclared.
- **Sensitivity:** primary "answerable if any annotation" result plus exclude-mixed
  and all-annotators-answerable sensitivity without retuning.

### 7.6 Required records and provenance

Each row must retain current provenance plus:

- dense top-24 IDs/scores;
- positive BM25 IDs/scores;
- RRF top-16 IDs, fused scores and source ranks;
- all reranker candidate IDs/scores before top-four truncation;
- final context order and passage diversity features;
- per-window null score, best/second span scores, selected window, global-null
  alternative, candidate-start count and no-span reason;
- final feature vector, missing indicators, model output and threshold decision.

The manifest must fingerprint dataset splits, document/question IDs, feature
schema and code, exact model revisions, learned scaler/model artifacts, seeds,
partitions, thresholds and metrics. Cache reuse must validate every fingerprint.

### 7.7 Missing data, failures and sample limitations

- A no-span result is a valid categorical value, not missing data.
- Unexpected missing features receive a predeclared indicator and train-set
  imputation; no test-derived imputation. If >1% of final rows have unexpected
  missing features, the run fails integrity review.
- Failed rows are not dropped. Resume under the same fingerprint; if deterministic
  completion is impossible, report the run as failed.
- Report annotator disagreement and class counts before interpreting intervals.
- If final test has fewer than 40 consensus-unanswerable questions or the paired
  AURC interval is too wide to distinguish the minimum effect below, label the
  result inconclusive rather than pooling calibration and test.

## 8. Risks and limitations

- QASPER validation has already been inspected repeatedly and cannot serve as a
  fresh test for Milestone 4.
- Existing post-hoc signal AUCs and question-type segments were computed on
  Milestone 3 records; they can motivate features but cannot validate them.
- The current answerability target does not guarantee answer correctness. A
  method may improve answerability AURC while covered answers still have poor F1.
- Natural unanswerable prevalence is low, and one paper can contain correlated
  questions; question-level random splits or intervals are invalid.
- Mixed annotations make the label rule consequential. Test sensitivity cannot
  become a new optimization target.
- Generic rerankers and support verifiers may fail under scientific-domain shift.
- Added verification models increase latency and can create a second uncalibrated
  failure surface.
- QASPER-only success would not establish cross-domain safety or deployment
  readiness.

## 9. Explicit go/no-go criteria

### Go to final test

Proceed only if all are true before opening test labels:

1. Train/validation/test paper IDs are disjoint and their hashes are frozen.
2. Instrumentation reproduces the current A–E outputs when gating is disabled.
3. Every proposed feature is available at inference time and unexpected missing
   rate is ≤1% on train and validation.
4. The feature schema, logistic model, one secondary model, hyperparameters,
   metrics, thresholds and report template are frozen in a signed manifest.
5. Validation shows no pipeline-integrity regression and enough negative papers
   to estimate the predeclared effect; otherwise stop as underpowered.

### Claim success

The primary candidate passes only if:

1. Test AURC is at least 10% lower than current confidence **and** the upper bound
   of the paired document-bootstrap candidate-minus-baseline difference is below
   zero.
2. At matched coverage (0.28 ±0.02), answered-unanswerable rate is lower with a
   paired interval excluding zero.
3. Covered F1 is non-inferior: lower bound of candidate-minus-baseline difference
   is greater than −0.01 absolute.
4. Integrity, provenance and missing-data checks pass with zero dropped rows.
5. The direction remains consistent under the predeclared mixed-label sensitivity
   analyses; subgroup results need not be significant.

Any failure is **NO-GO for an improved-abstention claim**. It does not justify
post-hoc threshold changes or trying extra models on test. A technically valid
but underpowered result is **INCONCLUSIVE**, not success.

## 10. Recommended next controlled experiment

**Proceed with an instrumentation-only feature run, then a frozen logistic
calibrator experiment.** Specifically:

1. Add the missing pre-rerank and reader-window fields behind a versioned record
   schema; do not change retrieval order, reader outputs or online gating.
2. Reproduce a small integrity slice and then extract features on official QASPER
   train and validation.
3. Perform train-only grouped CV and one validation calibration pass comparing
   the current margin, retrieval-only signals, the predeclared two-signal gate and
   regularized logistic regression.
4. Freeze one primary candidate and one secondary baseline before acquiring or
   opening official test annotations.

Vistaar is **ready for this next controlled development experiment**, because
the baseline, provenance and failure modes are sufficiently characterized. It is
**not ready for confirmatory execution** until official train/test artifacts are
added, split hashes and negative counts are audited, and the full protocol is
frozen. Evidence justifying final-test execution is: deterministic feature
completeness, baseline reproduction, document-disjoint splits, validation-only
model selection, and adequate negative-paper sample size.

## 11. Literature grounding

These citations motivate candidate methods; they are not Vistaar measurements.

- Kamath, Jia, and Liang. 2020. [Selective Question Answering under Domain
  Shift](https://aclanthology.org/2020.acl-main.503/), ACL,
  DOI 10.18653/v1/2020.acl-main.503. Demonstrates softmax-confidence failure
  under QA domain shift and feature-based calibrators.
- Jiang et al. 2021. [How Can We Know When Language Models Know?](https://aclanthology.org/2021.tacl-1.57/),
  TACL 9, DOI 10.1162/tacl_a_00407. Separates probability calibration from raw
  confidence and studies post-hoc feature methods.
- Geifman and El-Yaniv. 2019. [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html),
  ICML/PMLR 97. Formalizes integrated selective prediction and risk–coverage.
- Angelopoulos et al. 2021/2022. [Learn then Test](https://arxiv.org/abs/2110.01052)
  and Angelopoulos & Bates, [A Gentle Introduction to Conformal
  Prediction](https://arxiv.org/abs/2107.07511). Motivate calibration-only risk
  control and its finite-sample/data-shift caveats.
- Nogueira and Cho. 2019. [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085).
  Establishes the MS MARCO passage-reranking setting underlying the current
  reranker family.
- Thakur et al. 2021. [BEIR](https://arxiv.org/abs/2104.08663), NeurIPS Datasets
  and Benchmarks. Documents heterogeneous zero-shot retrieval behavior; it
  supports testing domain transfer but does not explain Vistaar's observed loss.
- Bohnet et al. 2022. [Attributed Question Answering](https://arxiv.org/abs/2212.08037).
  Motivates evaluating answer support separately from answer text.
- Dasigi et al. 2021. [QASPER](https://aclanthology.org/2021.naacl-main.365),
  NAACL. Defines paper-disjoint train/validation/test splits (888/281/416 papers;
  2,593/1,005/1,451 questions).

## 12. Files inspected, commands and tests

### Repository files and artifacts

- `AGENTS.md`, `PROJECT_STATE.md`, `README.md`
- `docs/FULL_QASPER_EVALUATION.md`
- `results/milestone3/final_verification.json`
- `results/milestone3/full-qasper-validation-ae-v1/{records.jsonl,run_manifest.json,summary.json,sweep_E.json}`
- `results/milestone3/full-qasper-validation-ae-v1/threshold_protocol/protocol.json`
- `results/milestone3/full-qasper-validation-ae-v1/analysis/{analysis.json,verification.json,failure_cases.json,failure_summary.json,label_sensitivity.json,baseline_differences.json}`
- `vistaar/{config.py,index.py,pipeline.py,reader.py}`
- `eval/{run_eval.py,metrics.py,sweep.py,threshold_protocol.py,provenance.py,full_qasper_analysis.py}`
- all six files under `tests/`

`results/milestone2/` was deliberately not inspected or modified.

### Commands and analyses

- `git branch --show-current`, `git status --short --branch`, `git log --oneline -5`,
  `git remote -v`, `git fetch origin --prune`, and
  `git rev-list --left-right --count origin/main...main`
- `python -m pytest tests/ -q`
- Read-only Python analysis of Milestone 3 records, written only to the approved
  temporary directory as `m4_diagnosis.py` / `m4_diagnosis.json`; it recomputed
  confidence distributions, inference-only feature AUCs, question-prefix and
  paper segments, C/D top-k overlap, recall changes and evidence displacement.
- Direct inspection of the three consensus-unanswerable held-out false answers
  in the local QASPER validation parquet.
- Literature checks against ACL Anthology, PMLR, NeurIPS proceedings, arXiv and
  the Hugging Face QASPER dataset card.

No application code, tests, configuration, prior report or result artifact was
changed, and no new evaluation or API inference was run.
