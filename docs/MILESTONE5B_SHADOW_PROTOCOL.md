# Milestone 5B — Shadow Instrumentation Protocol

**Status:** development protocol; exploratory execution only  
**Workflow schema:** `vistaar-shadow-instrumentation-run-v1`  
**Diagnostics schema:** `vistaar-confidence-diagnostics-v1`  
**Frozen methods schema:** `vistaar-shadow-protocol-v1`  
**Baseline:** published Milestone 4B commit
`8c400eb010bf85258fc0bd72930a1bb1d20cbf83`

**Implementation boundary.** This checkpoint implements shadow diagnostic
collection, cache/schema safeguards, deterministic protocol calculations, and a
fingerprinted specification for future work. It does **not** implement the
calibrator feature-preprocessing, fitting, repeated-CV model-selection, or final-
refit workflow. The frozen estimator and preprocessing settings below are
protocol metadata, not evidence that training code ran or that a model exists.

## 1. Prerequisites and claim boundary

The repository currently contains only the official QASPER **validation**
parquet, `data/raw/qasper_validation.parquet`. Its published Milestone 3 hash is
`089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222`.
No local official train or test parquet is available, so their contents, hashes,
paper IDs, annotations, and disjointness have not been verified here.

The QASPER paper and dataset documentation define train/validation/test splits,
but documentation is not a substitute for local artifact verification. Until
official train and test artifacts are acquired, hashed, and checked, this work
is limited to instrumentation development and exploratory fixtures. The already
inspected validation split cannot support a new confirmatory claim.

## 2. Reproducible shadow workflow

The evaluator exposes an explicit local-only mode:

```bash
python -m eval.run_eval \
  --shadow \
  --dataset data/raw/qasper_validation.parquet \
  --dataset-partition validation \
  --papers 1 \
  --limit 2 \
  --only E \
  --workers 1
```

This command is an example only; it is not authorization for a larger run.
Shadow mode:

- refuses any reader backend other than `local`, before dataset loading;
- requests diagnostics explicitly for retrieval and answering;
- defaults to `results/cache/shadow-instrumentation/`, and refuses the ordinary
  `results/cache/eval/` root;
- defaults to a new timestamped directory under `results/shadow-runs/` and
  refuses to overwrite a nonempty output directory;
- includes the shadow workflow, diagnostics schema, dataset, selected IDs,
  document contents, code, model revisions, settings, configurations, package
  versions, execution settings, explicit partition role, and frozen protocol in
  the run fingerprint;
- rejects cache rows with absent diagnostics, a wrong schema, incomplete stage
  availability metadata, or a non-local decision backend.

Explicitly unavailable values remain `null`/absent according to the diagnostics
schema. They are never converted to zero. Legacy business fields keep their
existing record shape; `diagnostics` is additive and required only by the shadow
record contract.

Existing Milestone 3 caches cannot safely provide these features: their records
contain no diagnostics. Code/schema fingerprint changes already produce a new
run fingerprint, but shadow mode additionally uses a dedicated cache root and a
fail-closed row validator. Authoritative Milestone 3 artifacts and caches are
never overwritten.

`--dataset-partition` is required and accepts `train`, `validation`, `test`, or
`development-fixture`. Recognizable train/validation/test filenames must agree
with the supplied role. The manifest records dataset name `qasper` plus this
role; it never infers that arbitrary supplied data is validation.

## 3. Behavior-preservation gate

Before any multi-paper run:

1. Run deterministic diagnostics-disabled and diagnostics-enabled calls on the
   same local fixture.
2. Require exact equality of selected chunk IDs/order/scores, answer, raw answer,
   citations, confidence, abstention, and model identifier. Latency and the
   additive diagnostics object are excluded from equality.
3. Require every shadow record to pass the versioned diagnostic-record
   validator. Explicit unavailable reasons are valid; missing structure is not.
4. Require identical behavior with diagnostics disabled to the ordinary local
   inference path.
5. Stop on any mismatch. Do not drop or repair rows post hoc.

This gate is fixture-level evidence only. It does not establish full-QASPER or
full-distribution equivalence.

## 4. Intended paper-level partitions

After official artifacts are locally available and verified, the partitions
are:

| Role | Papers | Permitted use |
|---|---|---|
| Training/development | Official QASPER train | Fit preprocessing and the calibrator; paper-grouped inner CV for fixed hyperparameter selection. |
| Calibration | Official QASPER validation | Lock probability/decision calibration and operating thresholds once; no final claim. |
| Final evaluation | Official QASPER test | One evaluation after all code, features, hyperparameters, missing-data policy, and selection rules are frozen. |

The paper-ID sets must be pairwise disjoint. Each parquet SHA-256, ordered paper
and question IDs, annotation counts, loader implementation, and partition role
must be frozen in a protocol manifest before fitting. If the official test
annotations are unavailable or cannot be independently verified, there is no
confirmatory final evaluation; nested paper-grouped cross-validation may be
reported only as exploratory.

No question-level random split is allowed. The current validation fixture may
test mechanics but must not be relabeled as training or final evidence for the
confirmatory design.

### 4.1 Frozen grouped cross-validation

Calibrator hyperparameters use **5-fold, 20-repeat** grouped cross-validation on
official train papers only, with seed **20261028**. The group is QASPER paper ID;
all questions from a paper remain in one fold. Papers are stratified into those
with at least one unanswerable question and those without. For repeat `r`, each
stratum is sorted by SHA-256 of
`"{20261028 + r}:{stratum}:{paper_id}"`, then assigned round-robin to folds
0–4. This is invariant to input order.

The 5-fold/20-repeat design and seed family preserve the already documented
Milestone 3 grouped-CV convention; they are fixed for continuity rather than
chosen from new diagnostic outcomes.

The future calibrator workflow must fit preprocessing and the model independently
inside each training fold. It must select the lowest mean answerability AURC
across all 100 held-out folds; exact ties select the smaller `C`. Calibration and
official test are never used for hyperparameter selection. That fitting and
selection workflow is specified here but is not implemented in this checkpoint.

## 5. Experimental arms

### 5.1 Current margin baseline

The baseline is the unchanged local reader:

- score: sigmoid of selected best-span score minus the null score from the same
  overflow window;
- no-span outputs retain `found=False` and confidence zero;
- ranking baseline: saved confidence;
- operating thresholds: runtime default 0.5 and one threshold selected on
  calibration papers under the frozen selection rule.

Always-answer and always-abstain are mandatory reference policies.

### 5.2 Regularized calibrator arm

The primary learned arm is L2-regularized logistic regression. It is separate
from retrieval/reranker changes and uses only inference-time features:

- existing reader confidence and `found`;
- span/null margin, best-to-second span gap, selected/global null-range
  summaries, candidate counts, window count, answer passage rank, and answer
  length;
- predeclared dense, BM25, RRF, reranker, and selected-context score summaries;
- dense/BM25 overlap and candidate counts;
- question length and one fixed question-form encoding as a shortcut control.

The fixed question-form encoding is the normalized first-token bucket
`what/how/which/why/when/where/who`, a yes/no-auxiliary bucket for
`is/are/was/were/do/does/did/can/could/has/have/had/will/would`, and `other`.
It is frozen before fitting.

Counts use `log1p`; continuous features are standardized using training papers
only. Unexpected missing values receive a missing indicator and training-set
median imputation. Structurally unavailable fields are represented by their
availability/reason fields, not numeric zero.

Binary fields are encoded as 0/1. The fixed question-form categories and
versioned availability/reason categories use one-hot encoding with an explicit
`unknown` column. Each raw score family is standardized separately. A zero-
variance training feature receives scale 1. Every imputer, encoder, mean, and
scale is fit inside the grouped-CV training fold, then refit on all official
train papers after `C` is selected; no validation/test statistic enters
preprocessing.

Any future implementation is pinned to `scikit-learn==1.9.1`, using
`LogisticRegression(penalty="l2", solver="liblinear",
class_weight="balanced", fit_intercept=True, intercept_scaling=1.0,
max_iter=2000, tol=1e-8, random_state=20261028)`. The regularization grid is
`C in {0.01, 0.1, 1, 10}`. No feature may be added or removed after calibration
or final outcomes are inspected. A question-only model is reported as a control,
not as the primary candidate.

No preprocessor, logistic model, repeated-CV selector, or refitted artifact is
created by shadow mode. Implementing those components requires a separate review
before any labels or outcomes are used.

### 5.3 Recall-preserving reranker arm

This is a separate retrieval experiment and is not combined with the learned
calibrator for its primary comparison. It compares:

1. C: current hybrid/RRF order;
2. D: current cross-encoder top-four;
3. D-anchor: preserve the first two **unique** RRF candidates, then fill unique
   slots to a maximum of four by descending cross-encoder score.

D-anchor uses no gold evidence or answer labels. When two unique RRF anchors
exist, they always occupy output positions one and two in RRF order. Equal
cross-encoder scores are ordered by lower original RRF rank, then lexicographic
chunk ID. With fewer than two unique RRF candidates, preserve every available
anchor and fill from the reranker. With fewer than four unique candidates across
both inputs, return all unique candidates without padding. If the reranker pool
is short, remaining unique RRF candidates fill in RRF order. The primary
comparison is retrieval recall against D, not abstention utility. Any later
calibrator-plus-D-anchor combination is a new, separately declared experiment.

## 6. Metrics and uncertainty

### Calibrator comparison

- **Primary:** answerability AURC; lower is better. Scores rank higher as more
  answerable. At each target coverage in `{0.00, 0.01, ..., 1.00}`, select the
  score at rank `ceil(target * N)` and include the entire score-tie group at that
  cutoff. Risk is covered-unanswerable questions divided by covered questions;
  zero-coverage risk is defined as zero. Integrate risk over the **target**
  coverage grid by the trapezoid rule. No question-ID tie break is used.
- **Secondary:** ROC AUC, unanswerable-class PR AUC, balanced accuracy, coverage,
  answered-unanswerable rate, answerable abstention rate, conditional F1,
  covered F1, and complete risk–coverage curves.
- Brier score and ECE are reported only if probabilistic calibration is claimed.
- The matched-coverage operating point is selected on official validation from
  unique score thresholds plus all-abstain. Among candidates with coverage
  0.26–0.30, maximize balanced accuracy; ties choose coverage closest to 0.28,
  then fewer answered-unanswerable questions, then the higher threshold. If no
  candidate is in range, choose globally closest coverage with the same tie rule
  and report that fallback.
- The second operating point uses a policy-defined empirical risk ceiling of
  **0.05** (at most one unanswerable per 20 answered). On official validation,
  maximize coverage subject to risk <=0.05; ties choose higher balanced accuracy,
  then the higher threshold. All-abstain is the deterministic fallback. The 5%
  value is fixed as a policy ceiling, not selected from experiment outcomes and
  is not a statistical risk guarantee.

### Reranker comparison

- **Primary:** Ret-R@4 paired D-anchor minus D.
- **Secondary:** Ret-R@8, candidate recall before truncation, evidence
  displacement, best-over-gold raw-answer F1/EM, and citation metrics.

All differences use **5,000** paired paper-cluster bootstrap resamples with seed
**20261029**. For a partition with `N` papers, sample `N` paper IDs with
replacement and include every question from each sampled paper; both arms use
the identical sampled paper multiplicities. Intervals are the 2.5th and 97.5th
percentiles using NumPy's `linear` percentile method. Report the number of
papers/questions and answerable/unanswerable counts for every denominator.
Mixed-annotation sensitivity is reported without retuning. Subgroups are
descriptive unless a minimum sample size was frozen in advance.

Seed 20261029 is the next fixed seed in the published Milestone 3 protocol seed
family and is frozen before any shadow outcome is generated.

## 7. Selection and stopping rules

Instrumentation stops before model fitting if any of the following occurs:

- a diagnostics-enabled business output differs from its disabled counterpart;
- any row is dropped, duplicated, lacks provenance, or fails its cache/schema
  contract;
- unexpected feature missingness exceeds 1% on train or validation;
- dataset hashes or paper disjointness cannot be verified;
- the explicit dataset partition role conflicts with a recognizable filename;
- a hosted backend would be invoked.

The calibrator supports an improvement claim only if final-test AURC is at least
10% lower than the margin baseline, the paired candidate-minus-baseline AURC
interval is below zero, and covered F1 is non-inferior with a lower interval
bound greater than -0.01. Otherwise it is **NO-GO**; insufficient negative
papers or a wide interval is **INCONCLUSIVE**.

The reranker arm succeeds only if D-anchor improves Ret-R@4 over D with a paired
interval above zero and its best-over-gold F1 difference has a lower interval
bound greater than -0.01. Failure does not authorize changing the anchor count
after seeing final outcomes.

## 8. Leakage, missingness, and unsuccessful results

- Gold answers, answerability, gold evidence, F1, citation correctness, and
  retrieval recall are evaluation labels/metrics and never model features.
- Scalers, imputers, feature selection, class weights, and regularization are fit
  on training papers only. Calibration selects thresholds only. Final evaluation
  is read once.
- No final row is silently discarded. Deterministic failures are resumed under
  the same fingerprint; unresolved failures invalidate the run.
- Report every declared arm, missingness by feature and partition, cache
  rejections, malformed rows, class counts, intervals, and failed stopping rule.
- Negative or null results are retained as such. Do not pool partitions, add
  post-hoc models, retune thresholds, or combine calibrator and reranker arms to
  rescue an unsuccessful comparison.

## 9. Current authorization boundary

Authorized now: implementation, deterministic fixtures, manifest/cache tests,
and the offline test suite.

Not authorized: full-QASPER inference, calibrator fitting, final partition
selection, hosted API use, or confirmatory claims. A bounded local smoke run or
larger feature extraction requires separate review of its exact papers,
questions, configurations, output path, and expected CPU time.
