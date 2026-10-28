# Milestone 5D — Synthetic-Only Leakage-Safe Calibrator Mechanics

**Verdict:** `IMPLEMENTATION CHECKPOINT ONLY`

**Scope:** deterministic synthetic fixtures and local scikit-learn mechanics

**Repair status:** Milestone 5D-R and 5D-R1 adversarial leakage,
summary-invariant, and artifact-integrity repairs applied; official execution
remains blocked

**Not supported:** official-data fitting, test scoring, retrieval/reader
inference, research evaluation, D-anchor evaluation, or scientific claims

## 1. Outcome and safety boundary

Milestone 5D implements an isolated mechanics package for strict feature
allowlisting, fold-local preprocessing, repeated paper-grouped cross-validation,
synthetic refitting, partition-aware threshold selection, and JSON-only fitted
artifact validation.

The package has no CLI, dataset loader, model loader, cache default, environment
configuration, QASPER path, or held-out-test scoring entry point. Public fitting
functions require every observation to declare `origin="synthetic"` and the
`train` role. Threshold selection requires the synthetic `validation` role. The
only accepted roles are `train` and `validation`; `test` intentionally does not
exist in the mechanics role schema.

No official-data model was fitted. No retrieval/reader model was loaded, and no
retrieval/reader inference or research evaluation was run. Focused tests fit and
score only tiny synthetic logistic-regression fixtures. No scientific-
effectiveness claim is supported.

## 2. Modules and responsibilities

### `eval/calibrator_mechanics.py`

The module provides:

- versioned observation, feature, label, partition, preprocessor, estimator, CV,
  artifact, and threshold schemas;
- exact-key validation and an explicit ordered feature allowlist;
- deterministic extraction into an identity-free feature matrix;
- a private leakage-safe, scikit-learn-compatible transformer;
- a single `Pipeline` containing preprocessing and frozen logistic regression;
- validation around the existing frozen `grouped_cv_assignments` and
  `answerability_aurc` helpers;
- 5-fold, 20-repeat synthetic grouped-CV orchestration and smaller-`C` tie
  handling;
- train-role-only final synthetic refitting;
- deterministic configuration, data, fold, fitted-state, artifact, score, and
  threshold fingerprints;
- JSON-only save/load with exact compatibility and fitted-state checks; and
- validation-role-only wrappers around the existing frozen threshold arithmetic.

The module deliberately does not import `eval.qasper`, `eval.run_eval`, retrieval
or reader code, settings, environment variables, or application caches.

### `tests/test_calibrator_mechanics.py`

The tests create all observations as deterministic in-memory dictionaries. Their
only data artifacts are JSON files under pytest's temporary directory. No
repository dataset or result path is used.

## 3. Versioned contracts

| Contract | Version |
|---|---|
| Observation | `vistaar-synthetic-calibrator-observation-v1` |
| Features | `vistaar-calibrator-features-v1` |
| Labels | `vistaar-answerability-label-v1` |
| Partition | `vistaar-synthetic-partition-v1` |
| Preprocessor | `vistaar-calibrator-preprocessor-v1` |
| Estimator | `vistaar-calibrator-estimator-v1` |
| Grouped CV | `vistaar-synthetic-grouped-cv-v1` |
| Fitted artifact | `vistaar-synthetic-calibrator-artifact-v2` |
| Threshold selection | `vistaar-synthetic-threshold-selection-v1` |

### 3.1 Observation identity

An observation must contain exactly:

- its observation schema;
- `origin="synthetic"`;
- canonical `paper_id` and globally unique canonical `question_id`;
- a versioned partition object;
- a versioned feature object; and
- a separate versioned label object.

Unknown top-level keys, missing keys, incompatible schema versions, duplicate
question IDs, malformed identities, invalid roles, and non-synthetic origins fail
closed. Canonical identities are nonempty Unicode NFC strings without leading or
trailing whitespace. Case is preserved and never implicitly folded. Input order
is not trusted; validated rows are sorted by paper and question identity before
fitting and fingerprinting.

### 3.2 Label semantics

The only label is `answerable`, and it must be a Python Boolean. Integers,
strings, truthy values, and other implicit coercions are rejected. In this
checkpoint it means only “the deterministic synthetic fixture marks this row as
answerable.” It does not certify compatibility with QASPER annotation semantics.

Labels remain outside the feature matrix. They are used for synthetic estimator
fitting and for the CV fold AURC required by the frozen protocol.

### 3.3 Partition roles

| Operation | Required role |
|---|---|
| Grouped CV and hyperparameter selection | `train` |
| Final synthetic refit | `train` |
| Synthetic score generation | Explicit caller-supplied `train` or `validation` |
| Threshold selection | `validation` |
| Test/confirmatory scoring | Unavailable |

Train/validation pair validation rejects overlap in either paper IDs or question
IDs. Grouped-CV validation asserts that every paper receives exactly one
validation fold per repeat and that no paper appears on both sides of a fold.
Missing, unknown, contradictory, or operation-inconsistent roles fail closed.

## 4. Strict feature allowlist

Feature values must contain every allowed key and no other key. Outcome-bearing
names such as `answerable`, labels, IDs, partitions, question/answer text, gold
answers/evidence, citations, retrieval recall, F1, and exact match are explicitly
prohibited and also fail the exact allowlist.

This is syntactic isolation, not proof of semantic provenance. The validator
cannot prove that an allowed value was genuinely generated at inference time;
official use remains blocked on a reviewed diagnostics-to-feature mapper.

### 4.1 Count features

Counts are nonnegative integers or explicit `null`:

- question and answer token lengths;
- reader window, candidate-start, and candidate-span counts;
- answer passage rank;
- dense, BM25, RRF, reranker, and selected-context candidate counts; and
- dense/BM25 overlap count.

### 4.2 Continuous features

Continuous features are finite real numbers or explicit `null`:

- reader confidence;
- span/null margin;
- best-to-second span gap;
- selected null score plus global null-score minimum, maximum, and range; and
- `top_score`, `top1_top2_margin`, `mean`, `std`, `min`, `max`, and `range` for
  dense, BM25, RRF, reranker, and selected-context score families.

### 4.3 Binary and categorical features

- `reader_found` must be Boolean.
- `question_form` uses the frozen first-token buckets, one
  `yes_no_auxiliary` bucket, `other`, and explicit `unknown`.
- Dense, BM25, RRF, reranker, selected-context, and reader statuses use a
  versioned fixed vocabulary containing explicit availability/reason states and
  `unknown`.

An unrecognized category is not silently mapped to `unknown`; producers must
explicitly emit `unknown`. Feature ordering is the declared tuple
`RAW_FEATURE_ORDER`, and both raw and transformed order are fingerprinted.

The mechanics validator also enforces source-schema and derived algebraic
relationships: unavailable stages require null candidate/score numerics;
empty stages require zero candidates and null summaries; available nonempty
stages require finite summaries satisfying `min <= mean <= max`,
`min <= top_score <= max`, `0 <= top1_top2_margin <= range`, and an implied
second-best score `top_score - top1_top2_margin` inside `[min, max]`;
dense/BM25 overlap cannot exceed either candidate count; unavailable readers
cannot expose local-reader diagnostics; and complete null-score ranges must be
internally consistent. The flattened contract establishes top-1/top-2 relative
ordering but not full candidate ordering, so the validator requires a nonnegative
top margin without equating `top_score` with `max`. No upper bound is imposed on
`std`, because the flattened contract does not restate the source producer's
variance convention.
`externally_supplied_hits` remains unresolved because the frozen production
mapping does not define which upstream values are semantically available.

## 5. Fold-local preprocessing

An internal pipeline factory returns one scikit-learn `Pipeline` containing the
preprocessor and `LogisticRegression`. Underscoring and `__all__` exclusion mark
it as unsupported internal API; they are Python conventions, not access control,
and direct callers can still invoke it. The supported fitting paths instead
consume a validated `ValidatedBatch` snapshot plus explicit row masks. The
internal preprocessor defensively revalidates every direct matrix row for shape,
types, categories, and source-schema relationships, but it cannot establish
observation origin, partition role, paper grouping, or inference-time provenance.
Every CV fold constructs and fits a fresh pipeline using only that fold's
training rows.

The implemented preprocessing policy is:

1. Validate the complete raw schema before creating the matrix.
2. For counts, learn the raw median on the training fold, impute from that median,
   then apply `log1p`.
3. For continuous values, learn the training-fold median, impute, then learn and
   apply per-column standardization.
4. Append a fixed missingness-indicator column for every count and continuous
   feature. The output shape therefore does not depend on which fold happened to
   contain missing values.
5. Keep binary values as 0/1.
6. One-hot encode categorical features against fixed, versioned vocabularies.
7. Use scale 1 for a zero-variance training feature.
8. Reject an entirely missing training-fold numeric feature because no
   training-fold median exists and the frozen protocol defines no alternative.

Transforming validation rows never mutates learned medians, means, scales,
vocabularies, or output order. Extreme validation sentinels and categories absent
from training are covered by adversarial tests.

Validation produces a defensive snapshot whose NumPy arrays are read-only.
Grouped CV, final refit, prediction, training digest construction, and threshold
selection reuse one validated snapshot rather than reopening mutable caller data.

The estimator preserves the frozen settings:

- L2 penalty;
- `liblinear` solver;
- `class_weight="balanced"`;
- intercept enabled with scaling 1;
- `max_iter=2000`;
- `tol=1e-8`;
- `random_state=20261028`; and
- `C` in `{0.01, 0.1, 1, 10}`.

Dependencies are checked against `numpy==2.5.4` and
`scikit-learn==1.9.1` before fitting or loading an artifact.

## 6. Grouped cross-validation and refit boundaries

The coordinator reuses `eval.shadow_protocol.grouped_cv_assignments` without
changing its algorithm:

- five folds and twenty repeats;
- seed 20261028;
- paper-level grouping;
- paper stratification by presence of any unanswerable question;
- SHA-256 order inside each stratum; and
- round-robin fold assignment.

Before fitting, every repeat must contain all five folds, every training and
validation fold must be nonempty, and both sides must contain both answerability
classes. This is a fail-closed validation around the frozen algorithm, not a
replacement assignment rule.

Each of the four `C` values is evaluated on all 100 synthetic validation folds.
The selected value minimizes mean held-out-fold answerability AURC; an exact tie
selects the smaller `C`. Fold models are discarded. The selected pipeline is then
fit once on all observations in the explicit synthetic `train` role.

CV fold labels necessarily participate in AURC-based hyperparameter selection,
as required by the frozen protocol. “Held-out labels cannot influence selection”
is therefore interpreted as external validation and final-test labels: those are
not accepted by the CV/refit API. Permuting labels in a separate synthetic
validation partition cannot alter an already fitted model or selected `C`.

## 7. Fingerprints and artifact loading

The implementation reuses canonical JSON plus SHA-256 from
`eval.provenance.stable_fingerprint`.

The configuration fingerprint covers:

- every schema version;
- ordered feature allowlist and category vocabularies;
- label type and semantics;
- missingness, imputation, count transform, scaling, and zero-variance policy;
- estimator class, settings, and complete `C` grid;
- fold count, repeats, seed, assignment algorithm, metric, and tie rule; and
- NumPy, scikit-learn, and SciPy versions;
- Python implementation/version and platform system/machine; and
- SHA-256 identities for the calibrator module, frozen protocol helper, and
  canonical fingerprint helper.

The fitted artifact additionally covers:

- deterministic training identity/content digest, including synthetic labels and
  allowlisted values but no free text;
- declared training rows and papers, with the paper count cross-checked against
  assignment identities;
- all paper-to-validation-fold assignments and their fingerprint;
- mean AURC for every candidate `C`;
- selected `C`;
- ordered transformed feature names;
- learned medians, means, scales, logistic coefficients/intercept/classes, and
  iteration count; and
- a fingerprint of the complete fitted state and complete manifest.

Artifacts use strict JSON, not pickle or joblib. Parsing rejects duplicate object
keys and nonstandard `NaN`/`Infinity` constants. Loading independently recomputes
the fingerprint of the complete loaded configuration and requires the declared,
current, and optional caller-expected fingerprints to agree. It also rejects
unknown/missing keys, schema/config/dependency/implementation mismatches, invalid
digests, hash-inconsistent or malformed fold assignments, non-finite or wrongly
shaped fitted arrays, invalid classes or scales, feature-order changes,
fitted-state changes, selected-`C`/CV score disagreement, and disagreement between
serialized state and the reconstructed in-memory pipeline. Loading also requires
`training_papers` to equal the number of unique paper identities in the validated
fold assignments and `training_rows >= training_papers`. It cannot independently
reconstruct the exact row count because the artifact intentionally stores no
question identities or raw observations. Saving refuses to overwrite a file.

A matching fingerprint establishes canonical-content integrity and configuration
compatibility under this serializer. It is not a byte-level file hash: harmless
whitespace/key-order changes do not change canonical content. Because hashes are
unkeyed, an attacker can recompute every digest after changing content; these
fingerprints are not signatures or proof that data are authentic, official,
correctly split, scientifically valid, or free from every leakage path.

## 8. Threshold boundary

`select_synthetic_threshold` preserves the existing matched-coverage and 5%
empirical-risk calculations and tie rules. It adds an input contract:

- every observation must be synthetic and in the `validation` role;
- a complete compatible synthetic artifact must pass load/state validation;
- scores are computed internally and keyed by exact `question_id`, preventing a
  caller-supplied score/artifact or score/order mismatch; and
- the result fingerprints the method, validation-data digest, ordered score
  digest, parent artifact, and selected values.

No test-role threshold selection or held-out-test evaluator is implemented.

## 9. Leakage tests and evidence boundary

The focused suite contains 84 tests covering:

- rejection of labels, identities, free/outcome-bearing fields, unknown fields,
  missing fields, invalid labels, malformed counts, NaNs, and infinities;
- explicit category and missing-value policies;
- duplicate questions and paper/question overlap across partitions;
- missing, contradictory, swapped, non-synthetic, and test roles;
- deterministic identity-free feature ordering;
- fixed missing indicators and train-fold medians;
- fail-closed entirely missing training features;
- deterministic fixed category columns when a category is absent from training;
- all 400 actual coordinator fits receiving exactly the expected training-fold
  rows, fresh pipeline/preprocessor objects, and independently calculated
  training-only medians/means/scales;
- fold-specific validation sentinels and extremes never reaching actual
  coordinator `fit` calls;
- a negative-control global-fit implementation being rejected by an equivalent
  validation-row sentinel assertion;
- deterministic 20-repeat fold construction, one validation assignment per paper
  per repeat, complete folds, class support, and paper isolation;
- 100 scores per `C`, order-invariant CV results, and the smaller-`C` exact tie;
- exclusion of external validation labels from fitting/refit interfaces;
- configuration-override attack rejection, implementation-identity and
  training-data fingerprint sensitivity;
- strict JSON duplicate/nonstandard-number rejection, fold/CV metadata tampering,
  JSON round-trip prediction equality, and incompatible fitted-state rejection;
- training row/paper counts contradicting validated fold assignments even after
  outer-digest recomputation;
- algebraic score-summary contradictions, including mean/top-score bounds,
  negative or over-range top margins, an implied second score outside the range,
  and empty-stage numerics;
- two independent supported-runtime fits producing identical artifacts and
  predictions;
- canonical identity aliases, contradictory availability/value fields, direct
  private-preprocessor inputs, and immutable snapshot behavior; and
- validation-only threshold selection with exact score-identity matching.

These tests demonstrate the stated mechanics against deterministic synthetic
adversaries. They do not prove the absence of all possible leakage, correctness
on official data, calibration quality, model effectiveness, or production
security.

## 10. Underspecified choices and conservative resolutions

The frozen protocol does not fully define several production extraction details:

1. **Count operation order.** It says counts use `log1p` and missing values use a
   training median, but not which comes first. This package imputes the raw count
   from the raw training median and then applies `log1p`; the choice is versioned.
2. **Entirely missing training feature.** No median exists and no fallback is
   specified. This package fails closed rather than zero-filling or deleting the
   column.
3. **Availability/reason normalization.** The shadow schema exposes reason
   strings, but no final calibrator vocabulary/mapping is frozen. This package
   uses a small versioned mechanics vocabulary and requires explicit `unknown`.
   A reviewed diagnostics-to-feature mapper remains blocked.
4. **Derived-feature units.** The protocol does not freeze tokenization for
   question/answer length or exact derivations for best-to-second gap and null
   range. This package validates precomputed synthetic values but does not invent
   a production extractor.
5. **Unexpected-missingness gate.** The protocol's 1% stopping rule requires a
   reviewed distinction between structural unavailability and unexpected
   missingness. This package implements deterministic missing mechanics but does
   not claim that the official-data completeness gate is implemented.
6. **CV held-out terminology.** CV validation labels must be used to compute the
   specified selection AURC. External validation and test labels are excluded
   from model selection; no test interface exists.
7. **Implementation identity.** Source-file hashes bind artifacts to exact local
   mechanics/protocol/fingerprint-helper bytes. They establish compatibility,
   not code authorship or trust.

These resolutions are mechanics contracts, not retroactive scientific protocol
changes. Any official-data implementation requires a separate review.

## 11. Implemented versus still blocked

### Implemented and synthetic-tested

- Strict versioned synthetic observation/feature/label/partition schemas.
- Exact feature allowlist and deterministic feature ordering.
- Label/identity/outcome exclusion from the feature matrix.
- Fold-local imputation, count transformation, scaling, missing indicators, and
  fixed categorical encoding inside one pipeline.
- Fail-closed fold and partition checks.
- Algebraic score-summary invariants and training-count/assignment consistency
  checks.
- Frozen 5x20 grouped-CV mechanics, `C` selection, and synthetic train-only refit.
- Versioned JSON artifact fingerprints, compatibility validation, and round trip.
- Validation-role-only threshold-selection mechanics.

### Still `BLOCKED`

- Official dataset acquisition, registration, certification, and split checks.
- A reviewed shadow-diagnostics-to-production-feature mapper.
- Official feature completeness and <=1% unexpected-missingness gate.
- Any fitting, model selection, threshold selection, or scoring on QASPER.
- Independently sealed confirmatory execution and test access control.
- Unified scientific metric/bootstrap reporting.
- D-anchor integration/effectiveness testing and reranker changes.
- Full-QASPER behavior-equivalence tests or hosted-model diagnostics.
- Research results, performance claims, generalization claims, and deployment.

Gates A and D from Milestone 5C remain `BLOCKED`. Gate B is improved only at the
synthetic mechanics layer and does not pass for official execution.

## 12. Reproduction commands

Focused synthetic tests only:

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest \
  -p no:cacheprovider \
  tests/test_calibrator_mechanics.py -q
```

Focused mechanics plus frozen protocol helpers:

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest \
  -p no:cacheprovider \
  tests/test_shadow_protocol.py \
  tests/test_calibrator_mechanics.py -q
```

Compilation check:

```bash
./.venv/bin/python -m py_compile \
  eval/calibrator_mechanics.py \
  tests/test_calibrator_mechanics.py
```

Expected focused result at this checkpoint:

```text
84 passed
```

The combined focused command is expected to report `94 passed`.

These commands require no dataset path, model, inference service, network access,
or project result artifact.

## 13. Final interpretation

Milestone 5D makes the intended calibrator mechanics executable and adversarially
testable on synthetic data while refusing real/test roles. It narrows several
Gate B implementation gaps but does not improve data readiness, authorize an
experiment, validate the scientific method on QASPER, or establish effectiveness.

**Final verdict: `IMPLEMENTATION CHECKPOINT ONLY`, not research validation.**
