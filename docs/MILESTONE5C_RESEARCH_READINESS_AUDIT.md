# Milestone 5C — Research Readiness and Experiment-Gate Audit

**Audit type:** static repository, protocol, and local-artifact readiness review

**Audited checkpoint:** `4528fc2130294bc3c75eb0714e179e8b32ea8f8a`

**Decision:** **BLOCKED** for model-fitting development experiments and
confirmatory evaluation

**Permitted next scope:** synthetic-fixture implementation and leakage tests only;
no QASPER inference, fitting, or test-label access is authorized by this audit

## 1. Executive summary

Milestone 5B provides a credible **instrumentation foundation**, not an executable
calibrator experiment. The repository now has an explicit local-only shadow mode,
versioned diagnostics, separate cache/output defaults, resolved-path cache
containment, strict cached-record validation, deterministic protocol calculators,
and run/config/item provenance. Focused tests exercise those contracts on fakes
and temporary files.

Research execution is nevertheless blocked for two independent reasons:

1. **Data readiness is absent.** Local filesystem inspection found only QASPER
   validation and CUAD. Official QASPER train/test artifacts are absent, so their
   hashes, schemas, annotation compatibility, paper/question IDs, class counts,
   and split disjointness cannot be verified. The local validation parquet's
   SHA-256 matches the published Milestone 3 hash, but that does not verify the
   missing splits or create a fresh confirmatory test.
2. **The learned workflow is not implemented.** The protocol freezes intended
   preprocessing and logistic-regression settings, but no feature extractor,
   fit/transform boundary, repeated grouped-CV model selector, final refit,
   learned-artifact serializer/fingerprint, or partition-aware experiment
   orchestrator exists. No calibrator import or fit call is present under
   `eval/`.

The current CLI records a caller-supplied partition role and hashes the supplied
file, but it does not prove that a file is an official split, verify paper-level
disjointness across split files, or prevent a caller from loading a declared
`test` partition before a protocol/artifact freeze. Those are protocol statements,
not enforced experiment gates.

Therefore:

- **Shadow instrumentation mechanics:** `PASS` at fixture level.
- **Official-data development experiment:** `BLOCKED`.
- **Confirmatory experiment:** `BLOCKED`.
- **Calibrator effectiveness, D-anchor effectiveness, full-dataset equivalence,
  scientific generalization, and production readiness:** `UNKNOWN`.

## 2. Repository and checkpoint state

At audit time:

| Item | Status | Evidence |
|---|---|---|
| Branch | `PASS` | `main` |
| HEAD | `PASS` | `4528fc2130294bc3c75eb0714e179e8b32ea8f8a` |
| Milestone 5B ancestry | `PASS` | The audited commit is HEAD and therefore an ancestor of HEAD. |
| Remote synchronization | `PASS` | Refreshed local comparison reported `0` behind / `0` ahead against `origin/main`. |
| Tracked working tree | `PASS` | No staged or unstaged tracked changes. |
| Existing untracked work | `PARTIALLY VERIFIED` | `docs/MILESTONE5_DIAGNOSTIC_ANALYSIS.md` and the protected `results/milestone2/` path remain untracked. The latter was not opened or inspected. |
| Protected report | `PASS` | SHA-256 `d90ff73c4d7f5c20fe46898ab14088881c3cf884a114e726e33ca6f5be57c568`. |

No experiment, inference, model loading, fit, API call, or dataset/model download
was performed for this audit.

## 3. Evidence inventory

### 3.1 Local dataset artifacts

Only filesystem names, byte sizes, timestamps, and hashes were inspected; dataset
contents were not parsed.

| Local artifact | Bytes | SHA-256 | Readiness interpretation |
|---|---:|---|---|
| `data/raw/qasper_validation.parquet` | 4,749,127 | `089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222` | `PASS` for identity continuity with the published Milestone 3 validation artifact; already inspected and unsuitable as a new confirmatory test. |
| `data/raw/cuad_v1.json` | 40,128,638 | `ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae` | `UNKNOWN` for this protocol; no CUAD loader, split protocol, or experiment role is implemented here. |
| Official QASPER train | — | — | `BLOCKED`: absent locally. |
| Official QASPER test | — | — | `BLOCKED`: absent locally. |

The QASPER paper counts cited in the research plan (train 888, validation 281,
test 416) are documentation claims, not verification of local train/test bytes
(`docs/MILESTONE4_RESEARCH_PLAN.md:357-374`). Presence in a future directory would
still not prove official identity or compatibility.

### 3.2 Published baseline evidence

The verified Milestone 3 report establishes the validation baseline, not future
experiment readiness:

- Validation hash and 281-paper identity:
  `docs/FULL_QASPER_EVALUATION.md:24-33`.
- Pinned embedding, reranker, and reader revisions:
  `docs/FULL_QASPER_EVALUATION.md:35-49`.
- Paper-disjoint Milestone 3 calibration/test split:
  `docs/FULL_QASPER_EVALUATION.md:51-68`.
- Integrity and provenance checks:
  `docs/FULL_QASPER_EVALUATION.md:170-199`.

Milestone 5's post hoc analysis explicitly reports that 0/5,015 old records have
Milestone 4B diagnostics (`docs/MILESTONE5_DIAGNOSTIC_ANALYSIS.md:66-91`). Those
records cannot be relabeled as instrumented examples.

### 3.3 Current implementation evidence

- Shadow workflow and strict diagnostic contract: `eval/shadow.py`.
- Frozen protocol calculators and D-anchor helper: `eval/shadow_protocol.py`.
- Canonical run/config/item fingerprints and validated cache loading:
  `eval/provenance.py`.
- Explicit partition-role CLI, run manifest, diagnostics collection, and pending
  rejected-row recomputation: `eval/run_eval.py`.
- Protocol/claim boundary: `docs/MILESTONE5B_SHADOW_PROTOCOL.md`.
- Schema semantics: `docs/CONFIDENCE_DIAGNOSTICS.md`.

## 4. Capability matrix

| Capability | Status | Evidence and limit |
|---|---|---|
| Explicit local-only shadow entry point | `PASS` | `eval/run_eval.py:331-383`; hosted readers are rejected before `load_qasper`. |
| Separate default shadow cache/output paths | `PASS` | `eval/shadow.py:12-14`; `eval/run_eval.py:375-445`. |
| Ordinary-cache path containment | `PASS` | Resolved equality/descendant/symlink checks in `eval/shadow.py:60-76`; targeted tests in `tests/test_shadow_experiment.py:63-84`. |
| Additive/observational diagnostics contract | `PARTIALLY VERIFIED` | Fake-pipeline and real-loop synthetic tests compare business outputs (`tests/test_confidence_diagnostics.py:127-227,344-423`). No full-distribution or real-dataset shadow equivalence run exists. |
| Versioned inference-time schema | `PASS` | `docs/CONFIDENCE_DIAGNOSTICS.md:1-157`; gold metrics are excluded from diagnostics. |
| Strict cache row validation | `PASS` | Stage, candidate, score-summary, reader, and decision checks in `eval/shadow.py:79-348`; adversarial tests in `tests/test_eval_cache.py:335-627`. |
| Rejected-row rescheduling | `PASS` | `EvaluationCache.load` omits rejected rows (`eval/provenance.py:116-162`); `pending_items` returns them for future recomputation (`eval/run_eval.py:266-268,504-508`). |
| Run/config/item provenance | `PASS` | `eval/provenance.py:24-72,96-184`; deterministic payload in `eval/run_eval.py:123-231`. |
| Dataset partition role in fingerprint | `PASS` | `eval/run_eval.py:158-174`; role tests in `tests/test_eval_cache.py:630-700`. |
| Official partition identity verification | `NOT IMPLEMENTED` | `validate_partition_role` checks an allowed role and recognizable filename tokens only (`eval/shadow_protocol.py:138-154`). No trusted hash registry or official-source proof exists. |
| Cross-file paper-ID disjointness verifier | `NOT IMPLEMENTED` | Protocol requires it (`docs/MILESTONE5B_SHADOW_PROTOCOL.md:96-116`); no implementation was found. |
| Per-feature extraction contract/allowlist | `NOT IMPLEMENTED` | Diagnostics are saved beside gold labels in evaluation rows; no leakage-safe feature builder separates model inputs from evaluation labels. |
| Feature completeness/missingness report | `NOT IMPLEMENTED` | The protocol's <=1% unexpected-missingness stop rule is documented at `docs/MILESTONE5B_SHADOW_PROTOCOL.md:261-271`, but no checker or report artifact exists. |
| Fold-only imputation/scaling/encoding | `NOT IMPLEMENTED` | Settings are metadata only (`eval/shadow_protocol.py:107-120`); no transformer or fit/transform code exists. |
| Deterministic paper fold assignment | `PASS` | `grouped_cv_assignments`, five folds, 20 repeats, seed 20261028 (`eval/shadow_protocol.py:15-21,157-179`). |
| Repeated grouped-CV estimator fitting and C selection | `NOT IMPLEMENTED` | No `LogisticRegression`, preprocessor, fit, score aggregation, smaller-C tie selector, or refit workflow exists; only metadata is frozen. |
| AURC definition/calculation | `PASS` | Fixed 0.00-1.00 grid, whole-tie inclusion, target-grid trapezoids (`eval/shadow_protocol.py:22,182-231`); deterministic test at `tests/test_shadow_protocol.py:57-72`. |
| Matched-coverage threshold helper | `PASS` | `eval/shadow_protocol.py:269-300`; helper is not partition-aware and is not connected to model artifacts. |
| 5% empirical-risk threshold helper | `PASS` | `eval/shadow_protocol.py:303-316`; helper is not a statistical guarantee and does not enforce validation-only calling context. |
| Paired paper-cluster AURC bootstrap | `PASS` | Defaults 5,000/seed 20261029 and paired paper multiplicities (`eval/shadow_protocol.py:319-360`). Other declared metric intervals are not implemented in a unified future report pipeline. |
| D-anchor deterministic selector | `PASS` | `eval/shadow_protocol.py:375-421`; tests cover ties, zero/one/two anchors, short pools, RRF fallback, and uniqueness. |
| D-anchor evaluation/integration | `NOT IMPLEMENTED` | The helper is not wired into A-E configurations or an experiment runner; no effect estimate exists. |
| Learned preprocessor/model artifact fingerprinting | `NOT IMPLEMENTED` | Required by `docs/MILESTONE4_RESEARCH_PLAN.md:418-433`; no learned artifacts exist. |
| Confirmatory test access gate | `NOT IMPLEMENTED` | The CLI accepts `--dataset-partition test`; it does not require a frozen protocol manifest, official hash match, authorization record, or sealed-label gate before loading the file. |

## 5. Intended data flow and leakage audit

### 5.1 Intended future flow

The frozen design implies this sequence:

1. Verify official train/validation/test bytes, roles, schemas, annotations, IDs,
   and disjoint paper sets.
2. Run unchanged local inference with diagnostics and preserve run provenance.
3. Convert diagnostics into an inference-only feature table; keep labels/gold
   fields in a separate evaluation table joined only by immutable IDs.
4. On official **train** papers, construct repeated paper-grouped folds.
5. Within every training fold only, fit imputation, missingness indicators,
   encoding, scaling, and the candidate logistic model.
6. Select `C` by mean held-out-fold AURC with the frozen smaller-`C` tie rule.
7. Refit the selected pipeline on all official train papers and fingerprint every
   learned artifact.
8. On official **validation** only, choose the matched-coverage and empirical-risk
   operating points.
9. Freeze code, features, artifacts, thresholds, metrics, report template, and
   execution budget.
10. Evaluate once on an independently verified official test split; compute paired
    paper-cluster intervals and preserve all declared null/negative results.

Current code implements steps 2's record shape/provenance mechanics and several
metric/split helpers, but not the workflow spanning steps 1 and 3-10.

### 5.2 Leakage boundaries

#### Partition identity — `BLOCKED`

`--dataset-partition` is explicit and fingerprinted, which prevents silent
manifest relabeling. It remains a caller assertion. `validate_partition_role`
only rejects recognized filename-token conflicts (`eval/shadow_protocol.py:138-154`).
It cannot prove official origin or disjointness.

**Risk:** a renamed, modified, overlapping, or unofficial file can receive an
apparently valid role and independent run fingerprint.

**Smallest safe correction:** add a versioned dataset registry/verification
command that records trusted source identity, expected SHA-256, byte count,
schema fingerprint, paper/question-ID hashes, annotation counts, and pairwise
paper-set disjointness. Commit metadata only, never raw data.

#### Test access — `BLOCKED`

The protocol prohibits opening test outcomes before freeze
(`docs/MILESTONE5B_SHADOW_PROTOCOL.md:96-116`), but the general evaluator will
load any supplied `test` file after role validation (`eval/run_eval.py:331-384`).

**Risk:** procedural leakage remains possible even though it is documented as
forbidden.

**Smallest safe correction:** introduce a separate, fail-closed experiment
orchestrator. A test-role run must require a frozen protocol fingerprint, verified
dataset-registry entry, frozen learned-artifact hashes, locked thresholds, report
schema, explicit authorization record, and output path. Ordinary instrumentation
CLI examples should remain development-only.

#### Feature/label separation — `BLOCKED`

`build_evaluation_record` stores diagnostics together with `gold_answers`,
`answerable`, and `gold_evidence` (`eval/run_eval.py:271-328`). This is appropriate
for evaluation records but unsafe as an implicit model-input table.

**Risk:** a future generic flattening step could accidentally include labels,
gold evidence, retrieval recall, citation correctness, or answer F1.

**Smallest safe correction:** define a versioned feature schema with an explicit
allowlist rooted in inference fields (`question`-derived controls plus
`diagnostics`), a separate label schema, forbidden-key tests, and a serialized
feature-schema hash.

#### Fold preprocessing and model selection — `BLOCKED`

Paper assignments are implemented, but no pipeline consumes them. The protocol's
train-fold-only imputation/scaling/encoding and model selection are metadata
(`eval/shadow_protocol.py:41-120`; `docs/MILESTONE5B_SHADOW_PROTOCOL.md:118-194`).

**Risk:** a later implementation could fit transforms globally, tune `C` on
validation/test, or apply inconsistent missingness semantics while appearing to
follow the frozen plan.

**Smallest safe correction:** implement `fit(train_rows)` / `transform(rows)`
objects and a grouped-CV coordinator using synthetic records first. Assert that
fold-held-out sentinel values cannot affect training medians, categories, means,
or scales; assert each paper appears in exactly one held-out fold per repeat.

#### Fold construction — `PARTIALLY VERIFIED`

The assignment helper is deterministic and order-invariant in tests
(`tests/test_shadow_protocol.py:44-54`). It checks only total paper count >=5.
Because each stratum starts round-robin assignment at fold zero, a small fixture
with both strata individually smaller than five can leave a fold empty even when
the combined paper count is at least five.

**Risk:** future small development subsets could produce invalid folds or
class-deficient training/held-out partitions.

**Smallest safe correction:** validate nonempty folds and required class support;
either fail closed or use a documented deterministic offset/balancing rule. Add
adversarial small-stratum tests before using the helper for fitting.

#### Threshold selection — `PARTIALLY VERIFIED`

The matched-coverage and risk helpers implement the frozen arithmetic and
deterministic tie rules (`eval/shadow_protocol.py:234-316`). They accept only
scores/labels and do not carry partition identity.

**Risk:** a caller can accidentally select a threshold on test labels.

**Smallest safe correction:** make selection consume a verified calibration
artifact whose partition role and parent model fingerprint are checked; write a
locked threshold artifact with the selector implementation hash.

#### Bootstrap and comparisons — `PARTIALLY VERIFIED`

Paired AURC paper-cluster resampling is implemented and deterministic. The future
protocol also declares retrieval, answer-quality, and operating-point differences
(`docs/MILESTONE5B_SHADOW_PROTOCOL.md:216-259`), but no unified result/report
executor enforces paired samples and denominators for all of them.

**Smallest safe correction:** implement metric-specific paired estimators behind
a shared paper-resample index generator and test denominator/report metadata.

#### Artifacts and provenance — `PARTIALLY VERIFIED`

Run, config, item, source, model, package, and protocol metadata are fingerprinted
(`eval/provenance.py`; `eval/run_eval.py:123-231`). No learned artifacts exist.

**Smallest safe correction:** future scaler/encoder/model files must be written
atomically, content-hashed, linked to feature/data/fold/protocol fingerprints, and
loaded only on exact manifest match. Training failures and rejected rows must be
preserved rather than silently dropped.

## 6. Prioritized findings

### BLOCKER — Official train/test artifacts are absent

**Evidence:** local `data/raw/` contains only `qasper_validation.parquet` and
`cuad_v1.json`; protocol lines 19-29 also declare train/test absent.

**Impact:** no valid official-train development fit or independent confirmatory
evaluation can begin. Validation is already inspected and cannot substitute for
test.

**Fix:** complete Gate A using a reviewed, metadata-only dataset verification
process and explicit acquisition authorization.

### BLOCKER — Leakage-safe calibrator workflow is not implemented

**Evidence:** no estimator/preprocessor imports or fit calls under `eval/`; only
`LOGISTIC_SETTINGS` metadata appears in `eval/shadow_protocol.py:27-38`.

**Impact:** the protocol cannot currently produce a fitted candidate, select `C`,
refit, or create reproducible learned artifacts.

**Fix:** complete Gate B using synthetic fixtures before official data.

### HIGH — Official split identity/disjointness is asserted, not enforced

**Evidence:** `validate_partition_role` is filename-token validation only;
`build_run_payload` fingerprints one supplied partition but has no cross-split
registry (`eval/shadow_protocol.py:138-154`; `eval/run_eval.py:123-174`).

**Impact:** an unofficial or overlapping artifact could be mislabeled without a
cryptographic comparison to other splits.

**Fix:** dataset registry plus schema/ID/disjointness verifier.

### HIGH — Test-label access and threshold partition boundaries are procedural

**Evidence:** CLI accepts `test`; threshold helpers have no partition-bearing
input (`eval/run_eval.py:331-384`; `eval/shadow_protocol.py:269-316`).

**Impact:** accidental test-driven model/threshold choices are possible.

**Fix:** fail-closed experiment orchestrator and partition-stamped artifacts.

### HIGH — Feature allowlist and missingness gate are absent

**Evidence:** gold labels and diagnostics coexist in records; <=1% missingness is
only a stopping rule in protocol text (`eval/run_eval.py:271-328`;
`docs/MILESTONE5B_SHADOW_PROTOCOL.md:261-271`).

**Impact:** label leakage or inconsistent zero/missing treatment could invalidate
the experiment.

**Fix:** versioned inference-only extractor, forbidden-field tests, explicit
missing indicators, and completeness report.

### MEDIUM — Fold validity checks are incomplete for small strata

**Evidence:** total count is checked, but nonempty/class-supported folds are not
(`eval/shadow_protocol.py:157-179`).

**Impact:** synthetic or restricted development subsets may yield empty or
class-deficient folds.

**Fix:** fail-closed fold validation and adversarial tests.

### MEDIUM — Real diagnostic completeness and overhead are unknown

**Evidence:** behavior tests use fakes/synthetic loops; documentation explicitly
limits equivalence evidence (`docs/CONFIDENCE_DIAGNOSTICS.md:159-194`). No
instrumented QASPER run or overhead benchmark has occurred.

**Impact:** real missingness, record size, latency, and full-distribution output
equivalence are unknown.

**Fix:** after Gates A/B and explicit compute approval, run a bounded predeclared
development-only feature-completeness slice; do not use official test.

### MEDIUM — D-anchor is a helper, not an evaluated arm

**Evidence:** only `d_anchor_select` and unit tests exist; no runner configuration
or output artifact references it.

**Impact:** deterministic mechanics are known, effectiveness is not.

**Fix:** keep separate from calibrator; evaluate only after protocol/data gates
and explicit authorization.

## 7. Readiness gates

### Gate A — Data readiness

**Current status:** `BLOCKED`

**Supporting evidence:** validation bytes/hash are present and match Milestone 3;
official train/test files are absent. No current artifact proves split
disjointness or loader compatibility for train/test.

**Unmet prerequisites:**

- Explicit authorization to acquire official artifacts.
- Trusted source/revision/license record for train, validation, and test.
- SHA-256, byte size, schema fingerprint, and loader compatibility for each.
- Counts of papers, questions, annotations, answer forms, and usable answers.
- Unique paper/question IDs and pairwise paper-set disjointness.
- Frozen answerability aggregation rule and mixed-annotation audit.
- Confirmation that final test annotations are available for the declared metrics
  and can remain sealed until Gate D.

**Pass criteria:** a reviewed metadata manifest containing all evidence above,
with exact hashes and pairwise disjointness; raw files remain ignored and outside
Git.

**Prohibited until pass:** official-data model fitting, train/validation
statistics used for design changes, opening or summarizing final-test outcomes,
and any confirmatory claim.

### Gate B — Protocol readiness

**Current status:** `BLOCKED`

**Supporting evidence:** metric/split/threshold metadata and helper functions are
frozen; the executable learned workflow and leakage guards are absent.

**Unmet prerequisites:**

- Versioned feature allowlist and label separation.
- Implemented question-form categorization and answer-length semantics.
- Fit/transform preprocessing with training-fold-only state.
- Explicit unavailable-versus-unexpected-missing handling and <=1% checker.
- Repeated grouped-CV coordinator, deterministic `C` selection, and final refit.
- Fold nonempty/class-support checks.
- Partition-aware threshold locking.
- Learned artifact serialization and cryptographic manifest.
- Unified paired metric/bootstrap report implementation.
- Failure, duplicate, missing-row, and no-drop integrity report.
- Tests proving test records cannot affect training or selection.

**Pass criteria:** every item exists as code plus deterministic synthetic tests;
all configuration and implementation hashes appear in a protocol manifest before
official outcome inspection.

**Prohibited until pass:** fitting on QASPER, selecting hyperparameters or
thresholds from official records, and D-anchor/calibrator combined evaluation.

### Gate C — Development-experiment readiness

**Current status:** `BLOCKED` for fitting; instrumentation mechanics are
`PARTIALLY VERIFIED`

**Minimum permitted synthetic/local fixture:**

- At least 10 synthetic paper IDs so all five folds can be nonempty, with both
  answerability classes and at least five papers containing synthetic negatives.
- Multiple questions per paper to prove grouping.
- Hand-built `vistaar-confidence-diagnostics-v1` records covering available and
  unavailable dense/BM25/RRF/reranker/reader states.
- Deliberate missing values and fold-exclusive sentinel values that would expose
  global imputation/scaling/category leakage.
- Fixed expected feature vectors, fold assignments, training statistics,
  smaller-`C` tie result, refit manifest, threshold result, and bootstrap output.
- Fake estimator or tiny synthetic logistic fit only after explicit implementation
  approval; no pretrained model, QASPER file, hosted service, or scientific claim.

**Unmet prerequisites:** Gate B implementation and a reviewed fixture/test plan.

**Pass criteria:** all synthetic mechanics and leakage tests pass; learned
synthetic artifacts reproduce byte-for-byte or by content hash; no official data
or model inference is touched.

**Prohibited until pass:** official QASPER fit, feature selection from validation,
test access, effectiveness claims, and production changes.

### Gate D — Confirmatory-experiment readiness

**Current status:** `BLOCKED`

**Supporting evidence:** Gates A-C are blocked; validation is already inspected;
no final candidate or learned artifact exists.

**Unmet prerequisites:**

- Gates A, B, and C pass.
- One frozen primary candidate and declared baselines.
- Train-only model/preprocessing selection completed without test access.
- Validation-only operating points locked and fingerprinted.
- Final feature schema, model, artifact hashes, metrics, denominators, tie rules,
  bootstrap seed, report template, and failure policy frozen in Git.
- Independent review of final-test label sealing and one-shot execution path.
- Predeclared power/inconclusive rule using verified negative-paper support.
- Exact local-only compute budget, expected runtime/storage, output paths, and
  explicit user authorization.
- Independent post-run integrity/recalculation plan.

**Pass criteria:** a signed/reviewed execution manifest references exact code,
data, partitions, learned artifacts, thresholds, seeds, outputs, and budget;
normal fast-forward repository state and clean protected artifacts are confirmed.

**Prohibited until pass:** final-test inference, label inspection, retuning,
subgroup-driven changes, extra models, confirmatory language, and deployment
claims.

## 8. Smallest safe next implementation milestone

Implement a **synthetic-only leakage-safe calibrator mechanics package** without
reading QASPER or fitting a scientific candidate:

1. Define versioned inference-feature and separate label schemas.
2. Implement allowlisted feature extraction from synthetic diagnostics.
3. Implement fit/transform preprocessing with missing indicators and fold-local
   state.
4. Add fold validation and a grouped-CV coordinator with a fake estimator first.
5. Implement deterministic model-selection/refit artifact manifests and content
   hashes.
6. Make threshold helpers accept only partition-stamped calibration artifacts.
7. Prove leakage resistance with sentinel-valued synthetic held-out folds.

This milestone can pass Gate B/C mechanics without datasets, pretrained models,
inference, or scientific claims. Dataset acquisition/verification should be a
separate, explicitly authorized Gate A action; official test must remain sealed.

## 9. Explicit prohibited actions and unresolved questions

### Prohibited by this audit

- Downloading or parsing new official train/test artifacts without approval.
- Running shadow or ordinary QASPER inference.
- Fitting a calibrator on validation or any current project records.
- Evaluating D-anchor effectiveness.
- Calling Gemini or any hosted API.
- Treating Milestone 3 held-out questions as a fresh confirmatory test.
- Flattening evaluation records into features without an allowlist.
- Opening final-test labels before Gates A-D and execution authorization.
- Committing raw datasets, caches, model artifacts, or generated results.

### Unresolved questions (`UNKNOWN`)

- Are official train/test artifacts and test annotations available in a locally
  usable, license-compatible form?
- Do their schemas and answerability semantics match the current loader?
- How many consensus-unanswerable papers/questions exist in each official split?
- Is the declared 10% AURC effect detectable with the verified negative support?
- What are real diagnostic missingness, record-size, latency, and memory costs?
- Does the local reader's behavior remain exactly invariant over a bounded real
  development slice with diagnostics enabled?
- Does D-anchor improve Ret-R@4 without harming answer quality?
- Does any calibrator improve ranking or selective utility beyond the current
  margin on unseen papers?
- Does any result transfer beyond English scientific QASPER?

## 10. Reproducibility and audit trail requirements

Every future authorized stage must preserve:

- Git commit, clean/deliberately scoped working tree, and implementation hashes.
- Dataset source/revision, SHA-256, size, schema fingerprint, ID-set hashes, and
  partition role.
- Paper/question lists or cryptographic commitments and disjointness report.
- Exact model revisions, package versions, platform, seeds, worker count, and
  chunk/retrieval settings.
- Diagnostic, feature, label, preprocessing, model, threshold, metric, and report
  schema versions.
- Fold assignments and train/held-out paper IDs for every repeat.
- Content hashes for learned imputers, encoders, scalers, estimators, and locked
  threshold artifacts.
- Cache hit/miss/rejection/malformed counts and zero dropped rows.
- Full declared arm list, negative/null outcomes, failure logs, and stopping-rule
  decision.
- Paired paper-resample indices or deterministic seed/procedure sufficient for
  exact interval reproduction.
- New, non-overwriting output/cache namespaces; no protected or prior baseline
  artifact mutation.

## 11. Interview-safe summary

“Vistaar has published local-only shadow instrumentation with versioned retrieval
and reader diagnostics, strict cache validation, run/config/item provenance, and
deterministic protocol calculators. Unit fixtures support the observational
contract, but no full instrumented dataset run, calibrator fit, or D-anchor
evaluation has occurred. Official QASPER train/test artifacts are absent and
unverified, and the leakage-safe feature/preprocessing/CV/refit executor is not
implemented. The project is ready for a synthetic-only mechanics milestone, not
for a scientific model-fitting or confirmatory experiment.”

## 12. Audit procedure

This audit used only:

- Git branch/HEAD/status/log/remote metadata and fast-forward ancestry checks.
- Source/document/test inspection of the paths cited above.
- Local `data/raw/` directory names plus `stat` and SHA-256; dataset contents were
  not parsed.
- SHA-256 verification of the protected Milestone 5 report.
- Static searches confirming no calibrator fit/preprocessor implementation and no
  D-anchor integration outside its helper/tests.

No tests were rerun because the milestone requested a planning audit and source
inspection was sufficient for the readiness claims. No protected results path,
model, dataset contents, inference path, experiment, or hosted service was
accessed.

## 13. Post-audit status note — Milestone 5D

Milestone 5D subsequently implemented and synthetic-tested the smallest mechanics
package recommended in Section 8. See
`docs/MILESTONE5D_CALIBRATOR_MECHANICS.md` and
`eval/calibrator_mechanics.py`.

The following original findings are now `PARTIALLY VERIFIED` at the synthetic
mechanics layer only:

- strict versioned feature/label/partition schemas and feature allowlisting;
- fold-local preprocessing inside a single scikit-learn pipeline;
- validated 5-fold, 20-repeat paper-grouped CV and train-only synthetic refit;
- versioned JSON fitted-state/artifact fingerprints and compatibility checks; and
- validation-role-only wrappers around frozen threshold helpers.

This note does not change the original evidence inventory or improve official
data readiness. Gates A and D remain `BLOCKED`; Gate B remains `BLOCKED` for
official execution because dataset registration, a reviewed diagnostics-to-
feature mapper, the official missingness gate, unified reporting, and authorized
execution controls remain absent. No QASPER model was fitted or evaluated.

Milestone 5D-R subsequently repaired the synthetic mechanics audit findings:
configuration fingerprints are independently bound to loaded/current metadata,
strict JSON rejects duplicate keys and nonstandard constants, identities are
canonical, validated snapshots are read-only and reused, feature/status
contradictions fail closed where the source contract is defined, implementation
identity is fingerprinted, and coordinator-level sentinels cover all 400 CV fits.
Milestone 5D-R1 then added the remaining derived score-summary invariants,
cross-checked declared training counts against validated fold-assignment
identities, and corrected private-API and negative-control wording.
These repairs strengthen only synthetic mechanics. They do not change any
official-data or confirmatory gate status.
