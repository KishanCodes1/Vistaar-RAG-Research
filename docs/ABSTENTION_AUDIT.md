# Independent Audit of the Abstention Evaluation

**Date:** 2026-10-28  
**Commit audited:** `e87fa0c`  
**Recommendation:** **NO-GO** for scaling the current evaluator to full QASPER.

## Scope and method

This audit inspected `eval/`, `results/balanced/`, `paper/PAPER.md`,
`README.md`, `PROJECT_STATE.md`, the committed per-question records, and the
persisted local index. Metrics were independently recomputed from JSONL records.
Additional diagnostics and a clean-index config-E rerun were written outside
the repository. The committed milestone was not modified.

`python -m pytest tests/ -q` passes: **12 passed**. The current tests do not
cover repeated indexing, cache invalidation, or record/sweep consistency.

## Record-level reproduction

### Values that reproduce

| Dataset | Questions (A/U) | AUC | Peak balanced accuracy | Coverage there | Best conditional selective F1 |
|---|---:|---:|---:|---:|---:|
| Negative-enriched (`results/balanced`) | 231 (171/60) | 0.6172 | 0.5662 @ 0.10 | 0.2814 | 0.0549 @ 0.40 (coverage 0.1558) |
| Natural sample | 127 (122/5) | 0.6959 | 0.6393 @ 0.10 | 0.2677 | 0.0827 @ 0.50 (coverage 0.1496) |

The balanced sweep reproduces from `results/balanced/records.jsonl` within the
artifact's four-decimal rounding. Coverage, balanced accuracy, AUC, and counts
also reproduce for the natural sample.

The two peak balanced accuracies are **the same metric at the same threshold on
different samples**, not conflicting corrections:

- Balanced: `(54/171 + 49/60) / 2 = 0.5662`; coverage `65/231 = 0.2814`.
- Natural: `(34/122 + 5/5) / 2 = 0.6393`; coverage `34/127 = 0.2677`.

The natural value is elevated by correctly abstaining on all five negatives and
is consequently unstable. Neither value should overwrite the other.

### Natural selective-F1 artifact does not reproduce

The natural `results/sweep_E.json` is stale relative to the current
`eval/sweep.py`, which correctly evaluates the counterfactual raw span. Direct
recomputation gives:

| Threshold | Committed selective F1 | Recomputed from `raw_prediction` |
|---:|---:|---:|
| 0.00 | 0.0190 | **0.0507** |
| 0.10 | 0.0489 | **0.0561** |
| 0.20 | 0.0628 | 0.0628 |
| 0.50 | 0.0827 | 0.0827 |

The committed values reproduce exactly if the post-abstention `prediction` is
used instead. That incorrectly leaves the default abstention message in place
when simulating lower thresholds. Therefore the reported natural gain is not
4.4x; under the current definition it is **1.63x** (`0.0507 -> 0.0827`), an
absolute gain of 0.0320. The 0.0827 operating-point value itself reproduces.

The balanced artifact was generated with `raw_prediction` and does not have
this inconsistency.

## Integrity checks

- Label: answerable iff at least one QASPER annotation is not marked
  unanswerable. This matches `eval/qasper.py` and the saved records.
- Direction: higher confidence predicts answerable; `confidence >= threshold`
  means answer. Mean confidence is 0.1528 vs 0.1195 for balanced A/U and 0.1560
  vs 0.0142 for natural A/U, so the implemented direction is correct.
- Missing/duplicate data: zero missing confidence or retrieval scores, zero
  duplicate question IDs, and no saved answerable/empty-gold or
  unanswerable/nonempty-gold conflicts.
- Class prevalence: natural is 96.1% answerable. The so-called balanced slice
  is 74.0% answerable and is better described as **negative-enriched**, not
  class-balanced.
- Annotation ambiguity: 14/231 balanced and 13/127 natural questions have
  mixed answerable/unanswerable annotations. Excluding mixed labels changes
  AUC to 0.6251 and 0.7128, respectively. The label rule is defensible but must
  be disclosed and accompanied by sensitivity analysis.
- The score is a sigmoid-transformed SQuAD2 span-vs-null logit margin. No ECE,
  Brier score, or reliability analysis exists; it must not be described as a
  calibrated probability.
- No path from gold answers or evidence into retrieval/reader inference was
  found. The leakage is procedural: label-conditioned sampling, threshold
  tuning on evaluation records, and state/cache reuse across runs.

## Blocking provenance defect: persistent-index contamination

Fresh chunking with the committed code produces **2,987** chunks for the 40
natural documents and **3,784** for the 51 negative-enriched documents. The
reported counts are **7,391** and **4,645**.

Root cause:

1. `Store.get()` loads an existing namespace from `data/store`.
2. `Namespace.add()` appends every chunk without checking existing chunk IDs.
3. `eval/run_eval.py` indexes every selected document without resetting its
   namespace.
4. The cache is keyed only by config file and question ID, without a code,
   model, document, chunking, or index fingerprint.

The persisted state confirms repeated whole-document appends. Across the
natural documents it currently contains 7,782 rows for 2,987 unique chunk IDs;
the extra 391 rows after the reported 7,391 are exactly one new copy of the five
documents shared with the later balanced run. The balanced namespaces contain
4,645 rows for 3,784 unique IDs, with all 861 duplicates in those five shared
documents. Twenty-six questions overlap the two runs and have identical cached
outputs.

RRF returns unique chunk IDs, but duplicate candidates still consume the fixed
top-k lists and each repeated ID receives another reciprocal-rank contribution
before final deduplication. The defect can therefore alter retrieval and reader
confidence.

A clean temporary-store rerun of config E over the same records found:

| Metric | Saved records | Clean-index rerun |
|---|---:|---:|
| Natural AUC | 0.6959 | **0.6459** |
| Natural peak balanced accuracy | 0.6393 | **0.6352** |
| Balanced AUC | 0.6172 | **0.6181** |
| Balanced peak balanced accuracy | 0.5662 | **0.5662** |

Top-4 retrieval changed for 49/127 natural questions and 12/231 balanced
questions; raw predictions changed for 17 and 4 questions. The primary
balanced result happened to remain close, but that does not repair provenance,
and scaling with the current runner would append more duplicates while mixing
old cached and new predictions.

## Threshold selection and practical meaning

All reported operating points were selected and scored on the same evaluation
records. `sweep.py` places the maximum conditional selective F1 in `best`; the
paper separately reports the maximum balanced accuracy. Both are post-hoc test
set choices and are optimistic.

The selective-F1 metric averages only over answered questions known to be
answerable. It does not penalize answered-unanswerable questions in its
denominator. At the balanced F1 optimum, only 28 answerable questions contribute
and 8 unanswerable questions are also answered; total coverage is 36/231.
Accordingly it is a conditional answer-quality statistic, not standalone
selective-system utility. The balanced 1.43x ratio (`0.0385 -> 0.0549`)
reproduces numerically, but its absolute gain is only 0.0164.

Document-cluster bootstrap diagnostics (fixed reported thresholds, not adjusted
for threshold search) give:

- Balanced AUC 95% interval: **[0.544, 0.685]**.
- Natural AUC 95% interval: **[0.461, 0.906]**.
- Balanced selective-F1 gain: +0.0164, interval **[-0.0219, 0.0599]**.
- Natural gain under the current raw-span definition: +0.0320, interval
  **[-0.0248, 0.1016]**.

Neither selective-F1 gain is statistically resolved. In a diagnostic 100-times
repeated five-fold split grouped by document, training-fold selective-F1 threshold
selection yielded mean out-of-fold selective F1 **0.0381**, versus the balanced
full-coverage baseline **0.0385** (mean coverage 0.162; balanced accuracy 0.509).
The apparent balanced gain was not observed out of fold in this sample.

## Unresolved issues

- The cache has no manifest or per-record index fingerprint, so the exact
  historical namespace state for every cached prediction cannot be proven,
  although the append mechanism and aggregate duplicate counts are confirmed.
- Hugging Face model revisions are not pinned; future byte-for-byte reruns are
  not guaranteed even after fixing the local store.
- No untouched QASPER holdout has yet been evaluated under a locked threshold.

## Severity and required fixes

| Severity | Finding | Required fix |
|---|---|---|
| **Blocker** | Stateful indices append duplicate chunks and caches mix runs | Use a new run-scoped store or reset namespaces before indexing; reject duplicate chunk IDs; fingerprint all cache inputs |
| **High** | Natural 0.019 baseline and 4.4x gain use stale post-abstention predictions | Regenerate every sweep from immutable records with `raw_prediction`; add artifact-consistency tests |
| **High** | Thresholds are selected on the evaluation set | Tune one predeclared objective on calibration data, then lock it before test evaluation |
| **Moderate** | Selective F1 has low support and ignores false answers in its denominator | Rename it conditional answered-answerable F1 and report a utility/risk metric that penalizes answered negatives |
| **Moderate** | Label-conditioned sample is not class-balanced; annotation disagreement is hidden | Call it negative-enriched, retain natural-prevalence reporting, and add label sensitivity |

## Defensible evaluation protocol

1. Split by `doc_id`, never by question, to prevent questions from one paper
   crossing calibration/test or CV folds.
2. Use train/development papers for all model, signal, and threshold choices.
   Lock one objective and threshold before an untouched natural-prevalence test.
3. If a fixed holdout is too small, use nested stratified-group cross-validation:
   inner folds choose signal/threshold; outer folds estimate performance.
4. Report the negative-enriched slice only as a predeclared stress test. Do not
   use its labels to tune and evaluate on the same questions.
5. Compute intervals with paper-level cluster bootstrap and report AUC,
   balanced accuracy, coverage, answered-unanswerable rate, and a loss/utility
   over every covered example.
6. Record a run manifest containing commit, exact model revisions, input hashes,
   chunk settings, unique chunk counts, cache fingerprint, and output hashes.

## Go/no-go

**NO-GO.** Do not start a full-QASPER run with the current index/cache lifecycle.
First fix run isolation and cache fingerprinting, add regression tests, rerun
from an empty store, regenerate all derived sweeps, and lock a document-grouped
calibration/test protocol. After those conditions are met, a controlled full
run is appropriate; the existing records are suitable only for debugging and
record-conditional analysis, not final performance claims.
