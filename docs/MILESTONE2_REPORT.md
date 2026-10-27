# Milestone 2 — Evaluation Integrity and Reproducibility

**Date:** 2026-10-28  
**Milestone 1 preserved:** `e87fa0c075ab57bfdfdb7d7ce5750ae398b17f09`  
**Evaluation-scaling recommendation:** **GO, with controls**  
**Abstention-performance claim:** **NO-GO**; held-out performance remains weak.

## What changed

### Index lifecycle

- `Namespace.add` is now strict append and rejects duplicate chunk IDs.
- `Namespace.upsert_document` replaces only one deterministic document while
  retaining unrelated documents in the namespace.
- Re-indexing identical filename/content is idempotent across process reloads.
- Persisted snapshots are replaced rather than appended and are validated for
  vector/chunk count consistency and unique IDs when loaded.
- Evaluation indices live under a run-fingerprinted cache directory. The
  historical `data/store` was never opened for writing, deleted, or repaired in
  place.

### Cache and provenance

Evaluation fingerprints use SHA-256 over canonical JSON. They include dataset
hash and selection, ordered question IDs, document IDs/content, labels and gold
records, chunking, exact model revisions, retrieval configuration, reader and
generation settings, prompt/source hashes, package versions, worker count, and
platform. Every cache row must match run, config, and item fingerprints.

Each run writes `run_manifest.json`, index/cache statistics, runtime/resource
usage, all gold answers, and retrieved/cited chunk provenance. Non-empty output
directories are never overwritten implicitly.

### Metric correctness

An additional empty-span bug was found during the clean rerun: empty raw answers
fell back to the abstention message. This made otherwise identical dense configs
A and B differ. Empty raw spans are now preserved explicitly; A and B produce
identical raw predictions.

The threshold sweep uses the raw span if the field exists, including a valid
empty string. Existing sweep artifacts require explicit `--overwrite` before
replacement.

## Verification

### Automated tests

`python -m pytest tests/ -q` → **35 passed**.

Regression coverage includes duplicate rejection, document replacement,
persistence/reload idempotence, unrelated-document preservation, canonical
fingerprinting, relevant-input invalidation, cross-run cache isolation,
malformed/mixed cache rejection, empty raw spans, document-disjoint threshold
splits, and record-only metric reproduction.

### Final smoke test

Two identical config-E runs over 8 questions and 5 documents produced:

- 355 chunks, 355 unique IDs, 0 duplicates on both runs;
- 8 misses on the first run and 8 hits on the repeat;
- byte-identical cached records;
- no duplicate retrieved chunk IDs;
- identical run fingerprints and intended question IDs.

Artifacts: `results/milestone2/smoke-final-fresh/` and
`results/milestone2/smoke-final-repeat/`.

## Clean evaluation results

| Run | Fingerprint | Documents | Questions (A/U) | Chunks/unique | Errors | Time | Peak RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| Natural A–E | `dab15af5d3ab...` | 40 | 127 (122/5) | 2,987/2,987 | 0 | 430.8 s | 1,681 MiB |
| Negative-enriched E | `5b6f98b8586c...` | 51 | 231 (171/60) | 3,784/3,784 | 0 | 297.1 s | 1,672 MiB |

Every clean summary metric matches recomputation from its JSONL records after
four-decimal rounding. Sweeps also match direct record recomputation. The final
integrity report additionally confirms that reranking configs returned reranked
hits: `results/milestone2/integrity_verification.json`.

### Natural retrieval ablation

| Config | F1 | EM | Cite-P | Cite-R | Ret-R |
|---|---:|---:|---:|---:|---:|
| A: dense | 0.0604 | 0.0000 | 0.0787 | 0.1140 | 0.2267 |
| B: dense+abstain | 0.0604 | 0.0000 | 0.0787 | 0.1140 | 0.2267 |
| C: hybrid | 0.0719 | 0.0082 | 0.0945 | 0.1326 | 0.2667 |
| D: hybrid+rerank | 0.0695 | 0.0164 | 0.1181 | 0.1427 | 0.2446 |
| E: full | 0.0695 | 0.0164 | 0.1181 | 0.1427 | 0.2446 |

Hybrid improves retrieval recall over dense by +0.0400 (paired bootstrap
`p < 0.0001`). Reranking reduces it by -0.0221 (`p = 0.0014`). Unlike the
historical contaminated run, reranking also slightly reduces answer F1
(`0.0719 -> 0.0695`), but not significantly (`p = 0.8371`).

### Historical-to-clean changes

| Quantity | Historical | Clean |
|---|---:|---:|
| Natural stored / unique chunks | 7,391 / 2,987 | 2,987 / 2,987 |
| Natural AUC | 0.6959 | 0.6459 |
| Natural full-coverage conditional F1 | 0.0507 | 0.0452 |
| Natural best in-sample conditional F1 | 0.0827 | 0.0791 |
| Negative-enriched stored / unique chunks | 4,645 / 3,784 | 3,784 / 3,784 |
| Negative-enriched AUC | 0.6172 | 0.6181 |
| Negative-enriched peak balanced accuracy | 0.5662 | 0.5662 |

The historical 0.019 baseline and 4.4x gain remain invalid. They are preserved
only in the audit trail. Full comparison:
`results/milestone2/comparison-final/comparison.md`.

## Locked threshold protocol

Documents are deterministically split with seed `20261028`; 60% are used for
threshold selection and 40% are held out. The predeclared objective is
calibration-set balanced accuracy over thresholds 0.00–1.00 in 0.05 steps.
Test labels are not consulted when selecting the threshold. A repeated
five-fold grouped-CV diagnostic is also saved; its split percentiles are not
presented as confidence intervals.

### Negative-enriched analysis (primary abstention stress test)

- Calibration: 31 documents, 141 questions (101 A / 40 U).
- Held-out test: 20 documents, 90 questions (70 A / 20 U).
- Selected threshold: **0.25**.
- Held-out AUC: **0.6443**, document-bootstrap 95% interval **[0.5322, 0.7467]**.
- Locked-threshold balanced accuracy: **0.5179**
  (**[0.4289, 0.5938]**), versus 0.5000 for always answer.
- Coverage: **0.1778**; 13 answerable and 3 unanswerable questions answered.
- Conditional first-gold F1: 0.0115 versus 0.0107 for always answer.
- Covered F1, which penalizes answered unanswerable questions: 0.0094 versus
  0.0083 for always answer.

There is no resolved held-out improvement over chance balanced accuracy.

### Natural sample

The held-out natural partition has only two unanswerable questions. Its locked
threshold is 0.05, balanced accuracy is 0.4069, and AUC is 0.5686 with a very
wide document-bootstrap interval [0.2572, 0.9118]. It is not a reliable
abstention estimate and is retained only as a natural-prevalence diagnostic.

## Preserved artifacts and limitations

- Original `results/`, `results/balanced/`, legacy caches, and `data/store` are
  unchanged. Their checksums are in
  `results/milestone2/historical_artifacts.json`.
- Historical best-over-gold F1/EM cannot be reconstructed from historical
  JSONL alone because those rows saved only the first gold answer.
- Historical total runtime and peak memory were not recorded and were not
  estimated.
- The negative-enriched sample is 74% answerable, not class-balanced.
- QASPER annotation disagreement and single-domain limitations remain.
- The confidence is a thresholded model score, not a calibrated probability.

## Recommendation

The index/cache pipeline now meets the integrity conditions for a **controlled
full-QASPER evaluation** using new output directories and locked manifests. Do
not begin CUAD or claim useful abstention performance yet. Scaling is justified
to narrow uncertainty and test retrieval findings, not to promote the current
abstention score: held-out balanced accuracy remains statistically compatible
with chance.
