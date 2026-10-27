# Full-QASPER Evaluation

**Milestone:** 3  
**Dataset:** QASPER validation  
**Status:** complete and independently verified  
**Output:** `results/milestone3/full-qasper-validation-ae-v1/`

## 1. Objective

Evaluate all five Vistaar retrieval/abstention configurations over every
evaluable question in the 281-paper QASPER validation split using the
run-isolated Milestone 2 infrastructure. The run tests whether the 40-paper
retrieval findings persist at scale and estimates abstention at a threshold
selected without consulting this milestone's held-out papers.

## 2. Readiness and cost

Preflight evidence is stored in `results/milestone3/readiness.{json,md}`.
Thirty-five preflight tests passed. No compatible full-run cache existed.
Expected work was 5,015 question/config records, approximately 60–75 minutes on
local CPU, about 2 GiB peak RAM, and USD 0 in API charges. The local reader path
does not call Gemini.

## 3. Dataset

- Identity: `allenai/qasper`, validation parquet.
- File SHA-256:
  `089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222`.
- Papers: 281.
- Raw questions: 1,005.
- Evaluable questions: 1,003 (943 answerable / 60 unanswerable).
- Dropped: 2 answerable questions without usable answer text.
- Expected fresh chunks: 21,516 at 800 characters with 150 overlap.

## 4. Models and retrieval configurations

Inference uses local CPU models at pinned revisions:

- Embedding: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, normalized embeddings.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- Reader: `deepset/roberta-base-squad2`, revision
  `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`.

Configurations A–E are dense; dense+abstain; dense+BM25; hybrid+rerank; and
hybrid+rerank+abstain. `top_k=8`; the reader receives four passages. The
configured default threshold 0.5 is measured but is not selected as the final
operating point.

## 5. Predeclared threshold protocol

`results/milestone3/protocol_plan.json` was written before inference completed.
Documents are split by deterministic hash with seed 20261028:

- Calibration: 169 papers, 600 questions (560 answerable / 40 unanswerable).
- Held-out test: 112 papers, 403 questions (383 / 20).

Thresholds 0.00–1.00 in 0.05 steps are ranked by calibration balanced accuracy,
tie-breaking toward coverage. The selected threshold is then locked for test.
Questions from one paper cannot cross partitions. A 2,000-resample
document-cluster bootstrap and repeated five-fold grouped-CV diagnostic are
planned.

This is held out from threshold selection in Milestone 3, but it is not a
globally untouched external test: QASPER validation and all negative-containing
documents were inspected during earlier milestones. Claims will be qualified
accordingly.

## 6. Planned verification and analysis

After inference, the evaluator must confirm:

1. 5,015 records, zero run errors, and one unique question ID per config.
2. Stored chunks equal unique chunks and retrieved lists contain no duplicate IDs.
3. Cache metadata, code hashes, model revisions and chunk provenance match the
   saved run manifest.
4. Summaries and sweeps reproduce from records after documented rounding.
5. Calibration and test documents exactly match the predeclared plan.
6. The 40-paper overlap reproduces the clean Milestone 2 scientific fields.

Aggregate metrics will receive document-cluster intervals. Failure analysis will
cover answered-unanswerable questions, potentially useful answers rejected by
the locked threshold, and low-evidence retrieval cases.

## 7. Results

### 7.1 Completion and resources

The run completed normally with fingerprint
`77f3ef35739c9619e11aea426a76264332a3b53960c430c85c90ad6ba609dfa1`.
It produced all **5,015 expected records** (1,003 per configuration), with zero
inference errors. Indexing took 198.4 seconds; evaluation took 3,024.0 seconds;
total wall time was 3,222.5 seconds (53.7 minutes). Peak RSS was 1,874 MiB.
No hosted API was called.

The isolated index contains **21,516 stored / 21,516 unique chunks** across 281
namespaces. All five cache files contain exactly 1,003 rows matching the final
records.

### 7.2 Aggregate metrics

Intervals below are independent 2,000-resample document-cluster percentile
intervals. F1/EM denominators are the 943 answerable questions; citation and
retrieval metrics use all 1,003 questions.

| Config | F1 | EM | Cite-P | Cite-R | Ret-R |
|---|---:|---:|---:|---:|---:|
| A: dense | 0.0562 [0.0472, 0.0651] | 0.0032 [0.0000, 0.0073] | 0.0409 [0.0276, 0.0557] | 0.1195 [0.1114, 0.1272] | 0.2283 [0.2180, 0.2388] |
| B: dense+abstain | 0.0562 [0.0472, 0.0651] | 0.0032 [0.0000, 0.0073] | 0.0409 [0.0276, 0.0557] | 0.1195 [0.1114, 0.1272] | 0.2283 [0.2180, 0.2388] |
| C: hybrid | 0.0624 [0.0530, 0.0728] | 0.0064 [0.0021, 0.0117] | 0.0568 [0.0418, 0.0726] | 0.1409 [0.1326, 0.1495] | **0.2671 [0.2567, 0.2773]** |
| D: hybrid+rerank | 0.0625 [0.0523, 0.0729] | 0.0095 [0.0040, 0.0159] | 0.0608 [0.0451, 0.0770] | 0.1352 [0.1270, 0.1435] | 0.2392 [0.2291, 0.2496] |
| E: full | 0.0625 [0.0523, 0.0729] | 0.0095 [0.0040, 0.0159] | 0.0608 [0.0451, 0.0770] | 0.1352 [0.1270, 0.1435] | 0.2392 [0.2291, 0.2496] |

Document-cluster paired differences support the retrieval findings:

- A→C retrieval recall: **+0.0388 [0.0332, 0.0448]**.
- C→D retrieval recall: **−0.0279 [−0.0349, −0.0212]**.
- A→E retrieval recall: +0.0108 [0.0040, 0.0178].
- A→C answer F1: +0.0062 [0.0001, 0.0129], a small effect with its lower
  bound near zero.
- C→D answer F1: +0.0001 [−0.0070, 0.0075], no resolved change.

The 241-paper expansion subset excluding the previously studied 40 papers gives
nearly identical retrieval values (A 0.2286, C 0.2672, D/E 0.2384), so the
retrieval result is not driven by the original sample.

### 7.3 Descriptive full-sample abstention

On all 1,003 questions, confidence AUC is 0.6227. Higher confidence correctly
points toward answerable: mean 0.1711 for answerable versus 0.1195 for
unanswerable questions. The in-sample balanced-accuracy maximum is 0.5594 at
threshold 0.10; this is descriptive and was not used as the held-out estimate.

### 7.4 Predeclared held-out threshold result

Calibration selected threshold **0.10** with balanced accuracy 0.5563 on 169
papers / 600 questions. The threshold was then locked for the 112-paper,
403-question test partition.

| Held-out metric | Locked 0.10 | Always answer | Default 0.50 |
|---|---:|---:|---:|
| AUC | 0.6636 [0.5286, 0.7912] | — | — |
| Balanced accuracy | 0.5686 [0.4773, 0.6441] | 0.5000 | 0.5166 |
| Coverage | 0.2804 [0.2422, 0.3192] | 1.0000 | 0.1315 |
| Conditional first-gold F1 | 0.0463 [0.0288, 0.0677] | 0.0401 | 0.0514 |
| Covered F1 | 0.0451 [0.0277, 0.0659] | 0.0381 | 0.0494 |
| Answered answerable | 110 / 383 | 383 / 383 | 51 / 383 |
| Answered unanswerable | 3 / 20 | 20 / 20 | 2 / 20 |
| Unnecessarily abstained answerable | 273 / 383 | 0 | 332 / 383 |

The apparent locked-threshold gains over always-answer are **not statistically
resolved** under document-cluster bootstrap:

- Balanced-accuracy difference: +0.0686, interval approximately
  **[−0.0214, 0.1444]**.
- Conditional-F1 difference: +0.0062 **[−0.0092, 0.0226]**.
- Covered-F1 difference: +0.0070 **[−0.0081, 0.0229]**.

The threshold sharply lowers answered-unanswerable errors but declines 71.3% of
answerable test questions. Plain abstention accuracy is only 0.315 because the
test prevalence is 95% answerable; always answering scores 0.950 on plain
accuracy. Balanced accuracy is therefore necessary but not sufficient for
deployment utility.

Repeated document-grouped five-fold CV yields mean balanced accuracy 0.5448 and
mean coverage 0.3192. Its split percentiles describe sensitivity, not a
confidence interval.

## 8. Independent integrity verification

Three verification artifacts agree:

- `independent_integrity_preprotocol.json`: direct parquet-label, cache-row,
  persisted-FAISS and chunk-provenance audit.
- `independent_metrics.json`: separately implemented normalization, F1/EM,
  evidence metrics, pairwise AUC, threshold selection, baselines and bootstrap.
- `analysis/verification.json`: manifest, record, sweep, protocol-plan and
  Milestone 2 continuity checks.

The terminal decision and authoritative artifact hashes are recorded in
`results/milestone3/final_verification.json`.

Verified facts:

- 5,015/5,015 parseable records ending in a complete newline; zero missing
  fields, invalid scores, label/document mismatches or cache mismatches.
- 281 persisted namespaces, 21,516 FAISS vectors/chunks, no duplicate chunk IDs.
- Every retrieved text and chunk ID maps through the manifest to its QASPER paper.
- Exact model/config/source hashes match the run manifest.
- Calibration and test paper sets are disjoint and exactly match the plan written
  before inference completed.
- Saved summaries, sweep, selected threshold and all 2,000 bootstrap intervals
  reproduce exactly from records.
- All 635 overlapping records from the clean 40-paper run are scientifically
  identical (excluding latency and cache metadata).
- Final test suite: **38 passed**.

No deviation from the predeclared protocol was found.

## 9. Failure analysis

At threshold 0.10 on held-out papers:

- **3/20 unanswerable questions were answered.** One false answer had confidence
  0.9936, showing that score magnitude is not calibrated correctness.
- **273/383 answerable questions were declined.** Six declined questions had a
  raw answer matching one of their gold answers at F1 1.0.
- 13 answerable questions had zero evidence recall; 51 had recall below 0.1.
- 313/383 answerable questions had zero first-gold token F1, reflecting both
  reader failure and extractive/abstractive mismatch.

Examples and retrieved-passage previews are saved in
`analysis/failure_cases.json`. They show plausible spans confidently extracted
for questions QASPER labels unanswerable, while several exact short answers are
rejected at very low confidence.

Label aggregation remains consequential. Seventy-five full-split questions have
mixed answerability annotations (28 in held-out test). At the locked threshold,
excluding mixed labels changes balanced accuracy only from 0.5686 to 0.5701,
but requiring every annotator to call a question answerable reduces AUC to
0.5808 and balanced accuracy to 0.5409. This sensitivity is not a retuned result.

## 10. Comparison with Milestone 2

The full run confirms the clean 40-paper retrieval pattern:

- Dense Ret-R: 0.2267 → 0.2283.
- Hybrid Ret-R: 0.2667 → 0.2671.
- Hybrid+rerank Ret-R: 0.2446 → 0.2392.

Full-run F1 is slightly lower for A/C and similar for D/E, but all absolute F1
values remain low. Full-sample confidence AUC (0.6227) is close to the
negative-enriched Milestone 2 value (0.6181), while the larger held-out split
still cannot establish a robust balanced-accuracy improvement over baseline.

## 11. Supported claims and limitations

**Supported:** hybrid dense+lexical retrieval improves evidence recall, and the
general-domain reranker materially reduces evidence recall on QASPER. Both
paired document-bootstrap intervals exclude zero and persist outside the
original 40-paper sample.

**Not supported:** production readiness, probabilistic calibration, or a robust
selective-answering utility gain. AUC indicates modest rank discrimination, but
the locked operating point's balanced-accuracy and F1 differences include zero.

Additional limitations:

- This is QASPER validation, not an untouched external dataset. Earlier
  milestones inspected the first 40 papers and all negative-containing papers.
- The held-out split is protected from threshold selection in this milestone,
  but not globally untouched from prior analysis.
- QASPER's natural unanswerable rate is only 6%, and annotation disagreement is
  substantial.
- Citation correctness uses lexical evidence overlap and the reader is
  extractive; neither substitutes for human evaluation.
- Runtime and latency are CPU/platform specific.

## 12. Decision and next milestone

**Milestone 3 is complete.** The run, records, index, caches, protocol and
independent calculations agree, and historical artifacts were not overwritten.

**Recommendation: GO for a scoped next milestone, but NO-GO for deployment or
strong abstention claims.** Preserve the retrieval finding as the primary result.
Next work should improve or replace the confidence/reader signal and evaluate it
on a genuinely external domain or untouched split before revisiting abstention
claims. CUAD may serve as that domain only with a predeclared, independently
fingerprinted protocol; it must not be used to tune and test the same threshold.
