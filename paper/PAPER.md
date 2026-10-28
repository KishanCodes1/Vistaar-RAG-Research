# Vistaar: Citation Grounding and Thresholded Abstention in Retrieval-Augmented QA

**Status:** draft v1.2 — integrity rerun completed 2026-10-28. All current
numbers come from fresh, run-isolated indices with fingerprinted caches. Earlier
Milestone 1 artifacts are preserved and compared in
`docs/ABSTENTION_AUDIT.md` and `docs/MILESTONE2_REPORT.md`.

---

## Abstract

Retrieval-augmented generation (RAG) is routinely described as *grounded*
because retrieved text is placed in the prompt. This is a claim about input, not
about output. We argue that grounding must be measured as three separable
properties — was the evidence retrieved, was the evidence cited, and did the
system decline when the evidence was absent — and that reporting answer-text
overlap alone hides failures on all three.

We present **Vistaar**, a retriever–reader RAG system in which every answer is
tied to the passage its span came from, and in which abstention is a thresholded
decision on a confidence score rather than a prompt instruction. We evaluate
five retrieval configurations on the QASPER validation split (40 papers, 127
questions, 2,987 unique chunks) and, separately, config E on a negative-enriched
slice of the full split (51 documents, 231 questions, 60 unanswerable, 3,784
unique chunks) so that abstention is not estimated from 5 negatives.

Three results stand out. First, hybrid dense+lexical retrieval significantly
improves gold-evidence recall over dense-only retrieval (0.227 → 0.267,
paired-bootstrap p < 0.0001). Second, adding a general-domain cross-encoder
reranker significantly degrades evidence recall (0.267 → 0.245, p = 0.0014)
and does not improve answer F1. Third, abstention remains weak under held-out
evaluation. A threshold selected on 31 calibration documents reaches balanced
accuracy 0.518 [0.429, 0.594] at 17.8% coverage on 20 held-out documents; its
interval includes chance. Test AUC is 0.644 [0.532, 0.747]. The natural sample
has only five negatives and is retained as a diagnostic, not a headline result.

We release the harness, pinned environment, per-question predictions and the
abstention sweep so that every number here is reproducible.

---

## 1. Introduction

Two failure modes survive a grounding prompt, and neither is visible in a
token-F1 score:

1. **Unsupported answers.** The model produces an answer the retrieved passages
   do not entail.
2. **Failure to abstain.** The document does not contain the answer, but the
   system answers anyway. In most deployments this is the more damaging failure,
   because a wrong answer and a confident abstention are consumed very
   differently.

Both are measurement problems before they are modelling problems. A system that
answers every question — including unanswerable ones — can score competitively
on F1 alone, because F1 never asks whether answering was the right decision.

**Contributions.**

1. An evaluation harness that measures four properties jointly: answer quality,
   gold-evidence recall, citation precision/recall, and abstention behaviour,
   each with bootstrap confidence intervals and paired significance tests.
2. A confidence formulation based on the reader's explicit no-answer
   hypothesis, making abstention an explicit, sweepable threshold rather than a
   prompt instruction. We measure discrimination, not probability calibration.
3. A document-disjoint calibration/test protocol on both a natural-prevalence
   and a negative-enriched sample, preventing test-set threshold selection.
4. **A negative result:** a general-domain cross-encoder reranker reduces
   gold-evidence recall on scientific documents, significant at p = 0.0014.

We do not claim a new model. The contribution is measurement, threshold
selection, and one reproducible negative finding.

---

## 2. Related work

**Retriever–reader RAG.** Lewis et al. (2020) established dense retrieval plus a
span reader. Our reader is a SQuAD2-trained encoder (Rajpurkar et al., 2018),
whose no-answer class supplies a native confidence signal.

**Hybrid retrieval.** Sparse and dense retrieval fail differently: BM25 preserves
rare identifiers and numbers that embeddings smooth away; dense retrieval handles
paraphrase. Reciprocal rank fusion (Cormack et al., 2009) combines ranked lists
without requiring comparable score scales.

**Reranking.** Cross-encoders (Nogueira & Cho, 2019) re-score candidates with
full query–document attention and are reported to improve top-k precision. Our
result (§5.2) qualifies this: the improvement is domain-dependent.

**Selective prediction.** El-Yaniv & Wiener (2010) formalised risk–coverage
trade-offs. We apply selective prediction to RAG and report the entire curve
rather than a single operating point.

**Attribution.** Rashkin et al. (2021) and Bohnet et al. (2022) establish that
citation *presence* is not citation *correctness*; we score the latter against
gold evidence spans.

**Benchmark.** QASPER (Dasigi et al., 2021) provides questions over NLP papers
with evidence annotations and explicit `unanswerable` labels — the property that
makes abstention measurable without new annotation.

---

## 3. System

### 3.1 Ingestion and chunking

Documents are extracted page by page (PyPDF2 for PDF, UTF-8 for text) and split
on the strongest available boundary — paragraph, line, sentence, word — then
packed into 800-character windows with 150 characters of overlap; fragments under
40 characters are dropped. Every chunk retains `doc_id`, `chunk_id`, page number,
section label and ordinal. This provenance is what makes citation verifiable.

### 3.2 Retrieval

`all-MiniLM-L6-v2` embeddings are L2-normalised, so inner product equals cosine
similarity and the index is `IndexFlatIP`. Lexical retrieval uses BM25 over the
same chunks. The two ranked lists are fused by reciprocal rank fusion (k = 60).
An optional `ms-marco-MiniLM-L-6` cross-encoder reranks the fused candidates.

Each namespace owns an independent index and lock, so documents are never shared
across sessions and concurrent ingestion cannot corrupt a search.

### 3.3 Reader and confidence

The reader is `deepset/roberta-base-squad2` on CPU. Question and context are
tokenised jointly with a sliding window (384 tokens, stride 128). For each window
we select the highest-scoring span and compare it to the SQuAD2 no-answer
candidate at position 0:

```
margin     = (start_logit[i] + end_logit[j]) - (start_logit[0] + end_logit[0])
confidence = sigmoid(margin)
```

We originally used a joint softmax ratio, but the distribution over hundreds of
candidate positions is spread so thin that every span appeared improbable and the
reader abstained on 76% of answerable questions. The logit margin is better
scaled: confidence = 0.5 marks equal span and null logits. It is not evidence
that the resulting sigmoid score is a calibrated probability.

The span's character offsets identify the source passage, producing a citation
with exact provenance. Raw confidence is recorded for every question; a
threshold is selected on calibration documents and locked before test scoring.

---

## 4. Experimental setup

**Data.** QASPER validation split; the first 40 papers in file order and all of
their questions → **127 questions (122 answerable, 5 unanswerable)**, yielding
**2,987 unique chunks**. No questions were dropped.

**Negative-enriched abstention slice.** Because 5 negatives are too few to estimate
AUC, we additionally evaluate config E on every document of the full 281-paper
validation split that contains at least one unanswerable question: **51
documents, 231 questions (171 answerable, 60 unanswerable), 3,784 unique chunks**. Only
config E is run on this slice, and it is used only for §5.4; Table 1 remains the
natural-prevalence ablation. This label-conditioned sample is 74% answerable; it
is a stress test, not a class-balanced or deployment-prevalence sample. Runs use
separate fingerprinted caches, so no predictions are shared across analyses.

> The 4% unanswerable rate is a property of QASPER's multi-annotator
> aggregation (a question counts as unanswerable only if every annotator said
> so). It makes plain abstention accuracy misleading, so we report balanced
> accuracy and the full risk–coverage curve instead.

**Configurations.**

| Config | Dense | BM25 | Rerank | Abstention |
|---|---|---|---|---|
| A | ✓ | – | – | – |
| B | ✓ | – | – | ✓ |
| C | ✓ | ✓ | – | – |
| D | ✓ | ✓ | ✓ | – |
| E | ✓ | ✓ | ✓ | ✓ |

The reader always receives 4 passages; configurations differ *which* 4 are
chosen. Retrieval metrics are therefore computed at depth 4 for all configs.

**Metrics.** Exact match and token F1 (SQuAD normalisation); gold-evidence recall
as the fraction of evidence vocabulary covered by the retrieved passages; citation
precision as the fraction of cited passages drawn ≥50% from the gold evidence;
citation recall as evidence vocabulary covered by cited passages; abstention
accuracy, balanced accuracy, confidence AUC, and latency.

Selective F1 in the sweeps is computed against the first gold answer, while
Table 1's F1 is the best score over all gold answers. Absolute values are not
directly comparable across the two; only within-sweep trends are.

**Evaluation integrity.** Every run uses a run-scoped persisted index and cache.
The manifest fingerprints the dataset and selected questions, document content,
chunking, exact model revisions, retrieval/reader settings, prompt/source code,
packages and platform. The evaluator verifies stored and unique chunk counts.

**Threshold protocol.** Documents are deterministically partitioned (seed
20261028): 60% calibration and 40% held-out test. Threshold 0.00–1.00 in 0.05
steps is selected by calibration balanced accuracy, tie-breaking toward
coverage. The test set is scored once at the locked threshold. We also report a
repeated five-fold document-grouped CV diagnostic.

**Statistics.** Answer-F1 intervals use 1,000 percentile-bootstrap resamples;
held-out abstention intervals use 2,000 document-cluster resamples. Paired
bootstrap tests centre differences under the null.

---

## 5. Results

**Table 1.** Ablation over retrieval configurations (n = 127, 122 answerable).

| Config | F1 [95% CI] | EM | Cite-P | Cite-R | Ret-R | Abst-Acc | Over-abst | Lat (s) |
|---|---|---|---|---|---|---|---|---|
| A: dense | 0.0604 [0.038, 0.087] | 0.0000 | 0.079 | 0.114 | 0.227 | 0.961 | 0.000 | 0.64 |
| B: dense+abstain | 0.0604 [0.038, 0.087] | 0.0000 | 0.079 | 0.114 | 0.227 | 0.158 | 0.877 | 0.74 |
| C: hybrid | 0.0719 [0.045, 0.103] | 0.0082 | 0.095 | 0.133 | 0.267 | 0.961 | 0.000 | 0.98 |
| D: hybrid+rerank | 0.0695 [0.043, 0.101] | 0.0164 | 0.118 | 0.143 | 0.245 | 0.961 | 0.000 | 1.16 |
| E: full | 0.0695 [0.043, 0.101] | 0.0164 | 0.118 | 0.143 | 0.245 | 0.173 | 0.861 | 1.15 |

### 5.1 Retrieval

Hybrid retrieval improves gold-evidence recall from 0.227 (dense) to 0.267, a
relative gain of 18%. A paired bootstrap test confirms the difference
(**p < 0.0001**, n = 127). This is the largest and most reliable effect in the
study: lexical matching recovers formulae, dataset names and numeric results that
the 384-dimensional embedding smooths away.

### 5.2 Reranking: a negative result

Adding the `ms-marco-MiniLM-L-6` cross-encoder **reduces** evidence recall from
0.267 to 0.245 (**p = 0.0014**) while slightly reducing answer F1 from 0.0719
to 0.0695 (not significant, p = 0.8371). The evidence-recall loss is therefore
not offset by a measured answer-F1 benefit.

We attribute this to domain mismatch: MS MARCO consists of short web passages,
whereas QASPER passages are long, reference-dense scientific paragraphs with
placeholder citation tokens (`BIBREF19`). A reranker trained on the former
systematically down-ranks the latter.

**Practically:** do not adopt a reranker without evaluating it on your domain.
The commonly assumed ordering — dense < hybrid < hybrid+rerank — did not hold
here.

### 5.3 Citation behaviour

Citation precision rises from 0.079 to 0.118 and citation recall from 0.114 to
0.143. Citation precision is systematically far below answer F1's apparent
quality: even the best
configuration draws only 12% of its cited passages primarily from gold evidence.
This is the metric that exposes "grounded" systems which cite plausible-looking
but wrong passages.

### 5.4 Abstention and risk–coverage

We report two abstention analyses. The natural-prevalence run (n = 127, 5
unanswerable) is comparable to Table 1 but cannot support stable inference. The
negative-enriched slice (n = 231, 60 unanswerable) is the primary stress test.
Full-sample sweeps are descriptive only; performance claims use a threshold
selected on calibration documents and evaluated on held-out documents.

#### 5.4.1 Primary analysis: document-disjoint negative-enriched slice

The deterministic split assigns 31 documents (141 questions: 101 answerable,
40 unanswerable) to calibration and 20 documents (90 questions: 70 answerable,
20 unanswerable) to held-out test. Maximising calibration balanced accuracy
selects threshold **0.25**.

**Table 2.** Held-out test results at the locked threshold. Intervals are
document-cluster bootstrap intervals at fixed threshold.

| Quantity | Locked threshold | Always answer |
|---|---:|---:|
| Coverage | 0.1778 [0.1011, 0.2651] | 1.0000 |
| Balanced accuracy | 0.5179 [0.4289, 0.5938] | 0.5000 |
| Conditional answerable F1 | 0.0115 [0.0000, 0.0346] | 0.0107 |
| Covered F1 | 0.0094 [0.0000, 0.0300] | 0.0083 |
| Answered unanswerable | 3 / 20 | 20 / 20 |

Held-out confidence AUC is **0.6443 [0.5322, 0.7467]**. The locked-threshold
balanced-accuracy interval includes chance, and the answer-quality differences
from always answering are negligible. We therefore do not claim useful
abstention performance.

For continuity with prior work, the full 231-question descriptive sweep has AUC
0.6181. Its in-sample peak balanced accuracy remains 0.5662 at threshold 0.10
(coverage 0.2814; 11 unanswerable answered). Maximising conditional F1 on those
same records yields 0.0569 at threshold 0.40 and 15.2% coverage; it is not an
unbiased test estimate.

**Table 3.** Descriptive alternative-signal AUCs on all 231 questions.

| Signal | AUC |
|---|---|
| Reader confidence (sigmoid logit margin) | **0.6181** |
| Retrieval score alone | 0.5830 |
| Confidence + z-scored retrieval score | 0.5827 |

Neither retrieval nor this simple combination beats confidence alone. The
reader's no-answer head is therefore doing work the retriever's score does not
capture, though the absolute level of separation remains weak. These data are
descriptive because signal selection on the same sample would require another
held-out evaluation.

#### 5.4.2 Natural-prevalence diagnostic

The clean full-sample AUC is **0.6459**, down from the contaminated historical
estimate of 0.6959. The in-sample peak balanced accuracy is 0.6352 at threshold
0.10. These values still rest on five negatives and are not final-test claims.

The document split places three negatives in calibration and only two in test.
Calibration selects threshold 0.05. On 53 held-out questions, coverage is
0.3208, balanced accuracy is 0.4069, and AUC is 0.5686 [0.2572, 0.9118]. The
interval is too broad to support an inference. The earlier selective-F1 baseline
of 0.019 and 4.4× gain were produced by retaining the default abstention message
when simulating lower thresholds; they are invalid and not retained.

The default threshold 0.5 still over-abstains: 86.1% of answerable natural
questions and 88.3% of answerable negative-enriched questions are declined.

> **Measurement note.** Conditional F1 in the abstention analysis is computed against the
> first gold answer for each question, whereas Table 1's F1 takes the best score
> over all gold answers. "Covered F1" additionally assigns zero to every
> answered-unanswerable question; it is the more complete selective utility
> measure. Absolute values are not directly comparable to Table 1.

---

## 6. Discussion

**F1 is the wrong headline metric.** F1 values of 0.06–0.07 look poor,
but they largely reflect a metric mismatch: QASPER gold answers are abstractive
("multilingual NMT (MNMT) BIBREF19") while the reader is extractive and returns
the shorter span ("multilingual NMT approaches"). Publishing F1 alone would
hide retrieval and citation failures, but weak held-out abstention means those
metrics cannot be replaced by a selective-F1 improvement claim.

**Retrieval and reader quality are confounded in end-to-end F1.** Config D
degrades evidence recall significantly while its small F1 difference is not
significant. Separating recall from answer overlap exposes the failure directly.

**Threshold selection is a first-class design problem.** Our first confidence
formula (joint softmax) was mathematically reasonable and practically useless:
it drove 76% over-abstention. The logit margin is better, but it is not
calibrated. The calibration-selected threshold reaches only 0.518 held-out
balanced accuracy, with an interval spanning chance. Anything that gates whether
a user gets an answer must be selected away from the test set and reported with
coverage, answered-unanswerable errors and uncertainty.

---

## 7. Limitations

- **Single domain, single language.** English NLP papers. No legal/contract
  domain, so no cross-domain transfer claim is made.
- **Abstention is weak.** On the negative-enriched held-out set, balanced
  accuracy is 0.518 [0.429, 0.594]. The natural held-out set has only two
  negatives. We do not claim probabilistic calibration.
- **Negative enrichment changes prevalence.** The stress slice is 74%
  answerable and must not be read as deployment coverage or accuracy.
- **Annotation ambiguity.** Fourteen stress-slice questions and thirteen
  natural-sample questions have mixed answerable/unanswerable annotations. We
  use the declared rule "answerable if any annotator supplied an answer."
- **Small held-out partitions.** Document grouping prevents leakage but leaves
  only 20 stress-test and 16 natural test documents. Cross-validation is
  reported as a stability diagnostic, not statistical certainty.
- **Selective F1 is measured against the first gold answer.** The sweep scores
  against `gold_answers[0]`, whereas Table 1 uses best-over-gold F1. Absolute
  values across the two are not comparable. Conditional F1 also excludes
  answered-unanswerable errors, so covered F1 is reported alongside it.
- **Extractive reader.** Answers are spans and are often shorter than gold.
  Token F1 understates quality; we report it rather than tuning the reader to
  the metric.
- **Vocabulary-overlap citations.** Citation correctness uses lexical overlap,
  which is strict and cannot credit correct paraphrase.
- **No human evaluation.**
- **Reranker negative result** is measured on one reranker and one domain; it
  should not be generalised beyond that.

---

## 8. Reproducibility

```bash
pip install -r requirements.txt          # fully pinned
make data                                # QASPER validation parquet
python -m eval.run_eval --dataset-partition validation \
    --papers 40 --workers 2 --out <new-natural-dir>
python -m eval.sweep --records <new-natural-dir>/records.jsonl \
    --config E --out <new-natural-dir>
python -m eval.run_eval --dataset-partition validation \
    --papers 281 --only E --balanced --workers 2 \
    --out <new-enriched-dir>
python -m eval.threshold_protocol \
    --records <new-enriched-dir>/records.jsonl --config E \
    --out <new-enriched-dir>/threshold_protocol
```

Each output contains an immutable input manifest, complete records, index/cache
integrity counts and runtime/resource metadata. Exact model revisions are pinned;
CPU inference, temperature 0 and statistical seeds are fixed. The audited runs
are in `results/milestone2/natural-clean-final/` and
`results/milestone2/balanced-clean-final/`. Historical outputs are preserved and
checksummed. Unit tests: `python -m pytest tests/ -q` (**38 tests**).

---

## References

Bohnet et al. 2022. Attributed Question Answering. *arXiv:2212.08037*.
Cormack, Clarke & Büttcher. 2009. Reciprocal Rank Fusion Outperforms Condorcet
and Individual Rank Learning Methods. *SIGIR*.
Dasigi et al. 2021. A Dataset of Information-Seeking Questions and Answers
Anchored in Research Papers (QASPER). *NAACL*.
El-Yaniv & Wiener. 2010. On the Foundations of Noise-free Selective
Classification. *JMLR*.
Lewis et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP
Tasks. *NeurIPS*.
Nogueira & Cho. 2019. Passage Re-ranking with BERT. *arXiv:1901.04085*.
Rajpurkar, Jia & Liang. 2018. Know What You Don't Know: Unanswerable Questions
for SQuAD. *ACL*.
Rashkin et al. 2021. Measuring Attribution in Natural Language Generation
Models. *arXiv:2112.12870*.
