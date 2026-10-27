# Full-QASPER record analysis

Run fingerprint: `77f3ef35739c9619e11aea426a76264332a3b53960c430c85c90ad6ba609dfa1`
Dataset: 281 papers, 1003 questions (943 answerable / 60 unanswerable).
Index: 21516 stored / 21516 unique chunks.

## Aggregate metrics with document-cluster 95% intervals

| Config | F1 | EM | Cite-P | Cite-R | Ret-R |
|---|---:|---:|---:|---:|---:|
| A: dense | 0.0562 [0.0472, 0.0651] | 0.0032 [0.0000, 0.0073] | 0.0409 [0.0276, 0.0557] | 0.1195 [0.1114, 0.1272] | 0.2283 [0.2180, 0.2388] |
| B: dense+abstain | 0.0562 [0.0472, 0.0651] | 0.0032 [0.0000, 0.0073] | 0.0409 [0.0276, 0.0557] | 0.1195 [0.1114, 0.1272] | 0.2283 [0.2180, 0.2388] |
| C: hybrid | 0.0624 [0.0530, 0.0728] | 0.0064 [0.0021, 0.0117] | 0.0568 [0.0418, 0.0726] | 0.1409 [0.1326, 0.1495] | 0.2671 [0.2567, 0.2773] |
| D: hybrid+rerank | 0.0625 [0.0523, 0.0729] | 0.0095 [0.0040, 0.0159] | 0.0608 [0.0451, 0.0770] | 0.1352 [0.1270, 0.1435] | 0.2392 [0.2291, 0.2496] |
| E: full (hybrid+rerank+abstain) | 0.0625 [0.0523, 0.0729] | 0.0095 [0.0040, 0.0159] | 0.0608 [0.0451, 0.0770] | 0.1352 [0.1270, 0.1435] | 0.2392 [0.2291, 0.2496] |

A separate expansion subset excludes the 40 papers in the clean Milestone 2 natural run: 241 papers, 876 questions per config (821 answerable / 55 unanswerable).

## Locked held-out abstention

Threshold `0.10` was selected only on 169 calibration documents and applied to 112 held-out documents.

| Metric | Locked threshold | Always answer |
|---|---:|---:|
| AUC | 0.6636 | — |
| Balanced accuracy | 0.5686 | 0.5000 |
| Coverage | 0.2804 | 1.0000 |
| Conditional answerable F1 | 0.0463 | 0.0401 |
| Covered F1 | 0.0451 | 0.0381 |
| Answered unanswerable | 3 | 20 |

## Integrity and continuity

All checks passed: **True**. The clean 40-paper overlap has 635/635 scientifically identical records.

## Failure counts on held-out documents

Answered unanswerable: 3; abstained answerable: 273.
