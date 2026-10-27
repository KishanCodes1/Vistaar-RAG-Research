# Document-grouped threshold protocol — E

Objective: `balanced_accuracy` on calibration documents only.
Locked threshold: **0.10**.

| Partition | Documents | Questions | Answerable | Unanswerable |
|---|---:|---:|---:|---:|
| Calibration | 169 | 600 | 560 | 40 |
| Held-out test | 112 | 403 | 383 | 20 |

| Held-out metric | Locked threshold | Always answer |
|---|---:|---:|
| Coverage | 0.2804 | 1.0000 |
| Balanced accuracy | 0.5686 | 0.5000 |
| Conditional answerable F1 | 0.0463 | 0.0401 |
| Covered F1 | 0.0451 | 0.0381 |
| Answered unanswerable | 3 | 20 |
| AUC | 0.6636 | — |

Cluster-bootstrap 95% intervals (fixed locked threshold):
balanced accuracy [0.4773, 0.6441], coverage [0.2422, 0.3192], AUC [0.5286, 0.7912].

Cross-validation percentiles describe split sensitivity, not a confidence interval.
