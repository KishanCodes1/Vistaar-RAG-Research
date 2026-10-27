# Milestone 3 artifact index

Authoritative run: `full-qasper-validation-ae-v1/`

- `run_manifest.json` — immutable run inputs, model/config/source fingerprints,
  per-document index counts and runtime metadata.
- `records.jsonl` — 5,015 source-traceable records (1,003 per config).
- `summary.json`, `table.md` — evaluator aggregate outputs.
- `sweep_E.*` — descriptive full-sample confidence sweep; not threshold selection.
- `threshold_protocol/` — predeclared calibration/test threshold result.
- `independent_integrity_preprotocol.json` — direct parquet/cache/index/provenance audit.
- `independent_metrics.json` — separately implemented metric and bootstrap audit.
- `analysis/analysis.*` — document-cluster intervals and comparison with Milestone 2.
- `analysis/failure_cases.json`, `failure_summary.json` — held-out failure analysis.
- `analysis/baseline_differences.json` — cluster-bootstrap baseline differences.
- `analysis/label_sensitivity.json` — annotation-rule sensitivity at locked threshold.

Milestone-level controls:

- `readiness.{json,md}` — preflight scope, cost and runtime estimate.
- `protocol_plan.json` — document split and threshold procedure fixed before run completion.
- `pre_run_preservation.json` — checksums of Milestone 2 artifacts before launch.
- `final_verification.json` — terminal completion decision and authoritative hashes.

No CUAD or second full-QASPER run was launched.
