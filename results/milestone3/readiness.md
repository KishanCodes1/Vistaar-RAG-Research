# Full-QASPER readiness report

**Decision:** PASS — local evaluation may proceed without paid API usage.

- Branch: `main`; Milestone 1 commit `e87fa0c` intact.
- Preflight tests: 35 passed.
- Dataset: QASPER validation, SHA-256
  `089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222`.
- Data: 281 papers; 1,005 raw questions; 1,003 evaluable questions
  (943 answerable / 60 unanswerable); 2 unusable-gold questions dropped.
- Expected fresh index: 21,516 unique chunks (800 characters, 150 overlap).
- Intended matrix: configurations A–E, 5,015 question/config records.
- No compatible 281-paper natural A–E cache existed before launch.
- Reader: local CPU `deepset/roberta-base-squad2` at pinned revision
  `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`.
- Embedding and reranker revisions are pinned and present in the local model
  cache. No Hugging Face token is required for this run.
- Gemini/API credentials are not required or used by the local-reader path.
- Runtime estimate: 60–75 minutes; peak RAM approximately 2 GiB.
- Storage estimate: 150–250 MiB across output, resumable records and index.
- API request estimate: 0; expected API cost: USD 0.
- New output: `results/milestone3/full-qasper-validation-ae-v1/`.

The threshold plan was fixed before inference completed in
`results/milestone3/protocol_plan.json`: seed 20261028, 60% calibration papers,
40% held-out papers, balanced-accuracy objective, thresholds 0.00–1.00 by 0.05.
