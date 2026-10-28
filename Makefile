.PHONY: install test serve eval sweep data clean

install:
	python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

test:
	./.venv/bin/python -m pytest tests/ -q

serve:
	./.venv/bin/uvicorn app:app --reload

data:
	mkdir -p data/raw
	curl -sL "https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/validation/0.parquet" \
	  -o data/raw/qasper_validation.parquet

eval: data
	./.venv/bin/python -m eval.run_eval --dataset-partition validation --papers 40 --workers 2

sweep:
	./.venv/bin/python -m eval.sweep --config E

clean:
	rm -rf results/cache .pytest_cache
