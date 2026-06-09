.PHONY: install fmt lint test up down ingest eval

install:        ## install package + dev tooling
	pip install -e ".[dev]"

fmt:            ## format with black + ruff
	black src tests scripts
	ruff check --fix src tests scripts

lint:           ## lint without modifying
	ruff check src tests scripts

test:           ## run the test suite
	pytest

up:             ## start the full stack
	docker-compose up

down:           ## stop the stack
	docker-compose down

ingest:         ## ingest the corpus (TODO: Phase 1)
	python scripts/ingest.py

eval:           ## run the evaluation harness (TODO: Phase 3)
	python scripts/run_eval.py
