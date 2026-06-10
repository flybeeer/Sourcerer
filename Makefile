.PHONY: install fmt lint test up down ingest eval route-report graphrag-index graphrag-eval

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

route-report:   ## report local-vs-API routing split + cost savings
	python scripts/route_report.py --simulate

graphrag-index: ## build the GraphRAG index (expensive — prompts to confirm)
	python scripts/graphrag_index.py

graphrag-eval:  ## compare GraphRAG global vs hybrid on overview questions
	python scripts/graphrag_eval.py
