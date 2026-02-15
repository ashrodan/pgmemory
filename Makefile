.PHONY: test unit integration pglite docker typecheck build clean

test: unit integration  ## Run all tests (unit + integration against local Postgres)

unit:  ## Unit tests (no DB required)
	uv run pytest tests/test_unit.py -v

integration:  ## Integration tests against local Postgres (port 5432)
	uv run scripts/test.py

pglite:  ## Integration tests via ephemeral Postgres cluster (port 55433)
	uv run scripts/test_pglite.py

docker:  ## Build + run tests in Docker container
	uv run scripts/test_docker.py

typecheck:  ## Run mypy
	uv run mypy src/pgmemory

build:  ## Build sdist + wheel
	uv build

clean:  ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info src/*.egg-info .mypy_cache .pytest_cache

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
