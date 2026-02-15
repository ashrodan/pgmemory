.PHONY: test unit integration pglite docker typecheck build clean release-patch release-minor release-major check-build docs docs-serve

test: unit integration  ## Run all tests (unit + integration against local Postgres)

unit:  ## Unit tests (no DB required)
	uv run pytest tests/test_unit.py -v

integration:  ## Integration tests against local Postgres (port 5432)
	uv run scripts/test.py

pglite:  ## Integration tests via ephemeral Postgres cluster (port 55433)
	uv run scripts/test_pglite.py

docker:  ## Build + run tests in Docker container
	uv run scripts/test_docker.py

typecheck:  ## Run ty type checker
	uvx ty check

build:  ## Build sdist + wheel
	uv build

clean:  ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info src/*.egg-info .mypy_cache .pytest_cache

release-patch:  ## Bump patch version, commit, and tag (e.g., 0.1.0 → 0.1.1)
	uv run bump-my-version bump patch

release-minor:  ## Bump minor version, commit, and tag (e.g., 0.1.0 → 0.2.0)
	uv run bump-my-version bump minor

release-major:  ## Bump major version, commit, and tag (e.g., 0.1.0 → 1.0.0)
	uv run bump-my-version bump major

check-build:  ## Build and validate package with twine
	uv build && uv run twine check dist/*

docs:  ## Build documentation site
	uv run --extra docs mkdocs build

docs-serve:  ## Serve docs locally with hot-reload
	uv run --extra docs mkdocs serve

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
