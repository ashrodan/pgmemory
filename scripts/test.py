#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run unit + integration tests against a running local Postgres.

Usage:
    uv run scripts/test.py
    uv run scripts/test.py --unit-only
"""
import argparse
import getpass
import os
import shutil
import subprocess
import sys

DB_NAME = "pgmemory_test"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, **kwargs)


def psql(user: str, db: str, sql: str) -> subprocess.CompletedProcess:
    return run(["psql", "-U", user, "-d", db, "-c", sql], capture_output=True, text=True)


def ensure_db(user: str) -> str:
    """Create test DB + pgvector if needed. Returns connection URL."""
    result = run(
        ["psql", "-U", user, "-d", "postgres", "-tAc",
         f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'"],
        capture_output=True, text=True,
    )
    if result.stdout.strip() != "1":
        psql(user, "postgres", f"CREATE DATABASE {DB_NAME};")
        print(f"Created database '{DB_NAME}'")

    psql(user, DB_NAME, "CREATE EXTENSION IF NOT EXISTS vector;")
    return f"postgresql+asyncpg://{user}@localhost:5432/{DB_NAME}"


def main():
    parser = argparse.ArgumentParser(description="pgmemory test runner")
    parser.add_argument("--unit-only", action="store_true", help="Skip integration tests")
    args = parser.parse_args()

    if not shutil.which("psql") and not args.unit_only:
        print("ERROR: psql not found. Use --unit-only or install PostgreSQL.", file=sys.stderr)
        sys.exit(1)

    print("=== pgmemory test runner ===\n")

    # Unit tests
    print("── Unit tests ──")
    result = run(["uv", "run", "pytest", "tests/test_unit.py", "-v"])
    if result.returncode != 0:
        sys.exit(result.returncode)

    if args.unit_only:
        print("\n=== Unit tests passed ===")
        sys.exit(0)

    # Integration tests
    user = os.environ.get("PGUSER", getpass.getuser())
    test_url = ensure_db(user)
    print(f"\n── Integration tests ({test_url}) ──")
    result = run(
        ["uv", "run", "pytest", "tests/test_integration.py", "-v"],
        env={**os.environ, "PGMEMORY_TEST_URL": test_url},
    )
    if result.returncode != 0:
        sys.exit(result.returncode)

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
