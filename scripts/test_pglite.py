#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Spin up an ephemeral Postgres cluster (pgvector), run integration tests, tear down.

No Docker. No testcontainers. Just your local Postgres installation.

Usage:
    uv run scripts/test_pglite.py
    uv run scripts/test_pglite.py --keep   # leave cluster running for inspection
"""
import argparse
import atexit
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def find_pg_bin() -> str:
    """Find the Postgres bin directory."""
    # Homebrew Apple Silicon
    for ver in ["17", "16", "15", "14"]:
        p = f"/opt/homebrew/opt/postgresql@{ver}/bin"
        if os.path.isdir(p):
            return p
    # Homebrew Intel
    for ver in ["17", "16", "15", "14"]:
        p = f"/usr/local/opt/postgresql@{ver}/bin"
        if os.path.isdir(p):
            return p
    # Linux
    for ver in ["17", "16", "15", "14"]:
        p = f"/usr/lib/postgresql/{ver}/bin"
        if os.path.isdir(p):
            return p
    # Fall back to PATH
    pg_ctl = shutil.which("pg_ctl")
    if pg_ctl:
        return os.path.dirname(pg_ctl)
    print("ERROR: Could not find PostgreSQL binaries", file=sys.stderr)
    sys.exit(1)


def check_pgvector(pg_bin: str, socket_dir: str, port: str) -> bool:
    """Check if pgvector extension is available."""
    result = subprocess.run(
        [os.path.join(pg_bin, "psql"), "-h", socket_dir, "-p", port,
         "-d", "postgres", "-tAc",
         "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() == "1"


PORT = "55433"
DB_NAME = "pgmemory_test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="Keep cluster after tests")
    args = parser.parse_args()

    pg_bin = find_pg_bin()
    print(f"Using Postgres from: {pg_bin}")

    data_dir = tempfile.mkdtemp(prefix="pgmemory_test_")
    socket_dir = data_dir
    print(f"Data dir: {data_dir}")

    def cleanup():
        print("\nStopping cluster...")
        subprocess.run(
            [os.path.join(pg_bin, "pg_ctl"), "stop", "-D", data_dir, "-m", "fast"],
            capture_output=True,
        )
        if not args.keep:
            shutil.rmtree(data_dir, ignore_errors=True)
            print("Cleaned up.")
        else:
            print(f"Cluster left at: {data_dir} (port {PORT})")

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda *_: sys.exit(1))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))

    # 1. Init cluster
    print("\n── initdb ──")
    subprocess.run(
        [os.path.join(pg_bin, "initdb"), "-D", data_dir, "-E", "UTF8", "--no-locale"],
        check=True, capture_output=True,
    )

    # 2. Start on ephemeral port with unix socket in data dir
    print(f"── starting on port {PORT} ──")
    subprocess.run(
        [os.path.join(pg_bin, "pg_ctl"), "start", "-D", data_dir, "-w",
         "-l", os.path.join(data_dir, "server.log"),
         "-o", f"-p {PORT} -k {socket_dir} -c shared_preload_libraries=''"],
        check=True, capture_output=True,
    )
    time.sleep(0.5)

    # 3. Check pgvector
    if not check_pgvector(pg_bin, socket_dir, PORT):
        print("ERROR: pgvector extension not available in this Postgres installation")
        print("Install it: brew install pgvector  (or apt install postgresql-17-pgvector)")
        sys.exit(1)
    print("pgvector: available")

    # 4. Create test DB + extension
    psql = os.path.join(pg_bin, "psql")
    subprocess.run(
        [psql, "-h", socket_dir, "-p", PORT, "-d", "postgres",
         "-c", f"CREATE DATABASE {DB_NAME};"],
        check=True, capture_output=True,
    )
    subprocess.run(
        [psql, "-h", socket_dir, "-p", PORT, "-d", DB_NAME,
         "-c", "CREATE EXTENSION IF NOT EXISTS vector;"],
        check=True, capture_output=True,
    )
    print(f"Database '{DB_NAME}' ready with pgvector\n")

    # 5. Run tests
    user = os.environ.get("USER", os.environ.get("LOGNAME", "postgres"))
    test_url = f"postgresql+asyncpg://{user}@localhost:{PORT}/{DB_NAME}"
    print(f"── integration tests ({test_url}) ──\n")

    result = subprocess.run(
        ["uv", "run", "pytest", "tests/test_integration.py", "-v"],
        env={**os.environ, "PGMEMORY_TEST_URL": test_url},
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
