#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build and run pgmemory tests inside a Docker container (pgvector image).

Usage:
    uv run scripts/test_docker.py
    uv run scripts/test_docker.py --no-cache
"""
import argparse
import os
import shutil
import subprocess
import sys

IMAGE = "pgmemory-test"
DOCKERFILE = "Dockerfile.test"


def main():
    parser = argparse.ArgumentParser(description="pgmemory Docker test runner")
    parser.add_argument("--no-cache", action="store_true", help="Build without cache")
    args = parser.parse_args()

    if not shutil.which("docker"):
        print("ERROR: docker not found", file=sys.stderr)
        sys.exit(1)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=== pgmemory Docker test runner ===\n")

    # Build
    print("── Building test image ──")
    build_cmd = ["docker", "build", "-f", DOCKERFILE, "-t", IMAGE, "."]
    if args.no_cache:
        build_cmd.insert(3, "--no-cache")
    result = subprocess.run(build_cmd, cwd=project_root)
    if result.returncode != 0:
        sys.exit(result.returncode)

    # Run
    print("\n── Running tests ──")
    result = subprocess.run(["docker", "run", "--rm", IMAGE])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
