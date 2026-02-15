#!/usr/bin/env bash
set -euo pipefail

export PGDATA=/var/lib/postgresql/data
export PGUSER=postgres

# Init cluster (ankane/pgvector skips this without the default entrypoint)
su postgres -c "initdb -D $PGDATA -E UTF8 --no-locale"

# Start Postgres
su postgres -c "pg_ctl start -w -l /tmp/pg.log"
sleep 1

# Setup test DB
su postgres -c "createdb pgmemory_test"
su postgres -c "psql -d pgmemory_test -c 'CREATE EXTENSION IF NOT EXISTS vector;'"

export PGMEMORY_TEST_URL="postgresql+asyncpg://postgres@localhost:5432/pgmemory_test"

echo ""
echo "── Unit tests ──"
/app/.venv/bin/pytest tests/test_unit.py -v

echo ""
echo "── Integration tests ──"
/app/.venv/bin/pytest tests/test_integration.py -v

echo ""
echo "=== All tests passed ==="
