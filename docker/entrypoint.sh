#!/bin/sh
# ==============================================================================
# Cryprobotik entrypoint
#
# 1. Waits for TimescaleDB to be reachable.
# 2. Applies schema.sql idempotently (CREATE ... IF NOT EXISTS everywhere).
# 3. Execs the bot so it receives signals (SIGTERM) directly.
# ==============================================================================
set -eu

: "${DATABASE_URL:?DATABASE_URL must be set}"

# Parse host:port from DATABASE_URL for the wait loop.
# Format: postgresql://user:pass@host:port/db
host_port="$(echo "$DATABASE_URL" | sed -E 's|^[a-z]+://[^@]+@([^/]+)/.*$|\1|')"
host="$(echo "$host_port" | cut -d: -f1)"
port="$(echo "$host_port" | cut -d: -f2)"

echo "[entrypoint] waiting for database at ${host}:${port}..."
i=0
until python -c "import socket, sys; s=socket.socket(); s.settimeout(2); s.connect(('${host}', ${port})); s.close()" 2>/dev/null; do
    i=$((i+1))
    if [ "$i" -ge 60 ]; then
        echo "[entrypoint] database did not become reachable after 60 attempts, giving up"
        exit 1
    fi
    sleep 1
done
echo "[entrypoint] database is reachable"

# Apply schema — schema.sql uses IF NOT EXISTS so this is safe on every boot.
echo "[entrypoint] applying schema..."
python -m src.data.storage --apply-schema

echo "[entrypoint] starting bot: $*"
exec "$@"
