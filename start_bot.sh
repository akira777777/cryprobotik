#!/usr/bin/env bash
# start_bot.sh — starts localtunnel + bot together
# Usage: bash start_bot.sh [paper|live]

set -e
MODE="${1:-paper}"
BOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$BOT_DIR/.venv313/Scripts/python.exe"
PORT=8080
SUBDOMAIN="cryprobotik"

echo "=== Cryprobotik startup ==="
echo "Mode: $MODE"
echo "Port: $PORT"

# Kill any existing bot process on port 8080
echo "Stopping existing bot (if any)..."
PIDS=$(netstat -ano 2>/dev/null | grep ":$PORT " | grep LISTENING | awk '{print $5}' || true)
for pid in $PIDS; do
  [ -n "$pid" ] && [ "$pid" != "0" ] && taskkill //F //PID $pid 2>/dev/null && echo "Killed PID $pid" || true
done
sleep 1

# Start localtunnel in background
echo "Starting tunnel (subdomain: $SUBDOMAIN)..."
lt --port $PORT --subdomain $SUBDOMAIN > /tmp/lt_out.txt 2>&1 &
LT_PID=$!
sleep 3

# Read URL from localtunnel output
LT_URL=$(grep -o 'https://[^ ]*\.loca\.lt' /tmp/lt_out.txt | head -1)
if [ -z "$LT_URL" ]; then
  # subdomain taken — use random URL
  LT_URL=$(grep -o 'https://[^ ]*\.loca\.lt' /tmp/lt_out.txt | head -1)
fi
if [ -z "$LT_URL" ]; then
  echo "WARNING: Could not detect tunnel URL. Check /tmp/lt_out.txt"
  LT_URL="http://localhost:$PORT"
fi

echo ""
echo "==========================================="
echo "  Mini App URL: $LT_URL/app"
echo "==========================================="
echo ""
echo "Set this in Telegram bot: MINIAPP_URL=$LT_URL/app"
echo ""

# Start bot with MINIAPP_URL set
export MINIAPP_URL="$LT_URL/app"
echo "Starting bot in $MODE mode..."
"$PYTHON" -m src.main --mode "$MODE" &
BOT_PID=$!

echo "Bot PID: $BOT_PID"
echo "Tunnel PID: $LT_PID"
echo ""
echo "Logs: tail -f bot.log  (if redirected)"
echo "Stop: kill $BOT_PID $LT_PID"
echo ""
echo "Waiting for bot..."
wait $BOT_PID
