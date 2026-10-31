#!/bin/sh
# Captures real Explorer views into .cache/media/capture.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/.cache/media/capture
PROFILE=$(mktemp -d)
mkdir -p "$OUT"
python3 "$ROOT/experiments/explorer/server.py" >/dev/null 2>&1 &
SERVER=$!
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu \
  --remote-debugging-port=9223 --user-data-dir="$PROFILE" about:blank >/dev/null 2>&1 &
CHROME=$!
trap 'kill $SERVER $CHROME 2>/dev/null; rm -rf "$PROFILE"' EXIT
sleep 3
node "$ROOT/media/capture/capture.mjs" "$OUT"
