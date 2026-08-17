#!/usr/bin/env bash
# Start or stop the whole project - Ollama and the voice server - with one command.
#
#   scripts/tepma.sh start     ollama serve + uvicorn on $PORT, waits until both answer
#   scripts/tepma.sh stop      stops both
#   scripts/tepma.sh status    what is running
#   scripts/tepma.sh tunnel    start, then expose it on a public https URL
#
# Both services log to logs/. Nothing is started twice: an already-running Ollama is
# reused, which matters because a second `ollama serve` fails with "address already in use".
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
VENV="$PROJECT_DIR/.venv/bin"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

ollama_up()  { curl -sf --max-time 1 "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; }
server_up()  { curl -sf --max-time 1 "http://$HOST:$PORT/" >/dev/null 2>&1; }

wait_for() {  # wait_for <name> <predicate> <seconds>
    local name="$1" check="$2" limit="${3:-60}" waited=0
    while ! $check; do
        sleep 1; waited=$((waited + 1))
        if [ "$waited" -ge "$limit" ]; then
            echo "  ! $name did not come up in ${limit}s - see $LOG_DIR/" >&2
            return 1
        fi
    done
    echo "  - $name ready"
}

start() {
    if ollama_up; then
        echo "  - ollama already running"
    else
        echo "  - starting ollama"
        nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
        wait_for ollama ollama_up 60
    fi

    if server_up; then
        echo "  - voice server already running on $HOST:$PORT"
    else
        echo "  - starting voice server on $HOST:$PORT"
        nohup "$VENV/uvicorn" server:app --host "$HOST" --port "$PORT" \
            > "$LOG_DIR/server.log" 2>&1 &
        wait_for "voice server" server_up 90
    fi

    echo
    echo "  open  http://$HOST:$PORT/auto"
    echo "  logs  $LOG_DIR/server.log  $LOG_DIR/ollama.log"
}

stop() {
    # The uvicorn pattern is pinned to this app so a developer's other servers survive.
    pkill -f "uvicorn server:app" 2>/dev/null && echo "  - voice server stopped" \
        || echo "  - voice server was not running"
    pkill -f "cloudflared tunnel" 2>/dev/null && echo "  - tunnel stopped" || true
    pkill -f "ollama serve" 2>/dev/null && echo "  - ollama stopped" \
        || echo "  - ollama was not running"
}

status() {
    ollama_up && echo "  ollama        running" || echo "  ollama        stopped"
    server_up && echo "  voice server  running on http://$HOST:$PORT" \
               || echo "  voice server  stopped"
    if ollama_up; then echo; ollama ps; fi
}

tunnel() {
    command -v cloudflared >/dev/null 2>&1 || {
        echo "cloudflared is not installed:  brew install cloudflared" >&2; exit 1; }
    start
    echo
    echo "  starting public tunnel - anyone with the URL can run an interview and print"
    nohup cloudflared tunnel --url "http://$HOST:$PORT" > "$LOG_DIR/tunnel.log" 2>&1 &
    for _ in $(seq 1 40); do
        url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/tunnel.log" 2>/dev/null | head -1)
        [ -n "${url:-}" ] && { echo; echo "  PUBLIC URL:  $url/auto"; echo; return 0; }
        sleep 1
    done
    echo "  ! no URL yet - check $LOG_DIR/tunnel.log" >&2
}

case "${1:-start}" in
    start)  start   ;;
    stop)   stop    ;;
    status) status  ;;
    tunnel) tunnel  ;;
    restart) stop; sleep 1; start ;;
    *) echo "usage: $0 {start|stop|restart|status|tunnel}" >&2; exit 1 ;;
esac
