#!/usr/bin/env bash
# Install, start, or stop the whole project - Ollama and the voice server - with one command.
#
#   scripts/tepma.sh setup     one-time install: venv, dependencies, models
#   scripts/tepma.sh start     ollama serve + uvicorn on $PORT, waits until both answer
#   scripts/tepma.sh stop      stops the server and frees the models from memory
#   scripts/tepma.sh unload    keeps the server up but evicts the models from RAM (~7 GB)
#   scripts/tepma.sh status    what is running, and what is loaded
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

# Marks an Ollama that this script started. Anything already running belongs to the user,
# and `stop` unloads our models from it rather than killing their process.
OLLAMA_OWNED="$LOG_DIR/.ollama-started-by-tepma"

# The models come from .env when it sets them, so a fine-tuned build is pulled and freed
# by the same commands as the defaults.
env_value() { [ -f "$PROJECT_DIR/.env" ] && sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*\"\{0,1\}\([^\"#]*\)\"\{0,1\}.*/\1/p" "$PROJECT_DIR/.env" | tail -1 | xargs || true; }
LLM_MODEL="${LLM_MODEL:-$(env_value LLM_MODEL)}";   LLM_MODEL="${LLM_MODEL:-qwen3:8b}"
TURN_MODEL="${TURN_MODEL:-$(env_value TURN_MODEL)}"; TURN_MODEL="${TURN_MODEL:-qwen3:4b-instruct}"

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
        touch "$OLLAMA_OWNED"
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
    echo "  open  http://$HOST:$PORT/assistant"
    echo "  logs  $LOG_DIR/server.log  $LOG_DIR/ollama.log"
}

stop() {
    # Pinned to this app AND this port: a second checkout, or a developer's own instance
    # on another port, is none of our business. Set PORT to stop a different one.
    pkill -f "uvicorn server:app --host $HOST --port $PORT" 2>/dev/null \
        && echo "  - voice server on $HOST:$PORT stopped" \
        || echo "  - no voice server running on $HOST:$PORT"
    pkill -f "cloudflared tunnel" 2>/dev/null && echo "  - tunnel stopped" || true

    if [ -f "$OLLAMA_OWNED" ]; then
        pkill -f "ollama serve" 2>/dev/null && echo "  - ollama stopped" \
            || echo "  - ollama was not running"
        rm -f "$OLLAMA_OWNED"
    elif ollama_up; then
        # Ollama was already running before we got here, so it is not ours to kill - but
        # our models are ours to free, and they hold about 7 GB between them.
        unload
        echo "  - ollama left running (it was not started by this script)"
    else
        echo "  - ollama was not running"
    fi
}

unload() {
    ollama_up || { echo "  - ollama is not running"; return 0; }
    for model in "$LLM_MODEL" "$TURN_MODEL"; do
        ollama stop "$model" >/dev/null 2>&1 && echo "  - unloaded $model" || true
    done
}

status() {
    [ -x "$VENV/python" ] && echo "  python env    installed" \
                          || echo "  python env    MISSING - run: scripts/tepma.sh setup"
    ollama_up && echo "  ollama        running" || echo "  ollama        stopped"
    server_up && echo "  voice server  running on http://$HOST:$PORT/assistant" \
               || echo "  voice server  stopped"
    if ollama_up; then
        # Captured first, not piped into grep: `grep -q` exits on the first match, which
        # kills `ollama list` with SIGPIPE, and `set -o pipefail` then reports the whole
        # pipeline as failed - so every pulled model looked missing.
        local pulled; pulled="$(ollama list 2>/dev/null || true)"
        for model in "$LLM_MODEL" "$TURN_MODEL"; do
            case "$pulled" in
                "$model "*|*"
$model "*) echo "  model         $model pulled" ;;
                *) echo "  model         $model MISSING - run: scripts/tepma.sh setup" ;;
            esac
        done
        echo; ollama ps
    fi
}

tunnel() {
    command -v cloudflared >/dev/null 2>&1 || {
        echo "cloudflared is not installed:  brew install cloudflared" >&2; exit 1; }
    start
    echo
    echo "  starting public tunnel - anyone with the URL can run an interview and print"
    nohup cloudflared tunnel --url "http://$HOST:$PORT" > "$LOG_DIR/tunnel.log" 2>&1 &
    for _ in $(seq 1 40); do
        url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/tunnel.log" 2>/dev/null | head -1 || true)
        [ -n "${url:-}" ] && { echo; echo "  PUBLIC URL:  $url/auto"; echo; return 0; }
        sleep 1
    done
    echo "  ! no URL yet - check $LOG_DIR/tunnel.log" >&2
}

setup() {
    echo "Setting up TePMA in $PROJECT_DIR"

    local python="${PYTHON:-python3.11}"
    command -v "$python" >/dev/null 2>&1 || python=python3
    command -v "$python" >/dev/null 2>&1 || {
        echo "  ! no Python found. Install Python 3.11:  brew install python@3.11" >&2; exit 1; }
    "$python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || {
        echo "  ! $python is too old - this project needs 3.10 or newer" >&2; exit 1; }

    if [ -x "$VENV/python" ]; then
        echo "  - virtual environment already exists"
    else
        echo "  - creating virtual environment (.venv)"
        "$python" -m venv "$PROJECT_DIR/.venv"
    fi

    echo "  - installing Python dependencies"
    "$VENV/pip" install -q --upgrade pip
    "$VENV/pip" install -q -r "$PROJECT_DIR/requirements.txt"

    command -v ollama >/dev/null 2>&1 || {
        echo "  ! ollama is not installed. Install it and run setup again:" >&2
        echo "      brew install ollama        (macOS)" >&2
        echo "      curl -fsSL https://ollama.com/install.sh | sh   (Linux)" >&2
        exit 1; }

    if ! ollama_up; then
        echo "  - starting ollama"
        nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
        touch "$OLLAMA_OWNED"
        wait_for ollama ollama_up 60
    fi

    local pulled; pulled="$(ollama list 2>/dev/null || true)"
    for model in "$LLM_MODEL" "$TURN_MODEL"; do
        case "$pulled" in
            "$model "*|*"
$model "*) echo "  - $model already pulled" ;;
            *) echo "  - pulling $model (several GB, once)"; ollama pull "$model" ;;
        esac
    done

    # Whisper and Kokoro download on first use and would otherwise make the first
    # interview the slow one. This also fills the speech cache for the greeting.
    "$VENV/python" "$PROJECT_DIR/scripts/prepare_models.py"

    [ -f "$PROJECT_DIR/.env" ] || {
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env" 2>/dev/null \
            && echo "  - wrote .env from .env.example"; }

    echo
    echo "Setup complete.  Start it with:  scripts/tepma.sh start"
}

case "${1:-start}" in
    setup)  setup   ;;
    start)  start   ;;
    stop)   stop    ;;
    unload) unload  ;;
    status) status  ;;
    tunnel) tunnel  ;;
    restart) stop; sleep 1; start ;;
    *) echo "usage: $0 {setup|start|stop|restart|unload|status|tunnel}" >&2; exit 1 ;;
esac
