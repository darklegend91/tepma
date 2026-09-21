"""The language model client: a local Ollama, or any OpenAI-compatible server.

Two backends behind one API, chosen by configuration:

    ollama   (default)  models on this machine, fully offline
    openai              an OpenAI-compatible endpoint - a vLLM box on the LAN, say

The kiosk asks for structured output on every call, and the two servers spell that
differently: Ollama takes `format: <schema>`, while an OpenAI-compatible server takes
`response_format: {"type": "json_schema", ...}`. vLLM's own `guided_json` is not used - the
server at hand accepted the field and then ignored it, returning prose where JSON was
required, which is worse than an error.

Qwen3 is a hybrid reasoning model on both backends and its thinking has to be off: it is
slow, and the reasoning leaks into text the kiosk reads out loud. Ollama takes
`think: false`; the OpenAI route passes `chat_template_kwargs: {"enable_thinking": false}`,
and retries without it if a server rejects the field.

Nothing here imports the speech or numeric stack, so it can be exercised on its own.
"""
import json
import os

import httpx

try:                                    # keeps this module importable without the extra
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _setting(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise RuntimeError(f"{name} must not be empty")
    return value


OLLAMA_URL = _setting("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
# Profile extraction: read once at the end of an interview, accuracy over latency.
LLM_MODEL = _setting("LLM_MODEL", "qwen3:8b")
# Live interview questions. On Ollama this is deliberately a different, non-thinking
# instruct model: qwen3:8b with "think": false leaked its bookkeeping into the spoken
# reply. An instruct model has no thinking mode to suppress.
TURN_MODEL = _setting("TURN_MODEL", "qwen3:4b-instruct")

# An OpenAI-compatible base URL, e.g. "http://172.16.20.39:8000/v1". Setting it is enough
# to switch backends; LLM_BACKEND forces one either way.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# One served model does both jobs unless a second is named.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip()
OPENAI_TURN_MODEL = os.getenv("OPENAI_TURN_MODEL", "").strip() or OPENAI_MODEL

LLM_BACKEND = os.getenv("LLM_BACKEND", "").strip().lower() or (
    "openai" if OPENAI_BASE_URL else "ollama")

TIMEOUT = float(os.getenv("LLM_TIMEOUT_S", "300"))


def using_remote() -> bool:
    return LLM_BACKEND == "openai" and bool(OPENAI_BASE_URL)


def describe() -> str:
    if using_remote():
        return f"{OPENAI_BASE_URL} ({OPENAI_MODEL or 'default model'})"
    return f"{OLLAMA_URL} ({LLM_MODEL} / {TURN_MODEL})"


def _remote_model(model: str | None) -> str:
    """Map an Ollama model name onto the served one.

    Callers ask for TURN_MODEL or LLM_MODEL by name. A remote server has its own catalogue
    and those names mean nothing to it, so the turn model maps to OPENAI_TURN_MODEL and
    everything else to OPENAI_MODEL.
    """
    if model and model == TURN_MODEL and OPENAI_TURN_MODEL:
        return OPENAI_TURN_MODEL
    return OPENAI_MODEL or OPENAI_TURN_MODEL or (model or LLM_MODEL)


def _headers() -> dict:
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"} if OPENAI_API_KEY else {}


def _openai_body(messages: list[dict], system: str, schema: dict | None,
                 model: str | None, max_tokens: int | None, stream: bool) -> dict:
    body = {
        "model": _remote_model(model),
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }
    if max_tokens:
        body["max_tokens"] = max_tokens
    return body


def _ollama_body(messages: list[dict], system: str, schema: dict | None,
                 model: str | None, max_tokens: int | None, stream: bool) -> dict:
    body = {
        "model": model or LLM_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": stream,
        "think": False,
    }
    if schema is not None:
        body["format"] = schema
    if max_tokens:
        body["options"] = {"num_predict": max_tokens}
    return body


def strip_thinking(text: str) -> str:
    return text.split("</think>", 1)[1].strip() if "</think>" in text else text.strip()


async def _post_openai(client: httpx.AsyncClient, body: dict) -> httpx.Response:
    """POST, retrying once without chat_template_kwargs for servers that reject it."""
    response = await client.post(f"{OPENAI_BASE_URL}/chat/completions",
                                 json=body, headers=_headers())
    if response.status_code == 400 and "chat_template_kwargs" in body:
        body = {key: value for key, value in body.items() if key != "chat_template_kwargs"}
        response = await client.post(f"{OPENAI_BASE_URL}/chat/completions",
                                     json=body, headers=_headers())
    response.raise_for_status()
    return response


async def llm_extract(messages: list[dict], system: str, schema: dict,
                      model: str | None = None, max_tokens: int | None = None) -> dict:
    """Run the model with structured output: the reply is forced to match schema.

    max_tokens caps the generation. Worth setting for anything spoken aloud: a small model
    writing an unfamiliar script can loop instead of stopping - a Punjabi interview turn
    ran until the client timeout and killed the session, where the same question takes
    about eight seconds when it terminates normally.
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        if using_remote():
            body = _openai_body(messages, system, schema, model, max_tokens, stream=False)
            response = await _post_openai(client, body)
            content = response.json()["choices"][0]["message"]["content"]
        else:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=_ollama_body(messages, system, schema, model, max_tokens, stream=False))
            response.raise_for_status()
            content = response.json()["message"]["content"]
    return json.loads(strip_thinking(content))


async def llm_stream(messages: list[dict], system: str, schema: dict,
                     model: str | None = None, max_tokens: int | None = None):
    """Yield response deltas as they are generated; the caller reassembles the JSON."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        if using_remote():
            body = _openai_body(messages, system, schema, model, max_tokens, stream=True)
            async with client.stream("POST", f"{OPENAI_BASE_URL}/chat/completions",
                                     json=body, headers=_headers()) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        return
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
            return

        body = _ollama_body(messages, system, schema, model, max_tokens, stream=True)
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                delta = (payload.get("message") or {}).get("content", "")
                if delta:
                    yield delta
                if payload.get("done"):
                    return


async def llm_chat(messages: list[dict], system: str) -> str:
    """A plain conversational reply, no schema."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        if using_remote():
            body = _openai_body(messages, system, None, None, None, stream=False)
            response = await _post_openai(client, body)
            content = response.json()["choices"][0]["message"]["content"]
        else:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=_ollama_body(messages, system, None, None, None, stream=False))
            response.raise_for_status()
            content = response.json()["message"]["content"]
    return strip_thinking(content)
