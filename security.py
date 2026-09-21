"""Who may talk to the server. Shared by the HTTP middleware and the WebSocket handlers.

The kiosk binds to 127.0.0.1, which keeps other machines out but not other *websites*: any
page open in the kiosk's browser can send requests to 127.0.0.1, and a hostile page can
rebind its own domain name to 127.0.0.1 to read the responses. Two rules close that:

* Host header - only requests addressed to this machine (or a tunnel hostname explicitly
  listed in ALLOWED_HOSTS) are served. A rebinding attack arrives with the attacker's
  hostname in Host, so it is refused.
* WebSocket Origin - browsers do not apply the same-origin policy to WebSockets, so the
  server has to. A socket opened by a page from another origin is closed before it can
  stream audio or read an interview.
"""
import fnmatch
import os
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv()   # read here, not relied on from engines.py: import order must not matter

LOCAL_HOSTS = ["127.0.0.1", "localhost"]
# Comma-separated extra hostnames, wildcards allowed - e.g. "*.trycloudflare.com" for a tunnel.
EXTRA_HOSTS = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()]
ALLOWED_HOSTS = LOCAL_HOSTS + EXTRA_HOSTS

# Longest line the /speak endpoints will synthesize. The longest scripted line is ~150
# characters and a capped interview question is well under this; without a limit, one
# request with 20,000 characters of text occupied the synthesizer for minutes.
MAX_SPEAK_CHARS = 800


def host_allowed(hostname: str) -> bool:
    return any(fnmatch.fnmatch(hostname.lower(), pattern.lower()) for pattern in ALLOWED_HOSTS)


def websocket_origin_allowed(headers) -> bool:
    """Same-origin, or an allowed host. No Origin at all means a non-browser client."""
    origin = headers.get("origin")
    if not origin:
        return True
    parts = urlsplit(origin)
    return parts.netloc == headers.get("host", "") or host_allowed(parts.hostname or "")
