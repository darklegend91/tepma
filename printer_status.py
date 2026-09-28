"""Small CUPS helpers shared by the UI status and automated printing flow."""
import subprocess

def _lpstat(*args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["lpstat", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def default_printer_status() -> dict:
    """Return whether the default CUPS printer is ready to accept a print job."""
    default = _lpstat("-d")
    prefix = "system default destination:"
    if default is None or default.returncode != 0:
        return {"connected": False, "name": None, "state": "not_connected",
                "detail": None}

    output = default.stdout.strip()
    if not output.lower().startswith(prefix):
        return {"connected": False, "name": None, "state": "not_connected",
                "detail": None}

    name = output[len(prefix):].strip()
    if not name:
        return {"connected": False, "name": None, "state": "not_connected",
                "detail": None}

    printer = _lpstat("-p", name)
    accepting = _lpstat("-a", name)
    details = " ".join(
        part.strip().lower()
        for result in (printer, accepting)
        if result is not None
        for part in (result.stdout, result.stderr)
        if part.strip()
    )
    ready = (
        printer is not None
        and printer.returncode == 0
        and accepting is not None
        and accepting.returncode == 0
        and "accepting requests" in details
        and not any(marker in details for marker in ERROR_MARKERS)
    )
    if ready:
        return {"connected": True, "name": name, "state": "ready", "detail": None}

    # Not ready is not the same as not there, and saying "not connected" about a printer
    # that is plugged in, is the default, and has simply had its queue stopped by CUPS
    # sends somebody looking for a loose cable. The most common of these by far is
    # "Filter failed": CUPS disables the whole queue after one job it could not render,
    # and every interview afterwards silently stops printing until a person runs
    # cupsenable. Say which it is.
    state, reason = "not_ready", None
    for marker, code in (("filter failed", "filter_failed"),
                         ("disabled", "queue_disabled"),
                         ("stopped", "queue_stopped"),
                         ("offline", "offline"),
                         ("paused", "queue_stopped")):
        if marker in details:
            state, reason = code, marker
            break
    else:
        if "not accepting requests" in details:
            state = "not_accepting"
    return {"connected": False, "name": name, "state": state,
            "detail": _first_line_about(printer, reason)}


ERROR_MARKERS = (
    "disabled",
    "offline",
    "not connected",
    "filter failed",
    "unable to",
    "printer error",
    "stopped",
)


def _first_line_about(printer, marker: str | None) -> str | None:
    """What CUPS actually said, so an operator has something to act on."""
    if printer is None or not printer.stdout:
        return None
    lines = [line.strip() for line in printer.stdout.splitlines() if line.strip()]
    if marker:
        for line in lines:
            if marker in line.lower():
                return line
    return lines[0] if lines else None


def queue_is_empty(name: str) -> bool | None:
    """Has everything finished printing on this queue? None when it cannot be told."""
    result = _lpstat("-o", name)
    if result is None or result.returncode != 0:
        return None
    return not result.stdout.strip()
