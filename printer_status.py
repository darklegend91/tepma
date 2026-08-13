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
        return {"connected": False, "name": None, "state": "not_connected"}

    output = default.stdout.strip()
    if not output.lower().startswith(prefix):
        return {"connected": False, "name": None, "state": "not_connected"}

    name = output[len(prefix):].strip()
    if not name:
        return {"connected": False, "name": None, "state": "not_connected"}

    printer = _lpstat("-p", name)
    accepting = _lpstat("-a", name)
    details = " ".join(
        part.strip().lower()
        for result in (printer, accepting)
        if result is not None
        for part in (result.stdout, result.stderr)
        if part.strip()
    )
    error_markers = (
        "disabled",
        "offline",
        "not connected",
        "filter failed",
        "unable to",
        "printer error",
        "stopped",
    )
    ready = (
        printer is not None
        and printer.returncode == 0
        and accepting is not None
        and accepting.returncode == 0
        and "accepting requests" in details
        and not any(marker in details for marker in error_markers)
    )
    return {
        "connected": ready,
        "name": name if ready else None,
        "state": "ready" if ready else "not_connected",
    }
