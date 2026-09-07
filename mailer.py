"""Emailing a finished resume.

Shared by the manual interview and the kiosk. Sending is always best-effort: the resume
is saved and printable before this runs, so a missing SMTP password or an unreachable
mail server must degrade to "not sent", never to a lost interview.
"""
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_USER") and os.environ.get("SMTP_PASS"))


def send_resume(to_addr: str, name: str, pdf: Path, docx: Path | None = None) -> None:
    """Send the resume, raising RuntimeError with a readable reason on failure."""
    to_addr = (to_addr or "").strip()
    if "@" not in to_addr:
        raise RuntimeError("No usable email address was captured")
    user, password = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    if not user or not password:
        raise RuntimeError("Email is not configured (SMTP_USER / SMTP_PASS are unset)")
    if not pdf.is_file():
        raise RuntimeError("The resume PDF has not been generated")

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = f"Your resume - {name or 'TePMA'}"
    msg.set_content(
        f"Hi {name or ''},\n\nYour profile has been created with TePMA. The attached PDF "
        "is print-ready and the Word file is editable.\n"
    )
    msg.add_attachment(pdf.read_bytes(), maintype="application", subtype="pdf",
                       filename="resume.pdf")
    if docx and docx.is_file():
        msg.add_attachment(
            docx.read_bytes(),
            maintype="application",
            subtype="vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename="resume.docx",
        )
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    try:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
    except Exception as exc:
        raise RuntimeError(f"Mail server rejected the message: {exc}") from exc
