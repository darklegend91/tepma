import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import routes_assistant
from document_render import render_application_docx, render_application_pdf
from server import app


SAMPLE_DOCUMENT = {
    "title": "Leave Application",
    "date": "13 August 2026",
    "recipient_lines": ["The Head of Department", "Example College"],
    "subject": "Request for one day of leave",
    "salutation": "Respected Sir/Madam,",
    "paragraphs": ["I request leave on 14 August 2026 due to a family commitment."],
    "closing": "Yours sincerely,",
    "sender_lines": ["Aman Singh"],
}


class RendererTests(unittest.TestCase):
    def test_application_renderers_create_real_files(self):
        pdf = render_application_pdf(SAMPLE_DOCUMENT)
        docx = render_application_docx(SAMPLE_DOCUMENT)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertTrue(docx.startswith(b"PK"))
        self.assertGreater(len(pdf), 1000)
        self.assertGreater(len(docx), 1000)

    def test_field_normalization_removes_duplicates_and_limits_size(self):
        raw = [
            {"key": "Full Name", "label": "Name", "question": "Your name?", "value": ""},
            {"key": "full-name", "label": "Duplicate", "question": "Again?", "value": ""},
        ]
        fields = routes_assistant._normalize_fields(raw, "en")
        self.assertEqual(fields, [{
            "key": "full_name", "label": "Name", "question": "Your name?", "value": "",
        }])


class AssistantWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.generated_patch = patch.object(
            routes_assistant, "GENERATED_DIR", Path(self.temp.name) / "generated"
        )
        self.generated_patch.start()
        routes_assistant._sessions.clear()

    async def asyncTearDown(self):
        self.generated_patch.stop()
        self.temp.cleanup()
        routes_assistant._sessions.clear()

    async def test_existing_document_is_selected_saved_and_not_printed_without_printer(self):
        decision = {
            "decision": "existing",
            "filename": "leave_application.pdf",
            "reason": "I found the leave application in the library.",
        }
        with (
            patch.object(routes_assistant, "llm_extract", AsyncMock(return_value=decision)),
            # The library is mocked rather than read from data/documents/: that folder is
            # runtime data, not in git, so a fresh clone has no leave_application.pdf and
            # the router correctly refuses a filename that is not in the library.
            patch.object(routes_assistant, "list_docs", return_value=["leave_application.pdf"]),
            patch.object(routes_assistant, "default_printer_status", return_value={
                "connected": False, "name": None, "state": "not_connected",
            }),
        ):
            result = await routes_assistant.document_start({
                "query": "Print the leave application", "language": "en",
            })
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["document"]["filename"], "leave_application.pdf")
        self.assertEqual(result["print_status"], "not_connected")
        self.assertFalse(result["printed"])

    async def test_missing_application_collects_details_and_generates_files(self):
        route = {
            "decision": "generate_application", "filename": "none",
            "reason": "A new request letter is needed.",
        }
        plan = {
            "document_title": "Library Card Request",
            "opening": "I need two details.",
            "fields": [
                {"key": "name", "label": "Name", "question": "Your full name?", "value": ""},
                {"key": "college", "label": "College", "question": "College name?", "value": ""},
            ],
        }
        model = AsyncMock(side_effect=[route, plan, SAMPLE_DOCUMENT])
        with (
            patch.object(routes_assistant, "llm_extract", model),
            patch.object(routes_assistant, "default_printer_status", return_value={
                "connected": False, "name": None, "state": "not_connected",
            }),
        ):
            started = await routes_assistant.document_start({
                "query": "Create a library card application", "language": "en",
            })
            first = await routes_assistant.document_turn({
                "session_id": started["session_id"], "answer": "Aman Singh",
            })
            completed = await routes_assistant.document_turn({
                "session_id": started["session_id"], "answer": "Example College",
            })

        self.assertEqual(started["status"], "collecting")
        self.assertEqual(first["collected"], 1)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["result"]["print_status"], "not_connected")
        directory = Path(self.temp.name) / "generated" / started["session_id"]
        self.assertTrue((directory / "document.pdf").is_file())
        self.assertTrue((directory / "document.docx").is_file())
        self.assertTrue((directory / "document.json").is_file())

        # A server reload can restore the workflow from disk for viewing/reprinting.
        routes_assistant._sessions.clear()
        restored = routes_assistant._session(started["session_id"])
        self.assertEqual(restored["status"], "completed")
        self.assertEqual(restored["dir"], directory)

    async def test_official_document_request_is_not_generated(self):
        decision = {
            "decision": "unsupported", "filename": "none",
            "reason": "A marksheet must be issued by the institution.",
        }
        with patch.object(routes_assistant, "llm_extract", AsyncMock(return_value=decision)):
            result = await routes_assistant.document_start({
                "query": "Generate my university marksheet", "language": "en",
            })
        self.assertEqual(result["status"], "unsupported")
        self.assertNotIn("pdf_url", result)


class PageTests(unittest.TestCase):
    def test_new_assistant_page_is_served(self):
        # A real host: the server refuses TestClient's default "testserver" by design,
        # since an unrecognised Host header is how a DNS-rebinding attack arrives.
        client = TestClient(app, base_url="http://127.0.0.1")
        response = client.get("/assistant")
        self.assertEqual(response.status_code, 200)
        # The kiosk is voice-only: one tap to begin, Stop as the only control.
        self.assertIn("Tap anywhere to begin", response.text)
        self.assertIn('id="stopBtn"', response.text)
        self.assertNotIn("<textarea", response.text)

    def test_unknown_host_is_refused(self):
        response = TestClient(app, base_url="http://evil.example").get("/assistant")
        self.assertEqual(response.status_code, 400)

    def test_legacy_resume_leak_is_not_mounted(self):
        client = TestClient(app, base_url="http://127.0.0.1")
        self.assertEqual(client.get("/interview/resume/latest").status_code, 404)
        self.assertEqual(client.post("/interview/email", json={"to": "x@example.com"}).status_code, 404)


class KioskLockTests(unittest.TestCase):
    """One microphone, one interview. Two tabs once ran two at the same time and printed
    a resume for a candidate who did not exist - see claim_kiosk()."""

    def setUp(self):
        import routes_auto
        self.routes_auto = routes_auto
        routes_auto._active = None
        self.addCleanup(setattr, routes_auto, "_active", None)

    def test_a_second_interview_is_refused_while_one_is_live(self):
        self.routes_auto.claim_kiosk("first")
        with self.assertRaises(self.routes_auto.KioskBusy):
            self.routes_auto.claim_kiosk("second")

    def test_releasing_frees_the_kiosk_for_the_next_person(self):
        self.routes_auto.claim_kiosk("first")
        self.routes_auto.release_kiosk("first")
        self.routes_auto.claim_kiosk("second")      # must not raise

    def test_an_abandoned_interview_expires(self):
        import time

        self.routes_auto.claim_kiosk("walked away")
        self.routes_auto._active["at"] = time.time() - self.routes_auto.KIOSK_IDLE_S - 1
        self.routes_auto.claim_kiosk("next person")  # must not raise


class SpokenDateTests(unittest.TestCase):
    """Dates only survive if the candidate said the numbers in them."""

    def test_a_year_nobody_said_is_dropped(self):
        from facts import apply_facts

        profile = {"education": [{"degree": "ITI", "institution": "ITI", "year": "2026"}]}
        result = apply_facts(profile, "Candidate: I did a carpentry course at ITI")
        self.assertEqual(result["education"][0]["year"], "")

    def test_a_year_said_in_words_is_kept(self):
        from facts import apply_facts

        profile = {"education": [{"degree": "BE", "institution": "Thapar", "year": "2025"}]}
        result = apply_facts(profile, "Candidate: I graduated in twenty twenty five")
        self.assertEqual(result["education"][0]["year"], "2025")

    def test_invented_job_dates_are_dropped(self):
        from facts import apply_facts

        profile = {"experience": [{"title": "Intern", "company": "A startup",
                                   "start": "2025-07-01", "end": "2025-12-31"}]}
        result = apply_facts(profile, "Candidate: I interned at a startup for six months")
        self.assertEqual(result["experience"][0]["start"], "")
        self.assertEqual(result["experience"][0]["end"], "")

    def test_an_academic_year_is_not_mistaken_for_an_invention(self):
        from facts import apply_facts

        profile = {"education": [{"degree": "BE", "institution": "Thapar",
                                  "year": "2024-25"}]}
        result = apply_facts(profile, "Candidate: I finished in 2024")
        self.assertEqual(result["education"][0]["year"], "2024-25")


if __name__ == "__main__":
    unittest.main()
