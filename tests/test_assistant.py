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
        response = TestClient(app).get("/assistant")
        self.assertEqual(response.status_code, 200)
        self.assertIn("I want a document", response.text)
        self.assertIn("I want a résumé", response.text)


if __name__ == "__main__":
    unittest.main()
