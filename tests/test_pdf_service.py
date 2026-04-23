import unittest
from unittest.mock import patch

from backend.services import pdf_service


class PdfServiceTests(unittest.TestCase):
    def test_build_pdf_response_success(self):
        turns = [{"turn_index": 1, "response_text": "# Title\n\n* item"}]
        with patch.object(pdf_service, "list_turns", return_value=turns):
            buffer, error = pdf_service.build_pdf_response("s1")

        self.assertIsNone(error)
        self.assertIsNotNone(buffer)
        self.assertTrue(buffer.read().startswith(b"%PDF"))

    def test_build_pdf_response_for_selected_turn(self):
        turns = [
            {"turn_index": 1, "response_text": "old response"},
            {"turn_index": 2, "response_text": "selected response"},
        ]
        with patch.object(pdf_service, "list_turns", return_value=turns):
            buffer, error = pdf_service.build_pdf_response("s1", turn_index="2")

        self.assertIsNone(error)
        self.assertIsNotNone(buffer)
        self.assertTrue(buffer.read().startswith(b"%PDF"))

    def test_build_pdf_response_for_message_index(self):
        turns = [
            {"turn_index": 1, "response_text": "first response"},
            {"turn_index": 2, "response_text": "second response"},
        ]
        with patch.object(pdf_service, "list_turns", return_value=turns):
            buffer, error = pdf_service.build_pdf_response("s1", message_index="0")

        self.assertIsNone(error)
        self.assertIsNotNone(buffer)
        self.assertTrue(buffer.read().startswith(b"%PDF"))

    def test_build_pdf_response_prefers_explicit_response_text(self):
        with patch.object(pdf_service, "list_turns", return_value=[]):
            buffer, error = pdf_service.build_pdf_response("s1", response_text="manual selected response")

        self.assertIsNone(error)
        self.assertIsNotNone(buffer)
        self.assertTrue(buffer.read().startswith(b"%PDF"))

    def test_build_pdf_response_turn_index_wins_over_response_text(self):
        turns = [
            {"turn_index": 1, "response_text": "first response"},
            {"turn_index": 2, "response_text": "selected response"},
        ]
        with patch.object(pdf_service, "list_turns", return_value=turns):
            buffer, error = pdf_service.build_pdf_response(
                "s1",
                turn_index="2",
                response_text="compat response",
            )

        self.assertIsNone(error)
        self.assertIsNotNone(buffer)
        self.assertTrue(buffer.read().startswith(b"%PDF"))

    def test_build_pdf_response_missing_session(self):
        buffer, error = pdf_service.build_pdf_response("")
        self.assertIsNone(buffer)
        self.assertEqual(error[1], 400)


if __name__ == "__main__":
    unittest.main()
