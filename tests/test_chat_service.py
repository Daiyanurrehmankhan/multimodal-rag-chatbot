import unittest
from unittest.mock import patch

from backend.services import chat_service


class ChatServiceTests(unittest.TestCase):
    def test_prepare_chat_request_valid(self):
        with patch.object(chat_service, "log_owner_resolution"):
            prepared, error = chat_service.prepare_chat_request(
                {
                    "query": "What is takaful?",
                    "selected_course": "AI/ML",
                    "user_role": "student",
                    "browser_id": "b-1",
                    "user_id": "9",
                }
            )
        self.assertIsNone(error)
        self.assertEqual(prepared["owner_key"], "b-1")
        self.assertEqual(prepared["user_id"], 9)
        self.assertEqual(prepared["selected_course"], "AI/ML")

    def test_prepare_chat_request_requires_query(self):
        prepared, error = chat_service.prepare_chat_request({"selected_course": "AI/ML"})
        self.assertIsNone(prepared)
        self.assertEqual(error[1], 400)
        self.assertIn("query", error[0]["error"])

    def test_start_chat_stream_collects_chunks_and_stores_last_response(self):
        prepared = {
            "query": "hello",
            "session_id": "s-1",
            "user_id": 1,
            "user_role": "student",
            "selected_course": "AI/ML",
            "owner_key": "browser-1",
        }

        def fake_chat(*_args, **_kwargs):
            yield "Hi"
            yield " there"

        with patch.object(chat_service, "rebuild_history", return_value=[]), patch.object(chat_service, "save_session") as save_session:
            with patch.dict(chat_service.chat_histories, {}, clear=True), patch.dict(chat_service.last_responses, {}, clear=True):
                chunks = list(chat_service.start_chat_stream(prepared, chat_callable=fake_chat))
                self.assertEqual(chat_service.last_responses["s-1"], "Hi there")

        self.assertEqual(chunks, ["Hi", " there"])
        save_session.assert_called_once()


if __name__ == "__main__":
    unittest.main()
