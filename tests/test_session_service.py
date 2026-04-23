import unittest
from unittest.mock import patch

from backend.services import session_service


class SessionServiceTests(unittest.TestCase):
    def test_get_chat_session_messages_returns_turns(self):
        sample_turn = {
            "turn_index": 1,
            "user_query": "q",
            "prompt_text": "p",
            "response_text": "r",
            "selected_course": "AI/ML",
            "user_role": "student",
            "created_at": None,
            "updated_at": None,
        }
        with patch.object(session_service, "list_sessions", return_value=[{"session_id": "s1"}]), patch.object(
            session_service, "list_turns", return_value=[sample_turn]
        ), patch.object(session_service, "log_owner_resolution"):
            payload, status = session_service.get_chat_session_messages("s1", {"owner_key": "owner-1"})

        self.assertEqual(status, 200)
        self.assertEqual(payload["session_id"], "s1")
        self.assertEqual(payload["turns"][0]["turn_index"], 1)

    def test_delete_chat_session_success(self):
        with patch.object(session_service, "delete_session", return_value=1), patch.object(
            session_service, "log_owner_resolution"
        ):
            payload, status = session_service.delete_chat_session("s1", {"owner_key": "owner-1"})

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "deleted")
        self.assertEqual(payload["deleted_count"], 1)

    def test_delete_chat_session_not_found(self):
        with patch.object(session_service, "delete_session", return_value=0), patch.object(
            session_service, "log_owner_resolution"
        ):
            payload, status = session_service.delete_chat_session("missing", {"owner_key": "owner-1"})

        self.assertEqual(status, 404)
        self.assertEqual(payload["status"], "not_found_or_error")


if __name__ == "__main__":
    unittest.main()
