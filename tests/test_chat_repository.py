import unittest
from unittest.mock import patch

from backend.repositories import chat_repository


class ChatRepositoryTests(unittest.TestCase):
    def test_delete_session_delegates_to_data_layer(self):
        with patch.object(chat_repository, "delete_chat_session", return_value=1) as delete_fn:
            result = chat_repository.delete_session("s1", "owner-1")

        self.assertEqual(result, 1)
        delete_fn.assert_called_once_with(session_id="s1", owner_key="owner-1")


if __name__ == "__main__":
    unittest.main()
