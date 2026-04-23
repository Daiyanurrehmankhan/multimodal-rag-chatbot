import unittest

from backend.utils.identity import resolve_owner_from_payload, resolve_owner_from_query


class IdentityUtilsTests(unittest.TestCase):
    def test_resolve_owner_from_payload_prefers_browser_id(self):
        owner_key, source = resolve_owner_from_payload(
            {
                "browser_id": "browser-123",
                "owner_key": "owner-abc",
                "user_id": 7,
            }
        )
        self.assertEqual(owner_key, "browser-123")
        self.assertEqual(source, "browser_id")

    def test_resolve_owner_from_query_falls_back_to_anonymous(self):
        owner_key, source = resolve_owner_from_query({})
        self.assertEqual(owner_key, "anonymous")
        self.assertEqual(source, "fallback")


if __name__ == "__main__":
    unittest.main()
