import unittest
from unittest.mock import patch

from backend.bootstrap import validate_required_environment


class BootstrapValidationTests(unittest.TestCase):
    def test_validate_required_environment_rejects_missing_values(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                validate_required_environment()

        self.assertIn("Missing required environment variables", str(ctx.exception))

    def test_validate_required_environment_rejects_non_integer_port(self):
        env = {
            "DB_HOST": "localhost",
            "DB_PORT": "not-a-number",
            "DB_NAME": "rag_db",
            "DB_USER": "user",
            "DB_PASS": "pass",
            "GEMINI_API_KEY": "gemini-key",
            "JINA_API_KEY": "jina-key",
        }
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                validate_required_environment()

        self.assertEqual("DB_PORT must be an integer", str(ctx.exception))

    def test_validate_required_environment_accepts_valid_config(self):
        env = {
            "DB_HOST": "localhost",
            "DB_PORT": "5432",
            "DB_NAME": "rag_db",
            "DB_USER": "user",
            "DB_PASS": "pass",
            "GEMINI_API_KEY": "gemini-key",
            "JINA_API_KEY": "jina-key",
        }
        with patch.dict("os.environ", env, clear=True):
            validate_required_environment()


if __name__ == "__main__":
    unittest.main()
