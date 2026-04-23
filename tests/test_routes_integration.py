import unittest
from io import BytesIO
from unittest.mock import patch


class RouteIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._runtime_patcher = patch("backend.bootstrap.initialize_runtime", return_value=None)
        cls._runtime_patcher.start()

        from rag_server import create_app

        cls.app = create_app()
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls._runtime_patcher.stop()

    def test_app_registers_expected_routes(self):
        rules = {rule.rule for rule in self.app.url_map.iter_rules()}
        self.assertIn("/chat", rules)
        self.assertIn("/chat_sessions", rules)
        self.assertIn("/chat_sessions/<session_id>", rules)
        self.assertIn("/download_pdf", rules)

    def test_chat_route_streams_response(self):
        prepared_payload = {
            "query": "hello",
            "session_id": "s-1",
            "user_id": 1,
            "user_role": "student",
            "selected_course": "AI/ML",
            "owner_key": "browser-1",
        }

        with patch("backend.routes.chat_routes.prepare_chat_request", return_value=(prepared_payload, None)), patch(
            "backend.routes.chat_routes.start_chat_stream", return_value=["Hi", " there"]
        ):
            response = self.client.post("/chat", json={"query": "hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/event-stream")
        self.assertEqual(response.get_data(as_text=True), "Hi there")
        self.assertEqual(response.headers.get("X-Accel-Buffering"), "no")
        self.assertEqual(response.headers.get("Cache-Control"), "no-cache, no-transform")

    def test_chat_session_history_route_returns_json(self):
        with patch(
            "backend.routes.session_routes.get_chat_session_messages",
            return_value=({"session_id": "s-1", "turns": [{"turn_index": 1}]}, 200),
        ):
            response = self.client.get("/chat_sessions/s-1?owner_key=owner-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["session_id"], "s-1")
        self.assertEqual(response.get_json()["turns"][0]["turn_index"], 1)

    def test_chat_session_delete_route_returns_deleted_payload(self):
        with patch(
            "backend.routes.session_routes.delete_chat_session",
            return_value=({"status": "deleted", "deleted_count": 1}, 200),
        ):
            response = self.client.delete("/chat_sessions/s-1?owner_key=owner-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "deleted")
        self.assertEqual(response.get_json()["deleted_count"], 1)

    def test_download_pdf_route_returns_pdf_attachment(self):
        buffer = BytesIO(b"%PDF-1.4\nroute-test")
        with patch("backend.routes.document_routes.build_pdf_response", return_value=(buffer, None)) as mocked_build:
            response = self.client.get("/download_pdf?session_id=s-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")
        self.assertIn("attachment; filename=response.pdf", response.headers.get("Content-Disposition", ""))
        self.assertTrue(response.data.startswith(b"%PDF"))
        mocked_build.assert_called_once_with(
            "s-1",
            turn_index=None,
            message_index=None,
            response_text=None,
        )

    def test_download_pdf_route_supports_turn_index_query(self):
        buffer = BytesIO(b"%PDF-1.4\nturn-test")
        with patch("backend.routes.document_routes.build_pdf_response", return_value=(buffer, None)) as mocked_build:
            response = self.client.get("/download_pdf?session_id=s-1&turn_index=2")

        self.assertEqual(response.status_code, 200)
        mocked_build.assert_called_once_with(
            "s-1",
            turn_index="2",
            message_index=None,
            response_text=None,
        )

    def test_download_pdf_route_supports_message_index_query(self):
        buffer = BytesIO(b"%PDF-1.4\nmessage-index-test")
        with patch("backend.routes.document_routes.build_pdf_response", return_value=(buffer, None)) as mocked_build:
            response = self.client.get("/download_pdf?session_id=s-1&message_index=0")

        self.assertEqual(response.status_code, 200)
        mocked_build.assert_called_once_with(
            "s-1",
            turn_index=None,
            message_index="0",
            response_text=None,
        )

    def test_download_pdf_route_supports_response_text_query(self):
        buffer = BytesIO(b"%PDF-1.4\nresponse-text-test")
        with patch("backend.routes.document_routes.build_pdf_response", return_value=(buffer, None)) as mocked_build:
            response = self.client.get("/download_pdf?session_id=s-1&response_text=selected")

        self.assertEqual(response.status_code, 200)
        mocked_build.assert_called_once_with(
            "s-1",
            turn_index=None,
            message_index=None,
            response_text="selected",
        )

    def test_download_pdf_route_rejects_post(self):
        response = self.client.post("/download_pdf", json={"session_id": "s-1"})

        self.assertEqual(response.status_code, 405)


if __name__ == "__main__":
    unittest.main()