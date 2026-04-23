import unittest

from backend.utils.streaming import stream_chunks


class StreamingUtilsTests(unittest.TestCase):
    def test_stream_chunks_passes_text_chunks(self):
        chunks = list(stream_chunks(["a", "", "b"]))
        self.assertEqual(chunks, ["a", "b"])

    def test_stream_chunks_emits_error_chunk(self):
        def broken_stream():
            yield "ok"
            raise RuntimeError("boom")

        chunks = list(stream_chunks(broken_stream()))
        self.assertEqual(chunks[0], "ok")
        self.assertIn("[stream_error]", chunks[1])

    def test_stream_chunks_supports_done_marker(self):
        chunks = list(stream_chunks(["x"], include_done_marker=True, done_marker="[done]"))
        self.assertEqual(chunks, ["x", "[done]"])


if __name__ == "__main__":
    unittest.main()
