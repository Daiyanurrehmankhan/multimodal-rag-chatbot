"""Embeddings provider used by indexing and retrieval modules."""

import os
import requests
from dotenv import load_dotenv


load_dotenv()


class JinaAPIEmbeddings:
    """Tiny adapter exposing embed_query/embed_documents for Jina Embeddings API."""

    def __init__(self, api_key: str, model: str = "jina-embeddings-v3", timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Request dense float embeddings for one or more input strings."""
        payload = {
            "model": self.model,
            "input": texts,
            "embedding_type": "float",
        }
        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise requests.HTTPError(
                f"Jina embeddings request failed ({response.status_code}): {response.text}"
            ) from exc
        data = response.json().get("data", [])
        return [item["embedding"] for item in data]

    def embed_query(self, text: str) -> list[float]:
        # Keep LangChain-like method name used throughout the existing codebase.
        embeddings = self._embed([text])
        if not embeddings:
            raise ValueError("Jina API returned no embedding for query.")
        return embeddings[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)


def get_embeddings():
    """Return the shared embeddings client used by retrieval and indexing code."""
    api_key = os.getenv("JINA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("JINA_API_KEY is missing. Add it to your .env file.")

    # Optional override so model can be switched without code changes.
    model_name = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3").strip() or "jina-embeddings-v3"
    return JinaAPIEmbeddings(api_key=api_key, model=model_name)
