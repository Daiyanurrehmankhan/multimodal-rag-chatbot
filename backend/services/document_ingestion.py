"""Document ingestion, image summarization, and chunking helpers."""

import base64
import os
from glob import glob

from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DEFAULT_IMAGE_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    print("Warning: Gemini Client not initialized.")
    gemini_client = None


def infer_corpus_from_filename(filename: str) -> str:
    """Lightweight, domain-agnostic corpus label based on filename."""
    return "General"


def _extract_text_from_parts(response_obj) -> str:
    """Return concatenated text-only parts from a Gemini response object."""
    texts = []
    candidates = getattr(response_obj, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                texts.append(part_text)
    return "".join(texts)


def _resolve_image_models() -> list[str]:
    """Return preferred image models from env or defaults, in failover order."""
    configured = os.getenv("GEMINI_IMAGE_MODELS", "").strip()
    if not configured:
        return DEFAULT_IMAGE_MODELS

    models = [model.strip() for model in configured.split(",") if model.strip()]
    return models or DEFAULT_IMAGE_MODELS


def process_image_to_document(file_path: str, client: genai.Client):
    """Generate a text summary for images so they can be embedded like text docs."""
    if not client:
        return []
    try:
        with open(file_path, "rb") as file_handle:
            base64_image = base64.b64encode(file_handle.read()).decode("utf-8")
        mime_type = f"image/{os.path.splitext(file_path)[1].lstrip('.')}"
        prompt = (
            "Provide a concise, detailed, professional summary of the image content. "
            "Focus on technical or informational aspects only."
        )
        response_text = ""
        last_error = None
        for model_name in _resolve_image_models():
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {"text": prompt},
                                {"inlineData": {"mimeType": mime_type, "data": base64_image}},
                            ],
                        }
                    ],
                )
                response_text = _extract_text_from_parts(response)
                if response_text.strip():
                    break
                raise RuntimeError(f"Model '{model_name}' returned an empty image description.")
            except Exception as exc:
                last_error = exc
                continue

        if not response_text.strip() and last_error is not None:
            raise RuntimeError(str(last_error))

        if not response_text.strip():
            response_text = "No descriptive text could be extracted from this image."
        return [
            Document(
                page_content=response_text,
                metadata={
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "type": "image_description",
                    "corpus": "Images",
                },
            )
        ]
    except Exception as exc:
        print(f"Image processing error: {exc}")
        return []


def load_and_prepare_documents():
    """Load supported files from data/, split content, and return clean chunk documents."""
    all_files = glob("data/*")
    documents = []
    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        loader = None
        if ext in [".txt", ".md"]:
            loader = TextLoader(file_path)
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
            print(f"Loaded PDF: {file_path}")
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext in [".jpg", ".jpeg", ".png"]:
            documents.extend(process_image_to_document(file_path, gemini_client))
            continue
        if not loader:
            continue
        try:
            loaded_docs = loader.load()
            for doc in loaded_docs:
                if not doc.page_content.strip():
                    continue
                source_path = doc.metadata.get("source", file_path)
                file_name = os.path.basename(source_path)
                doc.metadata["file_name"] = file_name
                doc.metadata["corpus"] = infer_corpus_from_filename(file_name)
                documents.append(doc)
        except Exception as exc:
            print(f"Error loading {file_path}: {exc}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    chunks = splitter.split_documents(documents)
    clean_chunks = [chunk for chunk in chunks if chunk.page_content and len(chunk.page_content.strip()) >= 30]

    for chunk in clean_chunks:
        source_path = chunk.metadata.get("source", "unknown")
        file_name = chunk.metadata.get("file_name", os.path.basename(source_path))
        chunk.metadata["file_name"] = file_name
        chunk.metadata["source"] = source_path

    return clean_chunks


def load_and_chunk_single_file(file_path: str):
    """Load one file and return list of chunk documents with metadata."""
    ext = os.path.splitext(file_path)[1].lower()
    documents = []
    if ext in [".txt", ".md"]:
        loader = TextLoader(file_path)
        for doc in loader.load():
            if doc.page_content.strip():
                doc.metadata["file_name"] = os.path.basename(file_path)
                doc.metadata["source"] = file_path
                documents.append(doc)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
        for doc in loader.load():
            if doc.page_content.strip():
                doc.metadata["file_name"] = os.path.basename(file_path)
                doc.metadata["source"] = file_path
                documents.append(doc)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
        for doc in loader.load():
            if doc.page_content.strip():
                doc.metadata["file_name"] = os.path.basename(file_path)
                doc.metadata["source"] = file_path
                documents.append(doc)
    elif ext in [".jpg", ".jpeg", ".png"] and gemini_client:
        documents = process_image_to_document(file_path, gemini_client)
    else:
        return []
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    chunks = splitter.split_documents(documents)
    return [chunk for chunk in chunks if chunk.page_content and len(chunk.page_content.strip()) >= 30]
