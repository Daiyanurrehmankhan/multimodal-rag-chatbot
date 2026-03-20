# embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs={"trust_remote_code": True, "device": "cpu"},  # no dtype
        encode_kwargs={"normalize_embeddings": True}
    )
