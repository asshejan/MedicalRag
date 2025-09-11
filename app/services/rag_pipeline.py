import os
from typing import List

import chromadb
from chromadb.config import Settings
from openai import OpenAI


def chunk_text(text: str, max_length: int = 500) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    for i in range(0, len(words), max_length):
        chunk = " ".join(words[i : i + max_length])
        chunks.append(chunk)
    return chunks


class RAGPipeline:
    """Lightweight RAG pipeline using OpenAI embeddings and ChromaDB.

    Avoids local transformer models to keep memory within Render free tier.
    """

    def __init__(self, collection_name: str = "rag_collection") -> None:
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._embed_model = "text-embedding-3-small"
        self._chroma = chromadb.Client(
            Settings(persist_directory=".chromadb")
        )
        self._collection = self._chroma.get_or_create_collection(collection_name)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=self._embed_model, input=texts
        )
        return [d.embedding for d in response.data]

    def add_document(self, text: str) -> None:
        chunks = chunk_text(text, max_length=500)
        if not chunks:
            return
        embeddings = self._embed_texts(chunks)
        start_index = self._collection.count()
        ids = [f"doc_{start_index + i}" for i in range(len(chunks))]
        self._collection.add(documents=chunks, embeddings=embeddings, ids=ids)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_emb = self._embed_texts([query])[0]
        results = self._collection.query(query_embeddings=[query_emb], n_results=top_k)
        return results.get("documents", [[]])[0]