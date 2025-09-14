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
        # Validate input
        if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("Input must be a non-empty list of non-empty strings")
        
        # Clean and validate each text chunk
        cleaned_texts = [t.strip() for t in texts if t and t.strip()]
        if not cleaned_texts:
            raise ValueError("No valid text chunks to embed after cleaning")
            
        response = self._client.embeddings.create(
            model=self._embed_model, input=cleaned_texts
        )
        return [d.embedding for d in response.data]

    def add_document(self, text: str) -> None:
        # Validate input
        if not isinstance(text, str) or not text.strip():
            print("Invalid input: text must be a non-empty string")
            return
            
        # Create and validate chunks
        chunks = chunk_text(text, max_length=500)
        if not chunks:
            print("No valid chunks to process after splitting text")
            return
            
        try:
            print(f"Processing {len(chunks)} chunks...")
            embeddings = self._embed_texts(chunks)
            start_index = self._collection.count()
            ids = [f"doc_{start_index + i}" for i in range(len(chunks))]
            self._collection.add(documents=chunks, embeddings=embeddings, ids=ids)
            print(f"Successfully added {len(chunks)} document chunks")
        except ValueError as e:
            print(f"Error processing document: {e}")
        except Exception as e:
            print(f"Unexpected error while adding document: {e}")

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_emb = self._embed_texts([query])[0]
        results = self._collection.query(query_embeddings=[query_emb], n_results=top_k)
        return results.get("documents", [[]])[0]