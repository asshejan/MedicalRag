from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Simple in-memory vector DB for demo

class SimpleVectorDB:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.embeddings = []

    def add(self, text, embedding):
        self.texts.append(text)
        self.embeddings.append(embedding)
        self.index.add(np.array([embedding]).astype('float32'))

    def search(self, query_embedding, top_k=5):
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        return [self.texts[i] for i in I[0] if i < len(self.texts)]

def chunk_text(text, max_length=500):
    # Simple chunking by words
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunk = ' '.join(words[i:i+max_length])
        chunks.append(chunk)
    return chunks

# RAG pipeline

class RAGPipeline:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.db = SimpleVectorDB(dim=self.model.get_sentence_embedding_dimension())

    def add_document(self, text):
        # Chunk the text before embedding
        chunks = chunk_text(text, max_length=500)
        for chunk in chunks:
            if chunk.strip():
                embedding = self.model.encode(chunk)
                self.db.add(chunk, embedding)

    def retrieve(self, query, top_k=5):
        query_emb = self.model.encode(query)
        return self.db.search(query_emb, top_k=top_k)
