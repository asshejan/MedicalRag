from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def chunk_text(text, max_length=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunk = ' '.join(words[i:i+max_length])
        chunks.append(chunk)
    return chunks

class RAGPipeline:
    def __init__(self, model_name='all-MiniLM-L6-v2', collection_name="rag_collection"):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.Client(Settings(
            persist_directory=".chromadb"  # You can change this path as needed
        ))
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_document(self, text):
        chunks = chunk_text(text, max_length=500)
        embeddings = self.model.encode(chunks)
        ids = [f"doc_{self.collection.count()}_{i}" for i in range(len(chunks))]
        # Add chunks and embeddings to ChromaDB
        self.collection.add(
            documents=chunks,
            embeddings=[emb.tolist() for emb in embeddings],
            ids=ids
        )

    def retrieve(self, query, top_k=5):
        query_emb = self.model.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        # results['documents'] is a list of lists (one per query)
        return results['documents'][0] if results['documents'] else []