import os
import time
from typing import List
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

class RAGPipelinePinecone:
    """RAG pipeline using OpenAI embeddings and Pinecone for persistent storage."""
    def __init__(self, index_name: str = "medical"):
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._embed_model = "text-embedding-3-small"
        self._pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        
        try:
            # Check if index exists
            if index_name in self._pc.list_indexes().names():
                # Get index description to check dimensions
                index_info = self._pc.describe_index(index_name)
                current_dim = index_info.dimension
                
                # Only recreate if dimensions don't match AND it's a critical mismatch
                if current_dim != 1536:
                    print(f"Warning: Index {index_name} has dimensions {current_dim}, expected 1536.")
                    print(f"This may cause compatibility issues. Consider recreating the index manually if needed.")
                    # Don't auto-delete existing indexes with data
                else:
                    print(f"Using existing index {index_name} with correct dimensions")
            else:
                # Create new index if it doesn't exist
                print(f"Creating new index {index_name} with dimension 1536...")
                self._create_index(index_name)
            
        except Exception as e:
            print(f"Error during index setup: {str(e)}")
            raise
            
        self._index = self._pc.Index(index_name)
        print(f"Successfully initialized index {index_name}")

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Validate and clean input texts
        if not texts:
            raise ValueError("Input must be a non-empty list of strings")
            
        # Clean and validate each text
        cleaned_texts = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError(f"All inputs must be strings, got {type(text)}")
            cleaned = text.strip()
            if cleaned:
                cleaned_texts.append(cleaned)
                
        if not cleaned_texts:
            raise ValueError("No valid text chunks to embed after cleaning")
        
        # Create embeddings in batches of 100
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i:i + batch_size]
            response = self._client.embeddings.create(
                model=self._embed_model,
                input=batch  # OpenAI expects a list of strings
            )
            all_embeddings.extend([d.embedding for d in response.data])
            
        return all_embeddings

    def chunk_text(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into chunks of approximately max_length words."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1  # +1 for space
            if current_length > max_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def _generate_document_fingerprint(self, text: str) -> str:
        """Generate a fingerprint for a document to identify duplicates."""
        import hashlib
        # Create a hash of the first 1000 characters to use as a fingerprint
        # This helps identify if the same document is being uploaded multiple times
        return hashlib.md5(text[:1000].encode()).hexdigest()
    
    def document_exists(self, text: str) -> bool:
        """Check if a document with similar content already exists in the index."""
        if not text or not text.strip():
            return False
            
        # Generate fingerprint from the first part of the document
        fingerprint = self._generate_document_fingerprint(text)
        
        try:
            # Create a query vector from the first chunk of text
            first_chunk = self.chunk_text(text)[0] if len(text) > 100 else text
            query_emb = self._embed_texts([first_chunk])[0]
            
            # Query with high similarity threshold
            results = self._index.query(
                vector=query_emb,
                top_k=5,
                include_metadata=True
            )
            
            # Check if any results have matching fingerprint in metadata
            for match in results.matches:
                if match.score > 0.95:  # High similarity threshold
                    stored_fingerprint = match.metadata.get("fingerprint")
                    if stored_fingerprint and stored_fingerprint == fingerprint:
                        print(f"Document already exists with fingerprint: {fingerprint}")
                        return True
                        
            return False
        except Exception as e:
            print(f"Error checking document existence: {str(e)}")
            return False
    
    def add_document(self, text: str, user_id: str = None, session_id: str = None, metadata: dict = None) -> dict:
        """Add a document to the RAG pipeline and return processing statistics."""
        # Validate input
        if not isinstance(text, str):
            raise ValueError(f"Input must be a string, got {type(text)}")
            
        text = text.strip()
        if not text:
            print("Empty text provided, nothing to process.")
            return {"chunks_processed": 0, "total_vectors": 0, "status": "empty_input"}
            
        try:
            # Create chunks
            print("Chunking text...")
            chunks = self.chunk_text(text)
            if not chunks:
                print("No valid chunks to embed after processing.")
                return {"chunks_processed": 0, "total_vectors": 0, "status": "no_chunks"}
                
            print(f"Processing {len(chunks)} chunks...")
            print(f"First chunk preview: {chunks[0][:100]}...")
            
            # Get embeddings
            print("Generating embeddings...")
            embeddings = self._embed_texts(chunks)
            print(f"Generated {len(embeddings)} embeddings")
            
            # Generate document fingerprint
            fingerprint = self._generate_document_fingerprint(text)
            
            # Create unique IDs
            ids = [f"doc_{i}_{os.urandom(4).hex()}" for i in range(len(chunks))]
            
            # Prepare vectors with metadata
            vectors = []
            for id, emb, chunk in zip(ids, embeddings, chunks):
                chunk_metadata = {
                    "text": chunk,
                    "fingerprint": fingerprint  # Add fingerprint to identify duplicates
                }
                if user_id:
                    chunk_metadata["user_id"] = user_id
                if session_id:
                    chunk_metadata["session_id"] = session_id
                # Add any additional metadata if provided
                if metadata:
                    chunk_metadata.update(metadata)
                vectors.append((id, emb, chunk_metadata))
            
            # Upsert to Pinecone in batches
            total_vectors = 0
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                print(f"Upserting batch {i//batch_size + 1} of {(len(vectors)-1)//batch_size + 1}...")
                self._index.upsert(vectors=batch)
                total_vectors += len(batch)
                
            stats = {
                "chunks_processed": len(chunks),
                "total_vectors": total_vectors,
                "status": "success",
                "first_chunk_size": len(chunks[0]) if chunks else 0
            }
            print(f"Successfully processed and stored document: {stats}")
            return stats
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")

    def retrieve(self, query: str, top_k: int = 5, user_id: str = None, session_id: str = None) -> List[str]:
        """Retrieve relevant text chunks based on query similarity."""
        print(f"Retrieving for query: '{query}' with top_k={top_k}")
        
        try:
            query_emb = self._embed_texts([query])[0]
            print(f"Generated query embedding with dimension: {len(query_emb)}")
            
            # Build filter for user-specific retrieval
            filter_dict = {}
            if user_id:
                filter_dict["user_id"] = user_id
            if session_id:
                filter_dict["session_id"] = session_id
            
            # Query with more detailed results
            query_params = {
                "vector": query_emb, 
                "top_k": top_k, 
                "include_metadata": True,
                "include_values": False  # We don't need the actual vector values
            }
            
            # Add filter if user_id or session_id is provided
            if filter_dict:
                query_params["filter"] = filter_dict
                print(f"Using filter: {filter_dict}")
            
            results = self._index.query(**query_params)
            
            print(f"Found {len(results.matches)} matches")
            
            # Log match details for debugging
            for i, match in enumerate(results.matches):
                print(f"Match {i+1}: Score={match.score:.4f}, Text preview: {match.metadata.get('text', 'No text')[:100]}...")
            
            # Extract text chunks
            retrieved_texts = []
            for match in results.matches:
                if 'text' in match.metadata:
                    retrieved_texts.append(match.metadata["text"])
                else:
                    print(f"Warning: Match {match.id} has no 'text' metadata")
            
            print(f"Successfully retrieved {len(retrieved_texts)} text chunks")
            return retrieved_texts
            
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            return []
    
    def retrieve_with_filter(self, query: str, filter_dict: dict, top_k: int = 5) -> List[tuple]:
        """Retrieve relevant text chunks based on query similarity with metadata filtering.
        Returns a list of tuples (text, metadata) for each match."""
        print(f"Retrieving for query: '{query}' with filter: {filter_dict}, top_k={top_k}")
        
        try:
            query_emb = self._embed_texts([query])[0]
            print(f"Generated query embedding with dimension: {len(query_emb)}")
            
            # Query with filter
            query_params = {
                "vector": query_emb, 
                "top_k": top_k, 
                "include_metadata": True,
                "include_values": False,
                "filter": filter_dict
            }
            
            results = self._index.query(**query_params)
            
            print(f"Found {len(results.matches)} matches with filter")
            
            # Extract text chunks and metadata
            retrieved_items = []
            for match in results.matches:
                text = match.metadata.get("text", "")
                retrieved_items.append((text, match.metadata))
            
            print(f"Successfully retrieved {len(retrieved_items)} items with filter")
            return retrieved_items
            
        except Exception as e:
            print(f"Error during filtered retrieval: {str(e)}")
            return []
    
    def get_index_stats(self) -> dict:
        """Get statistics about the current index."""
        try:
            stats = self._index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        except Exception as e:
            print(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}
    
    def check_index_health(self) -> dict:
        """Check if the index is healthy and accessible."""
        try:
            # Try a simple query to test index health
            test_query = [0.0] * 1536  # Zero vector for testing
            results = self._index.query(vector=test_query, top_k=1, include_metadata=False)
            
            stats = self.get_index_stats()
            return {
                "status": "healthy",
                "accessible": True,
                "stats": stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "accessible": False,
                "error": str(e)
            }
