import os
import pytest
from unittest.mock import Mock, patch
from app.services.rag_pipeline_pinecone import RAGPipelinePinecone

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables needed for testing"""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_key',
        'PINECONE_API_KEY': 'test_key',
        'PINECONE_ENVIRONMENT': 'test-aws'
    }):
        yield

@pytest.fixture
def mock_openai():
    with patch('openai.OpenAI') as mock_class:
        def create_embeddings(**kwargs):
            # Create a mock response with the same number of embeddings as input texts
            texts = kwargs.get('input', [])
            return Mock(data=[
                Mock(embedding=[0.1] * 1536) for _ in range(len(texts))
            ])
            
        mock_instance = mock_class.return_value
        mock_instance.embeddings.create.side_effect = create_embeddings
        yield mock_class

@pytest.fixture
def mock_pinecone():
    with patch('pinecone.Pinecone') as mock_class:
        mock_instance = mock_class.return_value
        
        # Mock index listing
        mock_instance.list_indexes.return_value.names.return_value = []
        
        # Mock Index creation and operations
        mock_index = Mock()
        mock_index.upsert = Mock()
        mock_index.query = Mock(return_value=Mock(
            matches=[
                Mock(metadata={"text": "test chunk 1"}),
                Mock(metadata={"text": "test chunk 2"})
            ]
        ))
        
        mock_instance.Index.return_value = mock_index
        
        # Mock other needed methods
        mock_instance.create_index = Mock()
        
        yield mock_class

def test_chunk_text(mock_openai, mock_pinecone):
    # Initialize pipeline with mocked dependencies
    with patch('app.services.rag_pipeline_pinecone.OpenAI', mock_openai), \
         patch('app.services.rag_pipeline_pinecone.Pinecone', mock_pinecone):
        pipeline = RAGPipelinePinecone()
        
        # Test with simple text
        text = "This is a test document. It should be split into chunks properly."
        chunks = pipeline.chunk_text(text, max_length=5)
        assert len(chunks) > 0
        assert all(len(chunk.split()) <= 5 for chunk in chunks)
        
        # Test with empty text
        empty_chunks = pipeline.chunk_text("", max_length=500)
        assert len(empty_chunks) == 0

def test_embed_texts(mock_openai, mock_pinecone):
    # Initialize pipeline with mocked dependencies
    with patch('app.services.rag_pipeline_pinecone.OpenAI', mock_openai), \
         patch('app.services.rag_pipeline_pinecone.Pinecone', mock_pinecone):
        pipeline = RAGPipelinePinecone()
        
        # Test with valid input
        texts = ["This is a test sentence.", "This is another test sentence."]
        embeddings = pipeline._embed_texts(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536  # text-embedding-3-small dimension
        
        # Verify OpenAI was called correctly
        mock_openai.return_value.embeddings.create.assert_called_with(
            model=pipeline._embed_model,
            input=texts
        )
        
        # Test with invalid input
        with pytest.raises(ValueError):
            pipeline._embed_texts([])  # Empty list
        
        with pytest.raises(ValueError):
            pipeline._embed_texts([None])  # Invalid input type
        
        with pytest.raises(ValueError):
            pipeline._embed_texts(["", "  "])  # Empty strings

def test_add_document(mock_openai, mock_pinecone):
    # Initialize pipeline with mocked dependencies
    with patch('app.services.rag_pipeline_pinecone.OpenAI', mock_openai), \
         patch('app.services.rag_pipeline_pinecone.Pinecone', mock_pinecone):
        pipeline = RAGPipelinePinecone()
        
        # Test with sample medical text
        sample_text = """
        The human heart is a muscular organ that pumps blood through the body.
        It is composed of four chambers: two atria and two ventricles.
        The right atrium receives deoxygenated blood from the body.
        The left ventricle pumps oxygenated blood to the rest of the body.
        """
        
        # Configure mock response for retrieval
        mock_pinecone.return_value.Index.return_value.query.return_value = Mock(
            matches=[
                Mock(metadata={"text": "It is composed of four chambers: two atria and two ventricles."}),
                Mock(metadata={"text": "The right atrium receives deoxygenated blood from the body."})
            ]
        )
        
        # Test document addition
        pipeline.add_document(sample_text)
        
        # Verify document was processed and stored
        assert mock_openai.return_value.embeddings.create.called
        assert mock_pinecone.return_value.Index.return_value.upsert.called
        
        # Test retrieval
        query = "What are the chambers of the heart?"
        results = pipeline.retrieve(query, top_k=2)
        assert len(results) == 2
        assert any("chambers" in result.lower() for result in results)
        
        # Test with invalid input
        with pytest.raises(ValueError):
            pipeline.add_document(None)
            
            # Test with empty text
            # Reset the mock's call history
            mock_pinecone.return_value.Index.return_value.upsert.reset_mock()
            pipeline.add_document("")
            assert not mock_pinecone.return_value.Index.return_value.upsert.called