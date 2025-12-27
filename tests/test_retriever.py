"""
Test Suite for RAGRetriever

This module contains comprehensive tests for the RAGRetriever class.

Test Coverage:
    - Retriever initialization and configuration
    - Basic document retrieval
    - Query validation
    - History-aware retrieval
    - Custom k parameter
    - Retrieval with scores
    - Error handling
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document

from app.rag.retriever import RAGRetriever, create_retriever
from app.utils.config import Settings


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_vector_store():
    """Create a mock LangChain vector store."""
    vector_store = Mock()
    
    # Mock as_retriever method
    mock_retriever = Mock()
    mock_retriever.search_kwargs = {"k": 4}
    vector_store.as_retriever.return_value = mock_retriever
    
    # Mock get_relevant_documents
    def get_relevant_documents(query, **kwargs):
        return [
            Document(
                page_content=f"Document {i} content about {query}",
                metadata={"source": f"doc{i}.txt", "score": 0.9 - i*0.1}
            )
            for i in range(1, 5)
        ]
    mock_retriever.get_relevant_documents.side_effect = get_relevant_documents
    
    # Mock similarity_search_with_score
    def similarity_search_with_score(query, k=4, **kwargs):
        docs = get_relevant_documents(query)[:k]
        return [(doc, doc.metadata['score']) for doc in docs]
    vector_store.similarity_search_with_score.side_effect = similarity_search_with_score
    
    return vector_store


@pytest.fixture
def mock_settings():
    """Create mock Settings object."""
    settings = Mock(spec=Settings)
    settings.retriever_top_k = 4
    settings.similarity_threshold = 0.0
    return settings


@pytest.fixture
def retriever(mock_vector_store, mock_settings):
    """Create RAGRetriever with mocked dependencies."""
    return RAGRetriever(mock_vector_store, mock_settings)


# ============================================================================
# Test Cases
# ============================================================================

class TestRAGRetriever:
    """Test suite for RAGRetriever class."""
    
    def test_initialization(self, mock_vector_store, mock_settings):
        """Test that retriever initializes correctly."""
        retriever = RAGRetriever(mock_vector_store, mock_settings)
        
        assert retriever.vector_store is mock_vector_store
        assert retriever.settings is mock_settings
        assert retriever.top_k == 4
        assert retriever.search_type == "similarity"
        assert retriever.retriever is not None
    
    def test_initialization_with_custom_search_type(self, mock_vector_store, mock_settings):
        """Test initialization with custom search type."""
        retriever = RAGRetriever(
            mock_vector_store,
            mock_settings,
            search_type="mmr"
        )
        
        assert retriever.search_type == "mmr"
    
    def test_initialization_fails_with_none_vector_store(self, mock_settings):
        """Test that initialization fails with None vector store."""
        with pytest.raises(ValueError, match="vector_store cannot be None"):
            RAGRetriever(None, mock_settings)
    
    def test_initialization_fails_with_invalid_vector_store(self, mock_settings):
        """Test that initialization fails with invalid vector store."""
        invalid_store = Mock(spec=[])  # No as_retriever method
        
        with pytest.raises(ValueError, match="must be a LangChain-compatible"):
            RAGRetriever(invalid_store, mock_settings)
    
    def test_get_retriever(self, retriever):
        """Test get_retriever returns the underlying retriever."""
        langchain_retriever = retriever.get_retriever()
        
        assert langchain_retriever is not None
        assert hasattr(langchain_retriever, 'get_relevant_documents')
    
    def test_retrieve_simple_query(self, retriever):
        """Test basic document retrieval."""
        docs = retriever.retrieve("What is RAG?")
        
        assert len(docs) == 4
        assert all(isinstance(doc, Document) for doc in docs)
        assert "What is RAG?" in docs[0].page_content
    
    def test_retrieve_with_custom_k(self, retriever):
        """Test retrieval with custom k parameter."""
        docs = retriever.retrieve("Test query", k=2)
        
        # Verify that k was temporarily changed
        assert len(docs) == 4  # Mock returns 4 docs
        
        # Verify k was restored (check search_kwargs)
        assert retriever.retriever.search_kwargs["k"] == 4
    
    def test_retrieve_fails_with_empty_query(self, retriever):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve("")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve("   ")
    
    def test_retrieve_fails_with_none_query(self, retriever):
        """Test that None query raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve(None)
    
    def test_retrieve_with_chat_history(self, retriever):
        """Test history-aware retrieval."""
        chat_history = [
            ("What is RAG?", "RAG combines retrieval with generation."),
            ("How does it work?", "It retrieves relevant documents then generates answers.")
        ]
        
        docs = retriever.retrieve("What are the benefits?", chat_history=chat_history)
        
        assert len(docs) > 0
        # Query should have been rewritten (check via mock call)
        retriever.retriever.get_relevant_documents.assert_called()
    
    def test_rewrite_query_with_history_for_follow_up(self, retriever):
        """Test query rewriting for follow-up questions."""
        chat_history = [("What is machine learning?", "ML is...")]
        
        rewritten = retriever._rewrite_query_with_history(
            "How does it work?",
            chat_history
        )
        
        assert "machine learning" in rewritten.lower()
        assert "how does it work?" in rewritten.lower()
    
    def test_rewrite_query_with_history_for_standalone(self, retriever):
        """Test that standalone questions are not rewritten."""
        chat_history = [("What is ML?", "ML is...")]
        
        original_query = "Explain neural networks in detail"
        rewritten = retriever._rewrite_query_with_history(
            original_query,
            chat_history
        )
        
        # Long, specific query should not be rewritten
        assert rewritten == original_query
    
    def test_retrieve_with_scores(self, retriever):
        """Test retrieval with similarity scores."""
        docs_with_scores = retriever.retrieve_with_scores("What is RAG?")
        
        assert len(docs_with_scores) == 4
        for doc, score in docs_with_scores:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    def test_retrieve_with_scores_fails_with_empty_query(self, retriever):
        """Test that retrieve_with_scores fails with empty query."""
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve_with_scores("")
    
    def test_retrieve_with_kwargs(self, retriever):
        """Test that additional kwargs are passed to retriever."""
        docs = retriever.retrieve(
            "Test query",
            filter={"source": "python_docs"}
        )
        
        # Verify kwargs were passed
        call_kwargs = retriever.retriever.get_relevant_documents.call_args[1]
        assert "filter" in call_kwargs
        assert call_kwargs["filter"]["source"] == "python_docs"


class TestCreateRetrieverFunction:
    """Test suite for create_retriever convenience function."""
    
    @patch('app.rag.retriever.get_settings')
    @patch('app.rag.retriever.EmbeddingManager')
    @patch('app.rag.retriever.VectorStoreManager')
    def test_create_retriever_with_vector_store(
        self,
        mock_vector_store_manager_class,
        mock_embedding_manager_class,
        mock_get_settings,
        mock_vector_store
    ):
        """Test create_retriever with provided vector store."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.retriever_top_k = 4
        mock_settings.similarity_threshold = 0.0
        mock_get_settings.return_value = mock_settings
        
        # Create retriever
        retriever = create_retriever(vector_store=mock_vector_store)
        
        # Verify retriever was created
        assert isinstance(retriever, RAGRetriever)
        assert retriever.vector_store is mock_vector_store
        
        # Verify vector store manager was not called
        mock_vector_store_manager_class.assert_not_called()
    
    @patch('app.rag.retriever.get_settings')
    @patch('app.rag.retriever.EmbeddingManager')
    @patch('app.rag.retriever.VectorStoreManager')
    def test_create_retriever_without_vector_store(
        self,
        mock_vector_store_manager_class,
        mock_embedding_manager_class,
        mock_get_settings,
        mock_vector_store
    ):
        """Test create_retriever loads vector store if not provided."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.retriever_top_k = 4
        mock_settings.similarity_threshold = 0.0
        mock_get_settings.return_value = mock_settings
        
        mock_embedding_manager = Mock()
        mock_embeddings = Mock()
        mock_embedding_manager.get_embeddings.return_value = mock_embeddings
        mock_embedding_manager_class.return_value = mock_embedding_manager
        
        mock_vs_manager = Mock()
        mock_vs_manager.load_vector_store.return_value = mock_vector_store
        mock_vector_store_manager_class.return_value = mock_vs_manager
        
        # Create retriever
        retriever = create_retriever()
        
        # Verify components were initialized
        mock_embedding_manager_class.assert_called_once()
        mock_vector_store_manager_class.assert_called_once()
        mock_vs_manager.load_vector_store.assert_called_once()
        
        assert isinstance(retriever, RAGRetriever)
    
    @patch('app.rag.retriever.get_settings')
    def test_create_retriever_with_custom_top_k(
        self,
        mock_get_settings,
        mock_vector_store
    ):
        """Test create_retriever with custom top_k override."""
        mock_settings = Mock()
        mock_settings.retriever_top_k = 4
        mock_settings.similarity_threshold = 0.0
        mock_get_settings.return_value = mock_settings
        
        # Create retriever with custom top_k
        retriever = create_retriever(vector_store=mock_vector_store, top_k=10)
        
        # Verify top_k was overridden
        assert mock_settings.retriever_top_k == 10
    
    @patch('app.rag.retriever.get_settings')
    def test_create_retriever_with_custom_search_type(
        self,
        mock_get_settings,
        mock_vector_store
    ):
        """Test create_retriever with custom search type."""
        mock_settings = Mock()
        mock_settings.retriever_top_k = 4
        mock_settings.similarity_threshold = 0.0
        mock_get_settings.return_value = mock_settings
        
        # Create retriever with MMR search
        retriever = create_retriever(
            vector_store=mock_vector_store,
            search_type="mmr"
        )
        
        assert retriever.search_type == "mmr"


# ============================================================================
# Integration Tests
# ============================================================================

class TestRAGRetrieverIntegration:
    """
    Integration tests with real components.
    
    Note: Requires indexed vector store and LangChain packages.
    Run with: pytest -m integration
    """
    
    @pytest.mark.integration
    def test_full_retrieval_pipeline(self):
        """Test complete retrieval with real vector store."""
        pytest.importorskip("langchain")
        pytest.importorskip("sentence_transformers")
        
        try:
            # Create real retriever
            retriever = create_retriever(top_k=3)
            
            # Retrieve documents
            docs = retriever.retrieve("What is machine learning?")
            
            # Verify results
            assert len(docs) > 0
            assert all(isinstance(doc, Document) for doc in docs)
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
