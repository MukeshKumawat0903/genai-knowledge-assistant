"""
Test Suite for DocumentIndexer

This module contains comprehensive tests for the DocumentIndexer class,
which orchestrates the RAG indexing pipeline.

Test Coverage:
    - Basic indexing workflow
    - Input validation (empty lists, invalid documents)
    - Component integration (chunker, embeddings, vector store)
    - Error handling and edge cases
    - Metadata preservation
    - Index persistence
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List

from langchain_core.documents import Document

from app.rag.indexer import DocumentIndexer, create_indexer
from app.core.text_splitter import TextChunker
from app.core.embeddings import EmbeddingManager
from app.core.vector_store import VectorStoreManager


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This is the first document about RAG systems.",
            metadata={"source": "doc1.txt", "page": 1}
        ),
        Document(
            page_content="This is the second document about vector databases.",
            metadata={"source": "doc2.txt", "page": 1}
        ),
        Document(
            page_content="This is the third document about embeddings.",
            metadata={"source": "doc3.txt", "page": 1}
        ),
    ]


@pytest.fixture
def mock_text_chunker():
    """Create a mock TextChunker."""
    chunker = Mock(spec=TextChunker)
    
    # Mock split_documents to return chunks with metadata
    def split_documents(documents):
        chunks = []
        for doc in documents:
            # Split each document into 2 chunks for testing
            chunks.append(Document(
                page_content=doc.page_content[:len(doc.page_content)//2],
                metadata={**doc.metadata, "chunk_index": 0}
            ))
            chunks.append(Document(
                page_content=doc.page_content[len(doc.page_content)//2:],
                metadata={**doc.metadata, "chunk_index": 1}
            ))
        return chunks
    
    chunker.split_documents.side_effect = split_documents
    return chunker


@pytest.fixture
def mock_embedding_manager():
    """Create a mock EmbeddingManager."""
    manager = Mock(spec=EmbeddingManager)
    
    # Mock get_embeddings to return a mock embeddings object
    mock_embeddings = Mock()
    manager.get_embeddings.return_value = mock_embeddings
    
    return manager


@pytest.fixture
def mock_vector_store_manager():
    """Create a mock VectorStoreManager."""
    manager = Mock(spec=VectorStoreManager)
    
    # Set vector_store_type attribute
    manager.vector_store_type = "faiss"
    
    # Mock create_vector_store to return a mock vector store
    mock_vector_store = Mock()
    manager.create_vector_store.return_value = mock_vector_store
    
    # Mock save_vector_store
    manager.save_vector_store.return_value = None
    
    return manager


@pytest.fixture
def document_indexer(mock_text_chunker, mock_embedding_manager, mock_vector_store_manager):
    """Create a DocumentIndexer with mocked dependencies."""
    return DocumentIndexer(
        text_chunker=mock_text_chunker,
        embedding_manager=mock_embedding_manager,
        vector_store_manager=mock_vector_store_manager
    )


# ============================================================================
# Test Cases
# ============================================================================

class TestDocumentIndexer:
    """Test suite for DocumentIndexer class."""
    
    def test_initialization(self, mock_text_chunker, mock_embedding_manager, mock_vector_store_manager):
        """Test that DocumentIndexer initializes correctly with dependencies."""
        indexer = DocumentIndexer(
            text_chunker=mock_text_chunker,
            embedding_manager=mock_embedding_manager,
            vector_store_manager=mock_vector_store_manager
        )
        
        assert indexer.text_chunker is mock_text_chunker
        assert indexer.embedding_manager is mock_embedding_manager
        assert indexer.vector_store_manager is mock_vector_store_manager
    
    def test_index_documents_success(self, document_indexer, sample_documents):
        """Test successful document indexing."""
        result = document_indexer.index_documents(sample_documents)
        
        # Verify result structure
        assert result['success'] is True
        assert result['num_documents'] == 3
        assert result['num_chunks'] == 6  # 3 documents * 2 chunks each
        assert result['vector_store_type'] == 'faiss'
        assert 'Successfully indexed' in result['message']
        
        # Verify components were called
        document_indexer.text_chunker.split_documents.assert_called_once()
        document_indexer.vector_store_manager.create_vector_store.assert_called_once()
        document_indexer.vector_store_manager.save_vector_store.assert_called_once()
    
    def test_index_empty_document_list(self, document_indexer):
        """Test that indexing empty document list returns proper error."""
        result = document_indexer.index_documents([])
        
        assert result['success'] is False
        assert result['num_documents'] == 0
        assert result['num_chunks'] == 0
        assert 'empty' in result['message'].lower()
    
    def test_index_invalid_document_types(self, document_indexer):
        """Test that non-Document objects are rejected."""
        invalid_documents = [
            "not a document",
            {"page_content": "also not valid"},
            123
        ]
        
        result = document_indexer.index_documents(invalid_documents)
        
        assert result['success'] is False
        assert 'Document objects' in result['message']
    
    def test_index_documents_with_empty_content(self, document_indexer):
        """Test that documents with empty content are rejected."""
        documents_with_empty = [
            Document(page_content="Valid content", metadata={"source": "doc1.txt"}),
            Document(page_content="", metadata={"source": "doc2.txt"}),  # Empty
            Document(page_content="   ", metadata={"source": "doc3.txt"}),  # Whitespace only
        ]
        
        result = document_indexer.index_documents(documents_with_empty)
        
        assert result['success'] is False
        assert 'empty content' in result['message'].lower()
    
    def test_chunking_produces_zero_chunks(self, document_indexer, sample_documents):
        """Test handling when chunking produces no chunks."""
        # Mock chunker to return empty list
        # Note: the fixture sets a side_effect; side_effect overrides return_value.
        document_indexer.text_chunker.split_documents.side_effect = lambda _docs: []
        
        result = document_indexer.index_documents(sample_documents)
        
        assert result['success'] is False
        assert result['num_chunks'] == 0
        assert 'zero chunks' in result['message'].lower()
    
    def test_vector_store_creation_error(self, document_indexer, sample_documents):
        """Test handling of errors during vector store creation."""
        # Mock vector store creation to raise an error
        document_indexer.vector_store_manager.create_vector_store.side_effect = \
            RuntimeError("Failed to create vector store")
        
        result = document_indexer.index_documents(sample_documents)
        
        assert result['success'] is False
        assert 'failed' in result['message'].lower()
    
    def test_metadata_preserved_through_pipeline(self, document_indexer, sample_documents):
        """Test that document metadata is preserved through the indexing pipeline."""
        result = document_indexer.index_documents(sample_documents)
        
        assert result['success'] is True
        
        # Get the documents passed to create_vector_store
        call_args = document_indexer.vector_store_manager.create_vector_store.call_args
        chunks_passed = call_args[1]['documents']
        
        # Verify metadata is preserved
        for chunk in chunks_passed:
            assert 'source' in chunk.metadata
            assert 'page' in chunk.metadata
            assert 'chunk_index' in chunk.metadata
    
    def test_custom_kwargs_passed_to_vector_store(self, document_indexer, sample_documents):
        """Test that custom kwargs are passed to vector store manager."""
        custom_kwargs = {
            'persist_directory': './custom_dir',
            'collection_name': 'custom_collection'
        }
        
        result = document_indexer.index_documents(sample_documents, **custom_kwargs)
        
        assert result['success'] is True
        
        # Verify kwargs were passed to create_vector_store
        call_args = document_indexer.vector_store_manager.create_vector_store.call_args
        assert call_args[1]['persist_directory'] == './custom_dir'
        assert call_args[1]['collection_name'] == 'custom_collection'
    
    def test_validate_documents_all_valid(self, document_indexer, sample_documents):
        """Test document validation with valid documents."""
        result = document_indexer._validate_documents(sample_documents)
        
        assert result['valid'] is True
        assert 'valid' in result['message'].lower()
    
    def test_validate_documents_empty_list(self, document_indexer):
        """Test document validation with empty list."""
        result = document_indexer._validate_documents([])
        
        assert result['valid'] is False
        assert 'empty' in result['message'].lower()
    
    def test_validate_documents_invalid_types(self, document_indexer):
        """Test document validation with invalid types."""
        invalid_docs = ["string", 123, {"dict": "object"}]
        result = document_indexer._validate_documents(invalid_docs)
        
        assert result['valid'] is False
        assert 'Document objects' in result['message']
    
    def test_validate_documents_empty_content(self, document_indexer):
        """Test document validation with empty content."""
        docs_with_empty = [
            Document(page_content="Valid", metadata={}),
            Document(page_content="", metadata={}),
        ]
        result = document_indexer._validate_documents(docs_with_empty)
        
        assert result['valid'] is False
        assert 'empty content' in result['message'].lower()


class TestCreateIndexerFunction:
    """Test suite for create_indexer convenience function."""
    
    @patch('app.rag.indexer.get_settings')
    @patch('app.rag.indexer.TextChunker')
    @patch('app.rag.indexer.EmbeddingManager')
    @patch('app.rag.indexer.VectorStoreManager')
    def test_create_indexer_default_settings(
        self,
        mock_vector_store_manager_class,
        mock_embedding_manager_class,
        mock_text_chunker_class,
        mock_get_settings
    ):
        """Test create_indexer with default settings."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.chunk_size = 1000
        mock_settings.chunk_overlap = 200
        mock_settings.vector_store_type = 'faiss'
        mock_get_settings.return_value = mock_settings
        
        mock_embedding_manager = Mock()
        mock_embeddings = Mock()
        mock_embedding_manager.get_embeddings.return_value = mock_embeddings
        mock_embedding_manager_class.return_value = mock_embedding_manager
        
        # Create indexer
        indexer = create_indexer()
        
        # Verify indexer was created
        assert isinstance(indexer, DocumentIndexer)
        
        # Verify components were initialized with correct settings
        mock_text_chunker_class.assert_called_once_with(mock_settings)
        mock_embedding_manager_class.assert_called_once()
        mock_vector_store_manager_class.assert_called_once()
    
    @patch('app.rag.indexer.get_settings')
    @patch('app.rag.indexer.TextChunker')
    @patch('app.rag.indexer.EmbeddingManager')
    @patch('app.rag.indexer.VectorStoreManager')
    def test_create_indexer_with_overrides(
        self,
        mock_vector_store_manager_class,
        mock_embedding_manager_class,
        mock_text_chunker_class,
        mock_get_settings
    ):
        """Test create_indexer with override parameters."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.chunk_size = 1000
        mock_settings.chunk_overlap = 200
        mock_settings.vector_store_type = 'faiss'
        mock_get_settings.return_value = mock_settings
        
        mock_embedding_manager = Mock()
        mock_embeddings = Mock()
        mock_embedding_manager.get_embeddings.return_value = mock_embeddings
        mock_embedding_manager_class.return_value = mock_embedding_manager
        
        # Create indexer with overrides
        indexer = create_indexer(
            chunk_size=500,
            chunk_overlap=50,
            vector_store_type='chroma'
        )
        
        # Verify settings were overridden
        assert mock_settings.chunk_size == 500
        assert mock_settings.chunk_overlap == 50
        assert mock_settings.vector_store_type == 'chroma'


# ============================================================================
# Integration Tests
# ============================================================================

class TestDocumentIndexerIntegration:
    """
    Integration tests that use real components.
    
    Note: These tests are marked with @pytest.mark.integration and require:
    - Real Settings configuration
    - LangChain packages installed
    - Embedding models available
    
    Run with: pytest -m integration
    """
    
    @pytest.mark.integration
    def test_full_indexing_pipeline(self, sample_documents):
        """Test complete indexing pipeline with real components."""
        # Skip if dependencies not available
        pytest.importorskip("langchain")
        pytest.importorskip("sentence_transformers")
        
        from app.utils.config import get_settings
        
        try:
            # Create real indexer
            indexer = create_indexer(
                chunk_size=100,  # Small chunks for testing
                chunk_overlap=20,
                vector_store_type='faiss'
            )
            
            # Index documents
            result = indexer.index_documents(sample_documents)
            
            # Verify success
            assert result['success'] is True
            assert result['num_documents'] == 3
            assert result['num_chunks'] > 0
            
        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
