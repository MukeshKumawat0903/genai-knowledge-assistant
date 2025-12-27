"""
Test script for VectorStoreManager

Validates vector store creation, persistence, and loading across backends.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.vector_store import VectorStoreManager
from app.utils.config import get_settings
try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document

try:
    from langchain_community.embeddings import FakeEmbeddings  # For testing without API keys
except ImportError:  # pragma: no cover
    from langchain.embeddings import FakeEmbeddings  # type: ignore


def test_vector_store_initialization():
    """Test vector store manager initialization."""
    print("=" * 70)
    print("Test 1: Vector Store Manager Initialization")
    print("=" * 70)
    
    settings = get_settings()
    embeddings = FakeEmbeddings(size=384)  # Fake embeddings for testing
    
    try:
        manager = VectorStoreManager(settings, embeddings)
        
        print(f"✓ VectorStoreManager initialized successfully")
        print(f"\nConfiguration:")
        info = manager.get_vector_store_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
    
    print()


def test_faiss_creation():
    """Test FAISS vector store creation."""
    print("=" * 70)
    print("Test 2: FAISS Vector Store Creation")
    print("=" * 70)
    
    # Create sample documents
    documents = [
        Document(
            page_content="Artificial Intelligence is transforming technology.",
            metadata={"source": "doc1.txt", "topic": "AI"}
        ),
        Document(
            page_content="Machine Learning enables computers to learn from data.",
            metadata={"source": "doc2.txt", "topic": "ML"}
        ),
        Document(
            page_content="RAG systems combine retrieval with generation.",
            metadata={"source": "doc3.txt", "topic": "RAG"}
        ),
    ]
    
    settings = get_settings()
    # Force FAISS for this test
    settings.vector_store_type = "faiss"
    
    embeddings = FakeEmbeddings(size=384)
    
    try:
        manager = VectorStoreManager(settings, embeddings)
        
        print(f"Creating FAISS store with {len(documents)} documents...")
        vectorstore = manager.create_vector_store(documents)
        
        print(f"✓ FAISS store created successfully")
        
        # Test search
        print("\nTesting similarity search...")
        results = vectorstore.similarity_search("What is machine learning?", k=2)
        
        print(f"✓ Found {len(results)} results")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Content: {doc.page_content[:60]}...")
            print(f"  Metadata: {doc.metadata}")
        
        # Test save
        print("\nSaving FAISS index...")
        manager.save_vector_store(vectorstore)
        
    except Exception as e:
        print(f"✗ FAISS test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_faiss_load():
    """Test loading existing FAISS vector store."""
    print("=" * 70)
    print("Test 3: FAISS Vector Store Loading")
    print("=" * 70)
    
    settings = get_settings()
    settings.vector_store_type = "faiss"
    embeddings = FakeEmbeddings(size=384)
    
    try:
        manager = VectorStoreManager(settings, embeddings)
        
        print("Loading existing FAISS index...")
        vectorstore = manager.load_vector_store()
        
        print(f"✓ FAISS store loaded successfully")
        
        # Test search on loaded store
        print("\nTesting search on loaded store...")
        results = vectorstore.similarity_search("AI and technology", k=2)
        
        print(f"✓ Found {len(results)} results")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Content: {doc.page_content[:60]}...")
    
    except FileNotFoundError as e:
        print(f"? Index not found (expected if test_faiss_creation not run): {e}")
    except Exception as e:
        print(f"✗ Loading failed: {e}")
    
    print()


def test_chroma_creation():
    """Test Chroma vector store creation."""
    print("=" * 70)
    print("Test 4: Chroma Vector Store Creation")
    print("=" * 70)
    
    documents = [
        Document(
            page_content="Vector databases enable semantic search.",
            metadata={"source": "vec1.txt", "topic": "Vector DB"}
        ),
        Document(
            page_content="Embeddings capture semantic meaning of text.",
            metadata={"source": "vec2.txt", "topic": "Embeddings"}
        ),
    ]
    
    settings = get_settings()
    settings.vector_store_type = "chroma"
    settings.collection_name = "test_collection"
    
    embeddings = FakeEmbeddings(size=384)
    
    try:
        manager = VectorStoreManager(settings, embeddings)
        
        print(f"Creating Chroma store with {len(documents)} documents...")
        vectorstore = manager.create_vector_store(documents)
        
        print(f"✓ Chroma store created successfully")
        
        # Test search
        print("\nTesting similarity search...")
        results = vectorstore.similarity_search("semantic search", k=2)
        
        print(f"✓ Found {len(results)} results")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Content: {doc.page_content}")
    
    except Exception as e:
        print(f"✗ Chroma test failed: {e}")
        print("Note: Chroma requires 'pip install chromadb'")
    
    print()


def test_empty_documents():
    """Test error handling for empty documents."""
    print("=" * 70)
    print("Test 5: Empty Documents Error Handling")
    print("=" * 70)
    
    settings = get_settings()
    embeddings = FakeEmbeddings(size=384)
    
    try:
        manager = VectorStoreManager(settings, embeddings)
        vectorstore = manager.create_vector_store([])
        
        print("✗ Should have raised ValueError for empty documents")
    
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"? Unexpected error: {e}")
    
    print()


def test_invalid_vector_store_type():
    """Test error handling for invalid vector store type."""
    print("=" * 70)
    print("Test 6: Invalid Vector Store Type")
    print("=" * 70)
    
    settings = get_settings()
    settings.vector_store_type = "invalid_type"
    embeddings = FakeEmbeddings(size=384)
    
    try:
        manager = VectorStoreManager(settings, embeddings)
        print("✗ Should have raised ValueError for invalid type")
    
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"? Unexpected error: {e}")
    
    print()


def test_pinecone_placeholder():
    """Test Pinecone placeholder raises NotImplementedError."""
    print("=" * 70)
    print("Test 7: Pinecone Placeholder")
    print("=" * 70)
    
    settings = get_settings()
    settings.vector_store_type = "pinecone"
    embeddings = FakeEmbeddings(size=384)
    
    documents = [
        Document(page_content="Test", metadata={"source": "test"})
    ]
    
    try:
        manager = VectorStoreManager(settings, embeddings)
        vectorstore = manager.create_vector_store(documents)
        
        print("✗ Should have raised NotImplementedError")
    
    except NotImplementedError as e:
        print(f"✓ Correctly raised NotImplementedError")
        print(f"\nMessage: {str(e)[:200]}...")
    except Exception as e:
        print(f"? Unexpected error: {e}")
    
    print()


def test_configuration_info():
    """Test getting vector store configuration info."""
    print("=" * 70)
    print("Test 8: Configuration Info")
    print("=" * 70)
    
    settings = get_settings()
    embeddings = FakeEmbeddings(size=384)
    
    try:
        manager = VectorStoreManager(settings, embeddings)
        info = manager.get_vector_store_info()
        
        print("✓ Configuration info retrieved")
        print("\nVector Store Configuration:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        required_keys = ["type", "path", "collection_name", "embeddings"]
        missing = [k for k in required_keys if k not in info]
        
        if not missing:
            print("\n✓ All required keys present")
        else:
            print(f"\n✗ Missing keys: {missing}")
    
    except Exception as e:
        print(f"✗ Failed to get info: {e}")
    
    print()


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 70)
    print("VectorStoreManager Test Suite")
    print("=" * 70 + "\n")
    
    test_vector_store_initialization()
    test_faiss_creation()
    test_faiss_load()
    test_chroma_creation()
    test_empty_documents()
    test_invalid_vector_store_type()
    test_pinecone_placeholder()
    test_configuration_info()
    
    print("=" * 70)
    print("Test Suite Complete")
    print("=" * 70)
    print("\nConfiguration:")
    settings = get_settings()
    print(f"  VECTOR_STORE_TYPE: {settings.vector_store_type}")
    print(f"  VECTOR_STORE_PATH: {settings.vector_store_path}")
    print(f"  COLLECTION_NAME: {settings.collection_name}")
    print("\nTo test different vector stores, edit .env:")
    print("  VECTOR_STORE_TYPE=faiss   # or chroma, pinecone")
    print("  VECTOR_STORE_PATH=./data/vector_store")
    print("  COLLECTION_NAME=documents")


if __name__ == "__main__":
    run_all_tests()
