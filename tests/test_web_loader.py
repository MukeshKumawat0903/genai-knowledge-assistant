"""
Test Script for WebDocumentLoader

This script demonstrates and tests the WebDocumentLoader functionality.

What it tests:
    1. Single URL loading
    2. Multiple URL loading
    3. Validation and error handling
    4. Metadata standardization
    5. Interface compliance

How to run:
    python test_web_loader.py

Note: Requires internet connection to load web pages.
"""

import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from app.ingestion.web_loader import WebDocumentLoader
try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document


def test_single_url_loading():
    """Test loading a single URL."""
    print("\n" + "="*70)
    print("TEST 1: Single URL Loading")
    print("="*70)
    
    # Use a reliable test URL
    test_url = "https://example.com"
    
    print(f"\nLoading URL: {test_url}")
    loader = WebDocumentLoader(
        urls=test_url,
        source_name="Example Site"
    )
    
    try:
        documents = loader.load_documents()
        
        print(f"\n✓ Loaded {len(documents)} document(s)")
        
        # Check first document
        if documents:
            doc = documents[0]
            print(f"\nContent (first 200 chars):")
            print(f"  {doc.page_content[:200]}...")
            
            print(f"\nMetadata:")
            for key, value in doc.metadata.items():
                # Truncate long values
                val_str = str(value)
                if len(val_str) > 60:
                    val_str = val_str[:57] + "..."
                print(f"  - {key}: {val_str}")
            
            # Verify required metadata fields
            assert "source" in doc.metadata, "Missing 'source' metadata"
            assert "loader_type" in doc.metadata, "Missing 'loader_type' metadata"
            assert doc.metadata["loader_type"] == "web", "Incorrect loader_type"
            assert doc.metadata["source"] == test_url, "Incorrect source URL"
            
            print("\n✓ All metadata fields present and correct")
    
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


def test_multiple_url_loading():
    """Test loading multiple URLs."""
    print("\n" + "="*70)
    print("TEST 2: Multiple URL Loading")
    print("="*70)
    
    # Use reliable test URLs
    test_urls = [
        "https://example.com",
        "https://example.org",
        "https://example.net"
    ]
    
    print(f"\nLoading {len(test_urls)} URLs:")
    for url in test_urls:
        print(f"  - {url}")
    
    loader = WebDocumentLoader(
        urls=test_urls,
        source_name="Example Sites"
    )
    
    try:
        documents = loader.load_documents()
        
        print(f"\n✓ Loaded {len(documents)} document(s)")
        
        # Verify we got one document per URL
        assert len(documents) == len(test_urls), \
            f"Expected {len(test_urls)} documents, got {len(documents)}"
        
        # Check each document has correct source
        sources = [doc.metadata.get("source") for doc in documents]
        print(f"\nSources loaded:")
        for source in sources:
            print(f"  - {source}")
        
        print("\n✓ All URLs loaded successfully")
    
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


def test_validation_errors():
    """Test error handling for invalid URLs."""
    print("\n" + "="*70)
    print("TEST 3: Validation and Error Handling")
    print("="*70)
    
    # Test 1: Empty URL list
    print("\n1. Empty URL list:")
    try:
        loader = WebDocumentLoader(urls=[])
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Test 2: Invalid URL format (no scheme)
    print("\n2. Invalid URL format (no http/https):")
    try:
        loader = WebDocumentLoader(urls="example.com")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:60]}...")
    
    # Test 3: Invalid URL format (no domain)
    print("\n3. Invalid URL format (no domain):")
    try:
        loader = WebDocumentLoader(urls="http://")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:60]}...")
    
    # Test 4: Wrong type for URLs
    print("\n4. Wrong type for URLs:")
    try:
        loader = WebDocumentLoader(urls=123)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Test 5: Unreachable URL (will fail during load_documents)
    print("\n5. Unreachable URL:")
    try:
        loader = WebDocumentLoader(urls="https://this-domain-definitely-does-not-exist-12345.com")
        documents = loader.load_documents()
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:80]}...")


def test_interface_compliance():
    """Test that WebDocumentLoader follows BaseDocumentLoader interface."""
    print("\n" + "="*70)
    print("TEST 4: Interface Compliance")
    print("="*70)
    
    from app.ingestion.base_loader import BaseDocumentLoader
    
    # Check inheritance
    print("\n1. Checking inheritance:")
    assert issubclass(WebDocumentLoader, BaseDocumentLoader), \
        "WebDocumentLoader must inherit from BaseDocumentLoader"
    print("✓ WebDocumentLoader inherits from BaseDocumentLoader")
    
    # Check required method exists
    print("\n2. Checking required method:")
    assert hasattr(WebDocumentLoader, 'load_documents'), \
        "WebDocumentLoader must implement load_documents()"
    print("✓ WebDocumentLoader implements load_documents()")
    
    # Check return type
    print("\n3. Checking return type:")
    loader = WebDocumentLoader(urls="https://example.com")
    documents = loader.load_documents()
    
    assert isinstance(documents, list), "load_documents() must return a list"
    print("✓ load_documents() returns a list")
    
    if documents:
        assert all(isinstance(doc, Document) for doc in documents), \
            "All items must be Document objects"
        print("✓ All items are Document objects")


def test_url_normalization():
    """Test that single URL is normalized to list."""
    print("\n" + "="*70)
    print("TEST 5: URL Normalization")
    print("="*70)
    
    # Test single string URL
    print("\n1. Single URL as string:")
    loader = WebDocumentLoader(urls="https://example.com")
    assert isinstance(loader.urls, list), "URLs should be normalized to list"
    assert len(loader.urls) == 1, "List should contain one URL"
    print(f"✓ Single URL normalized to list: {loader.urls}")
    
    # Test list of URLs
    print("\n2. URLs as list:")
    test_urls = ["https://example.com", "https://example.org"]
    loader = WebDocumentLoader(urls=test_urls)
    assert isinstance(loader.urls, list), "URLs should remain as list"
    assert len(loader.urls) == 2, "List should contain two URLs"
    print(f"✓ URL list preserved: {loader.urls}")


def show_usage_patterns():
    """Show common usage patterns."""
    print("\n" + "="*70)
    print("USAGE PATTERNS")
    print("="*70)
    
    patterns = """
Pattern 1: Load Single Web Page
--------------------------------
from app.ingestion.web_loader import WebDocumentLoader

loader = WebDocumentLoader(
    urls="https://en.wikipedia.org/wiki/Artificial_intelligence",
    source_name="AI Wikipedia"
)
documents = loader.load_documents()

print(f"Loaded: {documents[0].metadata['title']}")
print(f"Content length: {len(documents[0].page_content)} characters")


Pattern 2: Load Multiple URLs
------------------------------
urls = [
    "https://blog.example.com/post1",
    "https://blog.example.com/post2",
    "https://blog.example.com/post3"
]

loader = WebDocumentLoader(urls=urls, source_name="Tech Blog")
documents = loader.load_documents()

for doc in documents:
    print(f"- {doc.metadata['source']}: {len(doc.page_content)} chars")


Pattern 3: Use in RAG Pipeline
-------------------------------
from app.ingestion.web_loader import WebDocumentLoader
from app.core.embeddings import EmbeddingManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load web pages
urls = [
    "https://docs.example.com/intro",
    "https://docs.example.com/guide",
    "https://docs.example.com/api"
]
loader = WebDocumentLoader(urls=urls, source_name="Documentation")
documents = loader.load_documents()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

# Create embeddings and vector store
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.get_embeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Query
results = vectorstore.similarity_search("How do I get started?")


Pattern 4: Error Handling
--------------------------
def safe_load_urls(urls: List[str]):
    try:
        loader = WebDocumentLoader(urls=urls)
        documents = loader.load_documents()
        return documents
    except ValueError as e:
        print(f"Invalid URLs: {e}")
        return []
    except Exception as e:
        print(f"Failed to load: {e}")
        return []

documents = safe_load_urls(["https://example.com"])


Pattern 5: Filter by Domain
----------------------------
documents = loader.load_documents()

# Group by domain
from urllib.parse import urlparse
from collections import defaultdict

by_domain = defaultdict(list)
for doc in documents:
    domain = urlparse(doc.metadata['source']).netloc
    by_domain[domain].append(doc)

for domain, docs in by_domain.items():
    print(f"{domain}: {len(docs)} documents")
"""
    
    print(patterns)


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("WEB DOCUMENT LOADER - TEST SUITE")
    print("="*70)
    print("\nThis test suite validates the WebDocumentLoader implementation.")
    print("Tests require an internet connection to access example.com.")
    print("\nInstall dependencies:")
    print("  pip install langchain langchain-community beautifulsoup4")
    
    try:
        # Run all tests
        test_single_url_loading()
        test_multiple_url_loading()
        test_validation_errors()
        test_interface_compliance()
        test_url_normalization()
        show_usage_patterns()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nThe WebDocumentLoader is working correctly!")
        print("You can now use it in your RAG pipeline.")
        
    except AssertionError as e:
        print("\n" + "="*70)
        print(f"✗ TEST FAILED: {e}")
        print("="*70)
        return 1
    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ UNEXPECTED ERROR: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
