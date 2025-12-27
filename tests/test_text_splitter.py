"""
Test script for TextChunker

Validates the text splitting engine with various scenarios.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.text_splitter import TextChunker
from app.utils.config import get_settings
try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document


def test_basic_splitting():
    """Test basic document splitting functionality."""
    print("=" * 70)
    print("Test 1: Basic Document Splitting")
    print("=" * 70)
    
    # Create sample document
    long_text = "This is a test sentence. " * 100  # ~2500 characters
    documents = [
        Document(
            page_content=long_text,
            metadata={"source": "test.txt", "doc_id": 1}
        )
    ]
    
    # Split documents
    settings = get_settings()
    chunker = TextChunker(settings)
    chunks = chunker.split_documents(documents)
    
    print(f"Original documents: {len(documents)}")
    print(f"Chunk size: {settings.chunk_size}")
    print(f"Chunk overlap: {settings.chunk_overlap}")
    print(f"Generated chunks: {len(chunks)}")
    
    # Validate chunks
    if len(chunks) > 0:
        print("\n✓ Chunks generated successfully")
        print(f"\nFirst chunk:")
        print(f"  Length: {len(chunks[0].page_content)} chars")
        print(f"  Metadata: {chunks[0].metadata}")
        print(f"  Content preview: {chunks[0].page_content[:100]}...")
    else:
        print("✗ No chunks generated")
    
    print()


def test_metadata_preservation():
    """Test that original metadata is preserved in chunks."""
    print("=" * 70)
    print("Test 2: Metadata Preservation")
    print("=" * 70)
    
    # Create document with rich metadata
    documents = [
        Document(
            page_content="Sample text content. " * 200,
            metadata={
                "source": "sample.pdf",
                "page": 5,
                "author": "John Doe",
                "doc_type": "research_paper"
            }
        )
    ]
    
    settings = get_settings()
    chunker = TextChunker(settings)
    chunks = chunker.split_documents(documents)
    
    print(f"Generated {len(chunks)} chunks")
    
    # Check first chunk metadata
    if chunks:
        first_chunk = chunks[0]
        print("\nFirst chunk metadata:")
        for key, value in first_chunk.metadata.items():
            print(f"  {key}: {value}")
        
        # Verify original metadata present
        required_keys = ["source", "page", "author", "doc_type"]
        missing = [k for k in required_keys if k not in first_chunk.metadata]
        
        if not missing:
            print("\n✓ All original metadata preserved")
        else:
            print(f"\n✗ Missing metadata: {missing}")
        
        # Check chunk-specific metadata
        if "chunk_index" in first_chunk.metadata:
            print("✓ Chunk metadata added (chunk_index)")
        else:
            print("✗ Chunk metadata missing")
    
    print()


def test_empty_document_list():
    """Test handling of empty document list."""
    print("=" * 70)
    print("Test 3: Empty Document List")
    print("=" * 70)
    
    settings = get_settings()
    chunker = TextChunker(settings)
    
    try:
        chunks = chunker.split_documents([])
        
        if len(chunks) == 0:
            print("✓ Empty list handled gracefully")
            print("  Returned: []")
        else:
            print(f"✗ Expected empty list, got {len(chunks)} chunks")
    
    except Exception as e:
        print(f"✗ Error handling empty list: {e}")
    
    print()


def test_multiple_documents():
    """Test splitting multiple documents at once."""
    print("=" * 70)
    print("Test 4: Multiple Documents")
    print("=" * 70)
    
    # Create multiple documents
    documents = [
        Document(
            page_content="Document 1 content. " * 100,
            metadata={"source": "doc1.txt", "doc_id": 1}
        ),
        Document(
            page_content="Document 2 content. " * 100,
            metadata={"source": "doc2.txt", "doc_id": 2}
        ),
        Document(
            page_content="Document 3 content. " * 100,
            metadata={"source": "doc3.txt", "doc_id": 3}
        ),
    ]
    
    settings = get_settings()
    chunker = TextChunker(settings)
    chunks = chunker.split_documents(documents)
    
    print(f"Original documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")
    
    # Verify chunks from different documents
    sources = set(chunk.metadata.get("source") for chunk in chunks)
    print(f"\nUnique sources in chunks: {len(sources)}")
    print(f"Sources: {sources}")
    
    if len(sources) == len(documents):
        print("\n✓ All documents processed")
    else:
        print(f"\n✗ Expected {len(documents)} sources, found {len(sources)}")
    
    print()


def test_chunk_size_validation():
    """Test validation of chunk size parameters."""
    print("=" * 70)
    print("Test 5: Chunk Size Validation")
    print("=" * 70)
    
    settings = get_settings()
    
    # Test 1: Normal case (should work)
    print("Test 5a: Valid parameters")
    try:
        chunker = TextChunker(settings)
        print(f"  ✓ Created chunker with chunk_size={settings.chunk_size}, "
              f"overlap={settings.chunk_overlap}")
    except ValueError as e:
        print(f"  ✗ Unexpected error: {e}")
    
    # Test 2: Invalid chunk size (requires modifying settings)
    print("\nTest 5b: Invalid chunk_size (requires manual testing)")
    print("  To test: Set CHUNK_SIZE=0 or negative in .env")
    print("  Expected: ValueError with clear message")
    
    # Test 3: Overlap >= chunk_size
    print("\nTest 5c: Invalid overlap (requires manual testing)")
    print("  To test: Set CHUNK_OVERLAP >= CHUNK_SIZE in .env")
    print("  Expected: ValueError with clear message")
    
    print()


def test_chunk_count_estimate():
    """Test chunk count estimation."""
    print("=" * 70)
    print("Test 6: Chunk Count Estimation")
    print("=" * 70)
    
    # Create document with known size
    text_length = 5000
    text = "A" * text_length
    documents = [Document(page_content=text, metadata={"source": "test"})]
    
    settings = get_settings()
    chunker = TextChunker(settings)
    
    # Get estimate
    estimate = chunker.get_chunk_count_estimate(documents)
    
    # Get actual
    actual_chunks = chunker.split_documents(documents)
    actual_count = len(actual_chunks)
    
    print(f"Text length: {text_length} characters")
    print(f"Chunk size: {settings.chunk_size}")
    print(f"Chunk overlap: {settings.chunk_overlap}")
    print(f"\nEstimated chunks: {estimate}")
    print(f"Actual chunks: {actual_count}")
    print(f"Difference: {abs(estimate - actual_count)}")
    
    # Estimate should be close (within 50%)
    if abs(estimate - actual_count) <= actual_count * 0.5:
        print("\n✓ Estimate reasonably accurate")
    else:
        print("\n? Estimate differs significantly (may be expected)")
    
    print()


def test_overlap_behavior():
    """Test that overlap prevents information loss."""
    print("=" * 70)
    print("Test 7: Overlap Behavior")
    print("=" * 70)
    
    # Create document with distinctive sentences
    sentences = [f"This is sentence number {i}." for i in range(1, 101)]
    text = " ".join(sentences)
    
    documents = [Document(page_content=text, metadata={"source": "test"})]
    
    settings = get_settings()
    chunker = TextChunker(settings)
    chunks = chunker.split_documents(documents)
    
    print(f"Generated {len(chunks)} chunks")
    print(f"Overlap: {settings.chunk_overlap} characters")
    
    # Check if consecutive chunks have overlapping content
    if len(chunks) >= 2:
        chunk1_end = chunks[0].page_content[-100:]  # Last 100 chars
        chunk2_start = chunks[1].page_content[:100]  # First 100 chars
        
        # Look for any common substring
        has_overlap = any(
            chunk1_end[i:i+20] in chunk2_start
            for i in range(len(chunk1_end) - 20)
        )
        
        if has_overlap:
            print("✓ Overlap detected between consecutive chunks")
        else:
            print("? No obvious overlap found (may depend on split points)")
    
    print()


def test_small_documents():
    """Test handling of documents smaller than chunk size."""
    print("=" * 70)
    print("Test 8: Small Documents (< chunk_size)")
    print("=" * 70)
    
    # Create very small document
    small_text = "This is a very small document."
    documents = [
        Document(page_content=small_text, metadata={"source": "small.txt"})
    ]
    
    settings = get_settings()
    chunker = TextChunker(settings)
    chunks = chunker.split_documents(documents)
    
    print(f"Document length: {len(small_text)} characters")
    print(f"Chunk size: {settings.chunk_size} characters")
    print(f"Generated chunks: {len(chunks)}")
    
    if len(chunks) == 1:
        print("✓ Small document returned as single chunk")
        print(f"  Content: '{chunks[0].page_content}'")
    else:
        print(f"? Generated {len(chunks)} chunks (expected 1)")
    
    print()


def test_chunk_metadata_tracking():
    """Test chunk_index and total_chunks metadata."""
    print("=" * 70)
    print("Test 9: Chunk Metadata Tracking")
    print("=" * 70)
    
    # Create document that will split into multiple chunks
    documents = [
        Document(
            page_content="Test content. " * 500,  # ~7000 chars
            metadata={"source": "test.txt"}
        )
    ]
    
    settings = get_settings()
    chunker = TextChunker(settings)
    chunks = chunker.split_documents(documents, add_chunk_metadata=True)
    
    print(f"Generated {len(chunks)} chunks")
    
    # Check first and last chunk
    if chunks:
        first = chunks[0]
        last = chunks[-1]
        
        print("\nFirst chunk metadata:")
        print(f"  chunk_index: {first.metadata.get('chunk_index')}")
        print(f"  total_chunks: {first.metadata.get('total_chunks')}")
        print(f"  chunk_size: {first.metadata.get('chunk_size')}")
        
        print("\nLast chunk metadata:")
        print(f"  chunk_index: {last.metadata.get('chunk_index')}")
        print(f"  total_chunks: {last.metadata.get('total_chunks')}")
        print(f"  chunk_size: {last.metadata.get('chunk_size')}")
        
        # Validate
        if (first.metadata.get('chunk_index') == 0 and
            last.metadata.get('chunk_index') == len(chunks) - 1 and
            first.metadata.get('total_chunks') == len(chunks)):
            print("\n✓ Chunk metadata correct")
        else:
            print("\n✗ Chunk metadata incorrect")
    
    print()


def test_without_chunk_metadata():
    """Test splitting without adding chunk metadata."""
    print("=" * 70)
    print("Test 10: Without Chunk Metadata")
    print("=" * 70)
    
    documents = [
        Document(
            page_content="Test content. " * 200,
            metadata={"source": "test.txt"}
        )
    ]
    
    settings = get_settings()
    chunker = TextChunker(settings)
    chunks = chunker.split_documents(documents, add_chunk_metadata=False)
    
    print(f"Generated {len(chunks)} chunks")
    
    if chunks:
        first_chunk = chunks[0]
        print("\nFirst chunk metadata:")
        for key, value in first_chunk.metadata.items():
            print(f"  {key}: {value}")
        
        # Check if chunk_index is absent
        if "chunk_index" not in first_chunk.metadata:
            print("\n✓ Chunk metadata not added (as requested)")
        else:
            print("\n✗ Chunk metadata was added (unexpected)")
    
    print()


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 70)
    print("TextChunker Test Suite")
    print("=" * 70 + "\n")
    
    test_basic_splitting()
    test_metadata_preservation()
    test_empty_document_list()
    test_multiple_documents()
    test_chunk_size_validation()
    test_chunk_count_estimate()
    test_overlap_behavior()
    test_small_documents()
    test_chunk_metadata_tracking()
    test_without_chunk_metadata()
    
    print("=" * 70)
    print("Test Suite Complete")
    print("=" * 70)
    print("\nConfiguration used:")
    settings = get_settings()
    print(f"  CHUNK_SIZE: {settings.chunk_size}")
    print(f"  CHUNK_OVERLAP: {settings.chunk_overlap}")
    print("\nTo modify configuration, edit .env file:")
    print("  CHUNK_SIZE=1000")
    print("  CHUNK_OVERLAP=200")


if __name__ == "__main__":
    run_all_tests()
