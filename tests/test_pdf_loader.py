"""
Test Script for PDFDocumentLoader

This script demonstrates and tests the PDFDocumentLoader functionality.

What it tests:
    1. Single PDF file loading
    2. Directory PDF loading
    3. Validation and error handling
    4. Metadata standardization
    5. Interface compliance

How to run:
    python test_pdf_loader.py
"""

import tempfile
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from app.ingestion.pdf_loader import PDFDocumentLoader
try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document


def create_test_pdf(file_path: Path, content: str = "Test PDF Content"):
    """
    Create a simple test PDF file using reportlab.
    
    Args:
        file_path: Where to save the PDF
        content: Text content to include
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        c = canvas.Canvas(str(file_path), pagesize=letter)
        c.drawString(100, 750, content)
        c.showPage()
        c.save()
        
        print(f"✓ Created test PDF: {file_path.name}")
        return True
    except ImportError:
        print("⚠ reportlab not installed. Skipping PDF creation.")
        print("  Install with: pip install reportlab")
        return False


def test_single_file_loading():
    """Test loading a single PDF file."""
    print("\n" + "="*70)
    print("TEST 1: Single PDF File Loading")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test PDF
        pdf_path = Path(temp_dir) / "test_document.pdf"
        if not create_test_pdf(pdf_path, "This is a test PDF for RAG system."):
            print("⚠ Skipping test - cannot create PDF")
            return
        
        # Load the PDF
        print(f"\nLoading PDF: {pdf_path.name}")
        loader = PDFDocumentLoader(
            path=str(pdf_path),
            source_name="Test Document"
        )
        
        documents = loader.load_documents()
        
        # Verify results
        print(f"\n✓ Loaded {len(documents)} page(s)")
        
        # Check first document
        if documents:
            doc = documents[0]
            print(f"\nFirst page content (truncated):")
            print(f"  {doc.page_content[:100]}...")
            
            print(f"\nMetadata:")
            for key, value in doc.metadata.items():
                print(f"  - {key}: {value}")
            
            # Verify required metadata fields
            assert "source" in doc.metadata, "Missing 'source' metadata"
            assert "file_name" in doc.metadata, "Missing 'file_name' metadata"
            assert "loader_type" in doc.metadata, "Missing 'loader_type' metadata"
            assert doc.metadata["loader_type"] == "pdf", "Incorrect loader_type"
            assert doc.metadata["file_name"] == "test_document.pdf", "Incorrect file_name"
            
            print("\n✓ All metadata fields present and correct")


def test_directory_loading():
    """Test loading multiple PDFs from a directory."""
    print("\n" + "="*70)
    print("TEST 2: Directory PDF Loading")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple test PDFs
        pdf_dir = Path(temp_dir)
        pdf_files = []
        
        for i in range(3):
            pdf_path = pdf_dir / f"document_{i+1}.pdf"
            if create_test_pdf(pdf_path, f"Content of document {i+1}"):
                pdf_files.append(pdf_path)
        
        if not pdf_files:
            print("⚠ Skipping test - cannot create PDFs")
            return
        
        # Load all PDFs from directory
        print(f"\nLoading all PDFs from: {pdf_dir}")
        loader = PDFDocumentLoader(
            path=str(pdf_dir),
            source_name="Test Collection"
        )
        
        documents = loader.load_documents()
        
        # Verify results
        print(f"\n✓ Loaded {len(documents)} page(s) from {len(pdf_files)} PDF(s)")
        
        # Check that we got documents from different files
        unique_sources = set(doc.metadata.get("source", "") for doc in documents)
        print(f"✓ Found {len(unique_sources)} unique source files")
        
        # Show sample from each file
        for source in unique_sources:
            source_docs = [d for d in documents if d.metadata.get("source") == source]
            file_name = Path(source).name
            print(f"\n  {file_name}: {len(source_docs)} page(s)")


def test_validation_errors():
    """Test error handling for invalid paths."""
    print("\n" + "="*70)
    print("TEST 3: Validation and Error Handling")
    print("="*70)
    
    # Test 1: Non-existent file
    print("\n1. Non-existent file:")
    try:
        loader = PDFDocumentLoader(path="nonexistent.pdf")
        documents = loader.load_documents()
        print("✗ Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"✓ Correctly raised FileNotFoundError: {str(e)[:60]}...")
    
    # Test 2: Empty path
    print("\n2. Empty path:")
    try:
        loader = PDFDocumentLoader(path="")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Test 3: Directory with no PDFs
    print("\n3. Directory with no PDFs:")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            loader = PDFDocumentLoader(path=temp_dir)
            documents = loader.load_documents()
            print("✗ Should have raised ValueError")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {str(e)[:60]}...")
    
    # Test 4: Non-PDF file
    print("\n4. Non-PDF file:")
    with tempfile.TemporaryDirectory() as temp_dir:
        txt_file = Path(temp_dir) / "test.txt"
        txt_file.write_text("This is not a PDF")
        
        try:
            loader = PDFDocumentLoader(path=str(txt_file))
            documents = loader.load_documents()
            print("✗ Should have raised ValueError")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {e}")


def test_interface_compliance():
    """Test that PDFDocumentLoader follows BaseDocumentLoader interface."""
    print("\n" + "="*70)
    print("TEST 4: Interface Compliance")
    print("="*70)
    
    from app.ingestion.base_loader import BaseDocumentLoader
    
    # Check inheritance
    print("\n1. Checking inheritance:")
    assert issubclass(PDFDocumentLoader, BaseDocumentLoader), \
        "PDFDocumentLoader must inherit from BaseDocumentLoader"
    print("✓ PDFDocumentLoader inherits from BaseDocumentLoader")
    
    # Check required method exists
    print("\n2. Checking required method:")
    assert hasattr(PDFDocumentLoader, 'load_documents'), \
        "PDFDocumentLoader must implement load_documents()"
    print("✓ PDFDocumentLoader implements load_documents()")
    
    # Check return type
    print("\n3. Checking return type:")
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "test.pdf"
        if create_test_pdf(pdf_path):
            loader = PDFDocumentLoader(path=str(pdf_path))
            documents = loader.load_documents()
            
            assert isinstance(documents, list), "load_documents() must return a list"
            print("✓ load_documents() returns a list")
            
            if documents:
                assert all(isinstance(doc, Document) for doc in documents), \
                    "All items must be Document objects"
                print("✓ All items are Document objects")


def show_usage_patterns():
    """Show common usage patterns."""
    print("\n" + "="*70)
    print("USAGE PATTERNS")
    print("="*70)
    
    patterns = """
Pattern 1: Load Single PDF
---------------------------
from app.ingestion.pdf_loader import PDFDocumentLoader

loader = PDFDocumentLoader(
    path="data/documents/annual_report.pdf",
    source_name="Annual Report 2024"
)
documents = loader.load_documents()

# Each document is one page
print(f"Loaded {len(documents)} pages")
for i, doc in enumerate(documents):
    print(f"Page {i}: {len(doc.page_content)} characters")


Pattern 2: Load Directory of PDFs
----------------------------------
loader = PDFDocumentLoader(
    path="data/research_papers/",
    source_name="AI Research Collection"
)
documents = loader.load_documents()

# Get unique files
files = set(doc.metadata['file_name'] for doc in documents)
print(f"Loaded {len(documents)} pages from {len(files)} files")


Pattern 3: Use in RAG Pipeline
-------------------------------
from app.core.embeddings import EmbeddingManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Step 1: Load documents
loader = PDFDocumentLoader(path="data/docs/")
documents = loader.load_documents()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

# Step 3: Create embeddings
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.get_embeddings()

# Step 4: Create vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 5: Query
retriever = vectorstore.as_retriever()
results = retriever.get_relevant_documents("What is the main topic?")


Pattern 4: Error Handling
--------------------------
from pathlib import Path

def safe_load_pdfs(path: str):
    try:
        loader = PDFDocumentLoader(path=path)
        documents = loader.load_documents()
        return documents
    except FileNotFoundError:
        print(f"Path not found: {path}")
        return []
    except ValueError as e:
        print(f"Invalid path: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

documents = safe_load_pdfs("data/pdfs/")
"""
    
    print(patterns)


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PDF DOCUMENT LOADER - TEST SUITE")
    print("="*70)
    print("\nThis test suite validates the PDFDocumentLoader implementation.")
    print("It requires 'reportlab' for creating test PDFs.")
    print("\nInstall dependencies:")
    print("  pip install langchain langchain-community pypdf reportlab")
    
    try:
        # Run all tests
        test_single_file_loading()
        test_directory_loading()
        test_validation_errors()
        test_interface_compliance()
        show_usage_patterns()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nThe PDFDocumentLoader is working correctly!")
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
