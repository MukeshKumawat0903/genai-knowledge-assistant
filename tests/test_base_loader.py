"""
Base Document Loader Test/Demo Script

This script demonstrates how to create a concrete loader by inheriting from
BaseDocumentLoader and implementing the required interface.

Usage:
    python test_base_loader.py
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.ingestion.base_loader import BaseDocumentLoader


class SimpleTextLoader(BaseDocumentLoader):
    """
    Example concrete implementation of BaseDocumentLoader.
    
    This loader demonstrates how to:
    1. Inherit from BaseDocumentLoader
    2. Implement load_documents()
    3. Use helper methods (_validate_source, _add_metadata)
    4. Return proper LangChain Document objects
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the text loader.
        
        Args:
            file_path: Path to the text file to load
        """
        self.file_path = file_path
    
    def load_documents(self) -> List[Document]:
        """
        Load a text file and return as a Document.
        
        Returns:
            List[Document]: Single document containing file content
        """
        # Step 1: Validate source
        self._validate_source(self.file_path)
        
        # Step 2: Load content
        path = Path(self.file_path)
        content = path.read_text(encoding='utf-8')
        
        # Step 3: Create Document with metadata
        doc = Document(
            page_content=content,
            metadata=self._add_metadata({
                "source": self.file_path,
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "encoding": "utf-8"
            })
        )
        
        # Step 4: Return list of documents
        return [doc]
    
    def _validate_source(self, source: str) -> None:
        """
        Validate that the file exists and is readable.
        
        Args:
            source: File path to validate
            
        Raises:
            ValueError: If source is empty or invalid
            FileNotFoundError: If file doesn't exist
        """
        if not source:
            raise ValueError("File path cannot be empty")
        
        path = Path(source)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {source}")
        
        if not path.suffix in ['.txt', '.md']:
            raise ValueError(f"Unsupported file type: {path.suffix}")


class SimpleWebLoader(BaseDocumentLoader):
    """
    Example web loader (simulated - not doing actual HTTP requests).
    
    Demonstrates how different loader types can implement the same interface.
    """
    
    def __init__(self, url: str):
        """Initialize with URL."""
        self.url = url
    
    def load_documents(self) -> List[Document]:
        """
        Simulate loading a webpage.
        
        Returns:
            List[Document]: Single document with simulated content
        """
        # Validate
        self._validate_source(self.url)
        
        # Simulate loading (in real implementation, use requests/beautifulsoup)
        simulated_content = f"This is simulated content from {self.url}"
        
        # Create document
        doc = Document(
            page_content=simulated_content,
            metadata=self._add_metadata({
                "source": self.url,
                "source_type": "web",
                "title": "Example Page"
            })
        )
        
        return [doc]
    
    def _validate_source(self, source: str) -> None:
        """Validate URL format."""
        if not source:
            raise ValueError("URL cannot be empty")
        
        if not source.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL: {source}")


def test_text_loader():
    """Test the SimpleTextLoader."""
    print("=" * 70)
    print("Testing SimpleTextLoader (File-based)")
    print("=" * 70)
    print()
    
    # Create a test file
    test_file = Path("test_document.txt")
    test_content = """This is a test document for the GenAI Knowledge Assistant.

It demonstrates how the BaseDocumentLoader interface works.

Key concepts:
- Abstract base classes define interfaces
- Concrete loaders implement the interface
- All loaders return LangChain Document objects
- Metadata is standardized across loaders
"""
    
    test_file.write_text(test_content, encoding='utf-8')
    print(f"✅ Created test file: {test_file}")
    print()
    
    try:
        # Initialize loader
        print("📋 Initializing SimpleTextLoader...")
        loader = SimpleTextLoader(str(test_file))
        print(f"   Source: {test_file}")
        print()
        
        # Load documents
        print("📥 Loading documents...")
        documents = loader.load_documents()
        print(f"✅ Loaded {len(documents)} document(s)")
        print()
        
        # Display results
        doc = documents[0]
        print("📄 Document Details:")
        print(f"   Content length: {len(doc.page_content)} characters")
        print(f"   Content preview: {doc.page_content[:100]}...")
        print()
        
        print("🏷️  Metadata:")
        for key, value in doc.metadata.items():
            print(f"   • {key}: {value}")
        print()
        
        # Test validation
        print("=" * 70)
        print("Testing Validation")
        print("=" * 70)
        print()
        
        print("❌ Testing with non-existent file...")
        try:
            bad_loader = SimpleTextLoader("nonexistent.txt")
            bad_loader.load_documents()
            print("   ERROR: Should have raised FileNotFoundError")
        except FileNotFoundError as e:
            print(f"   ✅ Correctly raised: {type(e).__name__}")
            print(f"   Message: {e}")
        print()
        
        print("❌ Testing with empty path...")
        try:
            bad_loader = SimpleTextLoader("")
            bad_loader.load_documents()
            print("   ERROR: Should have raised ValueError")
        except ValueError as e:
            print(f"   ✅ Correctly raised: {type(e).__name__}")
            print(f"   Message: {e}")
        print()
        
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
            print("🧹 Cleaned up test file")
        print()


def test_web_loader():
    """Test the SimpleWebLoader."""
    print("=" * 70)
    print("Testing SimpleWebLoader (URL-based)")
    print("=" * 70)
    print()
    
    # Initialize loader
    url = "https://example.com/article"
    print(f"📋 Initializing SimpleWebLoader...")
    print(f"   URL: {url}")
    print()
    
    loader = SimpleWebLoader(url)
    
    # Load documents
    print("📥 Loading documents...")
    documents = loader.load_documents()
    print(f"✅ Loaded {len(documents)} document(s)")
    print()
    
    # Display results
    doc = documents[0]
    print("📄 Document Details:")
    print(f"   Content: {doc.page_content}")
    print()
    
    print("🏷️  Metadata:")
    for key, value in doc.metadata.items():
        print(f"   • {key}: {value}")
    print()
    
    # Test validation
    print("❌ Testing with invalid URL...")
    try:
        bad_loader = SimpleWebLoader("not-a-url")
        bad_loader.load_documents()
        print("   ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"   ✅ Correctly raised: {type(e).__name__}")
        print(f"   Message: {e}")
    print()


def test_interface_compliance():
    """Test that loaders comply with the interface."""
    print("=" * 70)
    print("Testing Interface Compliance")
    print("=" * 70)
    print()
    
    # Check that SimpleTextLoader is a proper subclass
    print("Checking SimpleTextLoader...")
    print(f"   Is subclass of BaseDocumentLoader: {issubclass(SimpleTextLoader, BaseDocumentLoader)}")
    print(f"   Has load_documents method: {hasattr(SimpleTextLoader, 'load_documents')}")
    print()
    
    print("Checking SimpleWebLoader...")
    print(f"   Is subclass of BaseDocumentLoader: {issubclass(SimpleWebLoader, BaseDocumentLoader)}")
    print(f"   Has load_documents method: {hasattr(SimpleWebLoader, 'load_documents')}")
    print()
    
    # Try to instantiate base class (should fail)
    print("❌ Attempting to instantiate abstract BaseDocumentLoader...")
    try:
        base = BaseDocumentLoader()
        print("   ERROR: Should not be able to instantiate abstract class")
    except TypeError as e:
        print(f"   ✅ Correctly raised: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")
    print()


def show_usage_patterns():
    """Show common usage patterns."""
    print("=" * 70)
    print("Common Usage Patterns")
    print("=" * 70)
    print()
    
    print("1. Basic Loading Pattern:")
    print("-" * 40)
    print("""
    from app.ingestion.base_loader import BaseDocumentLoader
    
    class MyLoader(BaseDocumentLoader):
        def load_documents(self) -> List[Document]:
            # Your loading logic
            return documents
    
    loader = MyLoader(source="...")
    documents = loader.load_documents()
    """)
    print()
    
    print("2. With Validation:")
    print("-" * 40)
    print("""
    def load_documents(self) -> List[Document]:
        # Validate first
        self._validate_source(self.source)
        
        # Then load
        content = self._load_content()
        
        # Create document
        return [Document(
            page_content=content,
            metadata=self._add_metadata({...})
        )]
    """)
    print()
    
    print("3. Multiple Documents:")
    print("-" * 40)
    print("""
    def load_documents(self) -> List[Document]:
        documents = []
        
        for item in self.items:
            doc = Document(
                page_content=item.content,
                metadata=self._add_metadata({...})
            )
            documents.append(doc)
        
        return documents
    """)
    print()


def main():
    """Run all tests."""
    print("\n")
    print("🧪 BaseDocumentLoader Interface Test Suite")
    print("\n")
    
    try:
        # Test concrete implementations
        test_text_loader()
        test_web_loader()
        
        # Test interface compliance
        test_interface_compliance()
        
        # Show usage patterns
        show_usage_patterns()
        
        print("=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
        print()
        print("💡 Key Takeaways:")
        print("   • BaseDocumentLoader defines a consistent interface")
        print("   • All loaders return List[Document]")
        print("   • Validation happens before loading")
        print("   • Metadata is standardized with _add_metadata()")
        print("   • Abstract base class prevents direct instantiation")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
