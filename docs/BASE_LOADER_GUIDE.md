# Base Document Loader - Complete Guide

## 🎯 Purpose

The `BaseDocumentLoader` is an abstract base class that defines a consistent interface for ALL document loaders in the GenAI Knowledge Assistant. It ensures that regardless of source type (PDF, web, YouTube, etc.), all loaders return the same data structure and follow the same contract.

## 🧠 Why This Matters

### The Problem Without a Base Interface

```python
# Without interface - inconsistent methods and return types
pdf_docs = pdf_loader.load_pdf()           # Returns custom PDF objects
web_docs = web_loader.scrape_web()         # Returns dict
youtube_docs = youtube_loader.get_videos() # Returns strings

# Consumer code must handle each type differently
if isinstance(docs, PDFDocument):
    content = docs.text
elif isinstance(docs, dict):
    content = docs['content']
elif isinstance(docs, str):
    content = docs
```

### The Solution With BaseDocumentLoader

```python
# With interface - consistent method and return type
pdf_docs = pdf_loader.load_documents()     # Returns List[Document]
web_docs = web_loader.load_documents()     # Returns List[Document]
youtube_docs = youtube_loader.load_documents()  # Returns List[Document]

# Consumer code is simple and consistent
for doc in docs:
    content = doc.page_content
    source = doc.metadata['source']
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│       BaseDocumentLoader (ABC)              │
│                                             │
│  + load_documents() → List[Document]        │
│  # _validate_source(source)                 │
│  # _add_metadata(custom) → Dict             │
└─────────────────┬───────────────────────────┘
                  │
          Implemented by
                  │
    ┌─────────────┼─────────────┬─────────────┐
    │             │             │             │
┌───▼──────┐ ┌───▼──────┐ ┌───▼──────┐ ┌───▼──────┐
│PDFLoader │ │WebLoader │ │YouTubeL. │ │Custom... │
│          │ │          │ │          │ │          │
│load_docs │ │load_docs │ │load_docs │ │load_docs │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
     │             │             │             │
     └─────────────┴─────────────┴─────────────┘
                   │
          Returns consistent
                   ↓
           List[Document]
```

## 📋 Interface Contract

### Required Method

Every loader MUST implement:

```python
@abstractmethod
def load_documents(self) -> List[Document]:
    """
    Load documents from source.
    
    Returns:
        List[Document]: LangChain Document objects
    """
    pass
```

### Optional Helper Methods

Loaders CAN use:

```python
def _validate_source(self, source: Any) -> None:
    """Validate source before loading."""
    pass

def _add_metadata(self, custom_metadata: Dict) -> Dict:
    """Add standard metadata fields."""
    pass
```

## 🚀 Implementation Guide

### Step 1: Create Loader Class

```python
from app.ingestion.base_loader import BaseDocumentLoader
from langchain.schema import Document
from typing import List

class MyCustomLoader(BaseDocumentLoader):
    """
    Custom loader for [describe your source type].
    """
    
    def __init__(self, source: str, **options):
        """
        Initialize the loader.
        
        Args:
            source: Path, URL, or ID of the source
            **options: Loader-specific options
        """
        self.source = source
        self.options = options
```

### Step 2: Implement load_documents()

```python
    def load_documents(self) -> List[Document]:
        """
        Load documents from the source.
        
        Returns:
            List[Document]: Loaded documents
        """
        # 1. Validate source
        self._validate_source(self.source)
        
        # 2. Load content (your custom logic here)
        content = self._load_content()
        
        # 3. Create Document with metadata
        doc = Document(
            page_content=content,
            metadata=self._add_metadata({
                "source": self.source,
                "custom_field": "value"
            })
        )
        
        # 4. Return list
        return [doc]
```

### Step 3: (Optional) Add Validation

```python
    def _validate_source(self, source: str) -> None:
        """
        Validate the source.
        
        Args:
            source: Source to validate
            
        Raises:
            ValueError: If source is invalid
        """
        if not source:
            raise ValueError("Source cannot be empty")
        
        # Add source-specific validation
        if not self._is_valid_format(source):
            raise ValueError(f"Invalid format: {source}")
```

### Step 4: Add Custom Loading Logic

```python
    def _load_content(self) -> str:
        """
        Load content from source.
        
        Returns:
            str: Loaded content
        """
        # Your source-specific loading logic
        # Examples:
        # - Read file: Path(self.source).read_text()
        # - HTTP request: requests.get(self.source).text
        # - API call: api_client.fetch(self.source)
        
        return content
```

## 📚 Complete Example

### Example 1: Simple File Loader

```python
from pathlib import Path
from typing import List
from langchain.schema import Document
from app.ingestion.base_loader import BaseDocumentLoader

class SimpleFileLoader(BaseDocumentLoader):
    """Load plain text files."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load_documents(self) -> List[Document]:
        # Validate
        self._validate_source(self.file_path)
        
        # Load
        content = Path(self.file_path).read_text(encoding='utf-8')
        
        # Create document
        doc = Document(
            page_content=content,
            metadata=self._add_metadata({
                "source": self.file_path,
                "file_name": Path(self.file_path).name,
                "file_size": Path(self.file_path).stat().st_size
            })
        )
        
        return [doc]
    
    def _validate_source(self, source: str) -> None:
        if not source:
            raise ValueError("File path cannot be empty")
        
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {source}")

# Usage
loader = SimpleFileLoader("document.txt")
documents = loader.load_documents()
```

### Example 2: Web Page Loader (Conceptual)

```python
import requests
from bs4 import BeautifulSoup

class SimpleWebLoader(BaseDocumentLoader):
    """Load web pages."""
    
    def __init__(self, url: str):
        self.url = url
    
    def load_documents(self) -> List[Document]:
        # Validate
        self._validate_source(self.url)
        
        # Load
        response = requests.get(self.url)
        response.raise_for_status()
        
        # Parse
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text()
        title = soup.find('title').text if soup.find('title') else "No title"
        
        # Create document
        doc = Document(
            page_content=content,
            metadata=self._add_metadata({
                "source": self.url,
                "title": title,
                "status_code": response.status_code
            })
        )
        
        return [doc]
    
    def _validate_source(self, source: str) -> None:
        if not source:
            raise ValueError("URL cannot be empty")
        
        if not source.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL: {source}")

# Usage
loader = SimpleWebLoader("https://example.com")
documents = loader.load_documents()
```

### Example 3: Multi-Document Loader

```python
class DirectoryLoader(BaseDocumentLoader):
    """Load all files from a directory."""
    
    def __init__(self, directory: str, pattern: str = "*.txt"):
        self.directory = directory
        self.pattern = pattern
    
    def load_documents(self) -> List[Document]:
        # Validate
        self._validate_source(self.directory)
        
        # Find files
        path = Path(self.directory)
        files = list(path.glob(self.pattern))
        
        if not files:
            raise ValueError(f"No files matching {self.pattern} in {self.directory}")
        
        # Load all files
        documents = []
        for file_path in files:
            content = file_path.read_text(encoding='utf-8')
            
            doc = Document(
                page_content=content,
                metadata=self._add_metadata({
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "directory": self.directory
                })
            )
            documents.append(doc)
        
        return documents
    
    def _validate_source(self, source: str) -> None:
        if not source:
            raise ValueError("Directory path cannot be empty")
        
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {source}")
        
        if not path.is_dir():
            raise ValueError(f"Not a directory: {source}")

# Usage
loader = DirectoryLoader("./documents", pattern="*.md")
documents = loader.load_documents()
print(f"Loaded {len(documents)} documents")
```

## 🎓 Design Patterns

### 1. Template Method Pattern

**BaseDocumentLoader defines the template:**
- All loaders must implement `load_documents()`
- Optional helpers provide reusable behavior

**Benefits:**
- Consistent interface
- Reusable code
- Enforced contract

### 2. Strategy Pattern

**Each loader implements its strategy:**
- PDFLoader: PDF parsing strategy
- WebLoader: Web scraping strategy
- YouTubeLoader: API fetching strategy

**Benefits:**
- Interchangeable implementations
- Easy to add new strategies
- No modification to existing code

### 3. Open/Closed Principle (SOLID)

**Open for extension:**
- Easy to add new loader types

**Closed for modification:**
- BaseDocumentLoader never needs to change

## 📊 Metadata Standardization

### Standard Fields (Added by _add_metadata)

```python
{
    "loader_type": "PDFLoader",           # Automatically added
    "ingestion_time": "2025-12-21T..."   # Automatically added
}
```

### Custom Fields (Added by Loader)

```python
# File loaders
{
    "source": "/path/to/file.pdf",
    "file_name": "file.pdf",
    "file_size": 1024,
    "page_number": 5
}

# Web loaders
{
    "source": "https://example.com",
    "title": "Example Page",
    "author": "John Doe",
    "publish_date": "2025-01-01"
}

# YouTube loaders
{
    "source": "video_id_123",
    "title": "Video Title",
    "duration": 360,
    "channel": "Channel Name"
}
```

## 🧪 Testing

### Run Test Suite

```powershell
python test_base_loader.py
```

### Test Your Loader

```python
def test_my_loader():
    # Test successful loading
    loader = MyLoader("valid_source")
    documents = loader.load_documents()
    
    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)
    assert all("source" in doc.metadata for doc in documents)
    
    # Test validation
    try:
        bad_loader = MyLoader("")
        bad_loader.load_documents()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
```

## 🐛 Common Pitfalls

### ❌ Don't: Return Wrong Type

```python
def load_documents(self) -> List[Document]:
    return "content"  # Wrong! Must return List[Document]
```

### ✅ Do: Return List[Document]

```python
def load_documents(self) -> List[Document]:
    doc = Document(page_content="content", metadata={})
    return [doc]  # Correct!
```

### ❌ Don't: Forget Metadata

```python
doc = Document(
    page_content=content,
    metadata={}  # Missing source!
)
```

### ✅ Do: Include Source

```python
doc = Document(
    page_content=content,
    metadata=self._add_metadata({
        "source": self.source  # Always include source
    })
)
```

### ❌ Don't: Skip Validation

```python
def load_documents(self) -> List[Document]:
    # Direct loading without validation
    content = Path(self.path).read_text()  # May fail unexpectedly
```

### ✅ Do: Validate First

```python
def load_documents(self) -> List[Document]:
    self._validate_source(self.path)  # Fail fast with clear error
    content = Path(self.path).read_text()
```

## 💡 Best Practices

1. **Always validate sources** before loading
2. **Always include "source" in metadata** for traceability
3. **Use _add_metadata()** for consistent timestamps
4. **Return List[Document]** even for single documents
5. **Raise clear exceptions** with actionable messages
6. **Keep loaders focused** on loading, not processing
7. **Document your loader** with clear docstrings

## 🎯 Interview-Ready Points

### Q: Why use an abstract base class?

**A:** "The abstract base class enforces a contract that all loaders must follow. This ensures consistency - regardless of whether we're loading PDFs, web pages, or YouTube videos, the interface is identical. This makes the system maintainable and allows loaders to be swapped without changing consumer code."

### Q: What's the Template Method pattern?

**A:** "BaseDocumentLoader defines a template - all loaders must implement `load_documents()`. This is the Template Method pattern. The base class defines 'what' must be done, and concrete classes define 'how' to do it."

### Q: Why separate validation from loading?

**A:** "Separation of concerns and fail-fast principle. By validating first in `_validate_source()`, we catch errors early with clear messages before wasting time on expensive operations. It also makes validation logic reusable across similar loaders."

### Q: How does this support the Open/Closed Principle?

**A:** "The system is open for extension (easy to add new loader types by inheriting BaseDocumentLoader) but closed for modification (BaseDocumentLoader itself never needs to change). New loaders don't require modifying existing code."

## 🔗 Integration with RAG Pipeline

```python
# Step 1: Load documents using any loader
from app.ingestion.base_loader import BaseDocumentLoader
loader = PDFLoader("document.pdf")  # Or WebLoader, YouTubeLoader, etc.
documents = loader.load_documents()

# Step 2: Split into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Step 3: Create embeddings
from app.core.embeddings import EmbeddingManager
embeddings = EmbeddingManager().get_embeddings()

# Step 4: Store in vector database
from langchain_community.vectorstores import FAISS
vector_store = FAISS.from_documents(chunks, embeddings)

# Step 5: Query
results = vector_store.similarity_search("What is RAG?", k=5)
```

## 📝 Summary

**What we built:**
- Clean abstract interface with `BaseDocumentLoader`
- Single required method: `load_documents()`
- Optional helpers: `_validate_source()`, `_add_metadata()`
- Comprehensive documentation and examples

**Why it's production-ready:**
- Enforces consistent interface
- Type hints throughout
- Clear error handling
- Beginner-friendly documentation
- Extensible design

**Key benefits:**
- All loaders return same type
- Easy to add new loaders
- Validation built-in
- Metadata standardized
- Testable and maintainable

---

**Status**: ✅ Production-Ready | 🎓 Interview-Ready | 📚 Fully Documented
