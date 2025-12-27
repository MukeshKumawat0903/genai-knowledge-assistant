# Base Document Loader - Quick Reference

## 🚀 Quick Start

### Create a Loader

```python
from app.ingestion.base_loader import BaseDocumentLoader
from langchain.schema import Document
from typing import List

class MyLoader(BaseDocumentLoader):
    def __init__(self, source: str):
        self.source = source
    
    def load_documents(self) -> List[Document]:
        # 1. Validate
        self._validate_source(self.source)
        
        # 2. Load content
        content = self._load_content()
        
        # 3. Create Document
        doc = Document(
            page_content=content,
            metadata=self._add_metadata({
                "source": self.source
            })
        )
        
        # 4. Return list
        return [doc]
```

## 📋 Interface Contract

### Required
```python
@abstractmethod
def load_documents(self) -> List[Document]:
    """MUST implement this method."""
    pass
```

### Optional Helpers
```python
def _validate_source(self, source: Any) -> None:
    """Validate source before loading."""
    pass

def _add_metadata(self, custom: Dict) -> Dict:
    """Add standard metadata fields."""
    pass
```

## 💡 Common Patterns

### File Loader
```python
class FileLoader(BaseDocumentLoader):
    def load_documents(self) -> List[Document]:
        from pathlib import Path
        
        self._validate_source(self.path)
        content = Path(self.path).read_text()
        
        return [Document(
            page_content=content,
            metadata=self._add_metadata({"source": self.path})
        )]
    
    def _validate_source(self, source: str) -> None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
```

### Web Loader
```python
class WebLoader(BaseDocumentLoader):
    def load_documents(self) -> List[Document]:
        import requests
        
        self._validate_source(self.url)
        response = requests.get(self.url)
        
        return [Document(
            page_content=response.text,
            metadata=self._add_metadata({"source": self.url})
        )]
    
    def _validate_source(self, source: str) -> None:
        if not source.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL: {source}")
```

### Multi-Document Loader
```python
class DirectoryLoader(BaseDocumentLoader):
    def load_documents(self) -> List[Document]:
        from pathlib import Path
        
        documents = []
        for file in Path(self.directory).glob("*.txt"):
            content = file.read_text()
            doc = Document(
                page_content=content,
                metadata=self._add_metadata({
                    "source": str(file)
                })
            )
            documents.append(doc)
        
        return documents
```

## 🧪 Testing

```python
# Test successful loading
loader = MyLoader("valid_source")
documents = loader.load_documents()

assert isinstance(documents, list)
assert all(isinstance(doc, Document) for doc in documents)
assert all("source" in doc.metadata for doc in documents)

# Test validation
try:
    bad_loader = MyLoader("")
    bad_loader.load_documents()
    assert False, "Should raise ValueError"
except ValueError:
    pass  # Expected
```

## 📊 Metadata Structure

### Standard Fields (Automatic)
```python
{
    "loader_type": "MyLoader",          # Automatically added
    "ingestion_time": "2025-12-21..."   # Automatically added
}
```

### Custom Fields (Your Choice)
```python
{
    "source": "/path/to/file",          # Always include!
    "file_name": "document.txt",        # File-specific
    "title": "Page Title",              # Web-specific
    "video_id": "abc123",               # YouTube-specific
    "page_number": 5,                   # Multi-page docs
}
```

## ✅ Do's and Don'ts

### ✅ Do
- Return `List[Document]` (even for single doc)
- Include `"source"` in metadata
- Validate before loading
- Use `_add_metadata()` for timestamps
- Raise clear exceptions

### ❌ Don't
- Return other types (str, dict, custom objects)
- Skip metadata
- Load without validation
- Hardcode timestamps
- Swallow exceptions silently

## 🎯 Integration

```python
# Load → Split → Embed → Store → Query

# 1. Load
loader = MyLoader("source")
documents = loader.load_documents()

# 2. Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(documents)

# 3. Embed
from app.core.embeddings import EmbeddingManager
embeddings = EmbeddingManager().get_embeddings()

# 4. Store
from langchain_community.vectorstores import FAISS
vector_store = FAISS.from_documents(chunks, embeddings)

# 5. Query
results = vector_store.similarity_search("query", k=5)
```

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| "Can't instantiate abstract class" | Don't create `BaseDocumentLoader()` directly |
| Missing metadata | Always include `"source"` field |
| Wrong return type | Must return `List[Document]` |
| No validation | Use `_validate_source()` first |
| Unclear errors | Raise `ValueError` with descriptive message |

## 📚 Resources

- **Full Guide**: [docs/BASE_LOADER_GUIDE.md](docs/BASE_LOADER_GUIDE.md)
- **Test Script**: [test_base_loader.py](test_base_loader.py)
- **Implementation**: [app/ingestion/base_loader.py](app/ingestion/base_loader.py)

## 🎓 Interview Points

**Q: Why abstract base class?**
A: Enforces consistent interface across all loaders

**Q: Why List[Document]?**
A: Standard LangChain type, works with all downstream tools

**Q: Why separate validation?**
A: Fail-fast with clear errors, reusable logic

**Q: Design patterns?**
A: Template Method (base class), Strategy (implementations)

---

**Remember**: One interface, many implementations!
