# PDF Loader - Quick Reference

## One-Page Cheat Sheet

### Installation
```bash
pip install langchain langchain-community pypdf reportlab
```

---

### Basic Usage

#### Load Single PDF
```python
from app.ingestion.pdf_loader import PDFDocumentLoader

loader = PDFDocumentLoader(
    path="data/report.pdf",
    source_name="Annual Report"
)
documents = loader.load_documents()  # One Document per page
```

#### Load Directory
```python
loader = PDFDocumentLoader(
    path="data/pdfs/",
    source_name="Research Papers"
)
documents = loader.load_documents()  # All PDFs in directory
```

---

### Complete RAG Pipeline
```python
from app.ingestion.pdf_loader import PDFDocumentLoader
from app.core.embeddings import EmbeddingManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 1. Load PDFs
loader = PDFDocumentLoader(path="data/docs/")
documents = loader.load_documents()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Create embeddings & vector store
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.get_embeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Query
retriever = vectorstore.as_retriever()
results = retriever.get_relevant_documents("What is the main topic?")
```

---

### Metadata Structure
| Field | Description | Example |
|-------|-------------|---------|
| `source` | Full file path | `"data/docs/report.pdf"` |
| `file_name` | File name only | `"report.pdf"` |
| `loader_type` | Always "pdf" | `"pdf"` |
| `page` | Page number (0-indexed) | `0` |
| `ingestion_time` | Load timestamp | `"2024-01-15T10:30:00"` |

---

### Error Handling
```python
try:
    loader = PDFDocumentLoader(path="data/doc.pdf")
    documents = loader.load_documents()
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Invalid file: {e}")
```

---

### Common Patterns

#### Filter by File
```python
documents = loader.load_documents()
report_pages = [doc for doc in documents 
                if doc.metadata['file_name'] == 'report.pdf']
```

#### Get Statistics
```python
documents = loader.load_documents()
unique_files = set(doc.metadata['file_name'] for doc in documents)
print(f"Loaded {len(documents)} pages from {len(unique_files)} files")
```

#### Page-Level Tracking
```python
for doc in documents:
    source = f"{doc.metadata['file_name']}:page {doc.metadata['page'] + 1}"
    print(f"[{source}] {doc.page_content[:100]}")
```

---

### Do's and Don'ts

| ✅ DO | ❌ DON'T |
|-------|----------|
| Load once, reuse documents | Load PDFs repeatedly in loops |
| Use metadata for tracking | Ignore source information |
| Handle errors explicitly | Assume files always exist |
| Split pages into chunks for RAG | Use raw pages for retrieval |
| Validate paths with Path.exists() | Trust user-provided paths |

---

### Testing
```bash
python test_pdf_loader.py
```

---

### Interface Contract
PDFDocumentLoader implements BaseDocumentLoader:
- **Required**: `load_documents() → List[Document]`
- **Inherited**: `_validate_source()`, `_add_metadata()`

---

### Future Enhancements (TODOs)

#### OCR for Scanned PDFs
```python
# Dependencies: pip install unstructured pytesseract
from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader(path, strategy="ocr_only")
documents = loader.load()
```

#### Password-Protected PDFs
```python
# Dependencies: pip install pikepdf
import pikepdf

with pikepdf.open(path, password=password) as pdf:
    # Extract text from decrypted PDF
    pass
```

#### Large PDF Optimization
```python
# Dependencies: pip install PyMuPDF
import fitz

doc = fitz.open(path)
for page_num in range(start_page, end_page):
    page = doc[page_num]
    text = page.get_text()
```

---

### Related Resources
- Full Guide: `docs/PDF_LOADER_GUIDE.md`
- Base Loader: `docs/BASE_LOADER_GUIDE.md`
- Embeddings: `EMBEDDING_QUICK_REF.md`
- LLMs: `QUICK_REFERENCE.md`

---

### Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` | Check path exists, use absolute paths |
| `ValueError: not a .pdf` | Verify file extension is `.pdf` |
| `No PDF files found` | Check directory contains PDFs |
| Import errors | `pip install langchain langchain-community pypdf` |
| Empty documents | PDF might be scanned (needs OCR) |

---

### Example: Safe Loading Function
```python
from pathlib import Path

def safe_load_pdfs(path: str, source_name: str = None):
    """Safely load PDFs with error handling."""
    try:
        loader = PDFDocumentLoader(path=path, source_name=source_name)
        documents = loader.load_documents()
        print(f"✓ Loaded {len(documents)} pages")
        return documents
    except FileNotFoundError:
        print(f"✗ Path not found: {path}")
        return []
    except ValueError as e:
        print(f"✗ Invalid path: {e}")
        return []
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return []

documents = safe_load_pdfs("data/pdfs/")
```

---

### Integration Example
```python
from app.ingestion.pdf_loader import PDFDocumentLoader
from app.core.embeddings import EmbeddingManager
from app.core.llm import LLMFactory
from app.utils.config import get_settings

# Load
loader = PDFDocumentLoader(path="data/docs/")
documents = loader.load_documents()

# Embed
embedding_manager = EmbeddingManager()
vectorstore = FAISS.from_documents(documents, embedding_manager.get_embeddings())

# Query
settings = get_settings()
llm = LLMFactory.create(settings)
chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
result = chain({"query": "Summarize the key findings"})
```

---

**Production-Ready**: Clean architecture, error handling, metadata tracking, and extensibility built-in.
