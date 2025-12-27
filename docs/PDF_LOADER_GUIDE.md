# PDF Loader - Complete Guide

## Overview

The **PDFDocumentLoader** is a production-ready PDF ingestion module that extends the `BaseDocumentLoader` interface. It provides a clean, robust way to load PDF documents into your RAG (Retrieval-Augmented Generation) pipeline.

### Why This Matters

PDFs are the most common document format in enterprise environments:
- Research papers, technical documentation, legal contracts
- Annual reports, whitepapers, user manuals
- Invoices, forms, and structured documents

A reliable PDF loader is essential for building knowledge-based AI systems.

---

## Architecture

```
PDFDocumentLoader (extends BaseDocumentLoader)
    │
    ├── __init__(path, source_name)
    │   └── Accepts file or directory path
    │
    ├── load_documents() → List[Document]
    │   ├── Validates path exists
    │   ├── Detects file vs directory
    │   ├── _load_single_pdf() → Uses PyPDFLoader
    │   ├── _load_directory() → Uses PyPDFDirectoryLoader
    │   └── Adds standardized metadata
    │
    └── Uses LangChain Loaders
        ├── PyPDFLoader (single file, page-level splitting)
        └── PyPDFDirectoryLoader (batch loading)
```

### Design Patterns Used

1. **Template Method**: BaseDocumentLoader defines the interface, PDFDocumentLoader implements specifics
2. **Adapter Pattern**: Wraps LangChain loaders with consistent interface
3. **Strategy Pattern**: Automatically chooses loader based on path type

---

## Installation

```bash
# Install required dependencies
pip install langchain langchain-community pypdf

# Optional: For testing
pip install reportlab

# Optional: Advanced PDF features
pip install pdfplumber pymupdf pytesseract
```

**Note**: The loader uses `pypdf` (modern PyPDF), not the older `PyPDF2`.

---

## Basic Usage

### 1. Load a Single PDF

```python
from app.ingestion.pdf_loader import PDFDocumentLoader

# Load one PDF file
loader = PDFDocumentLoader(
    path="data/documents/annual_report.pdf",
    source_name="Annual Report 2024"
)

documents = loader.load_documents()

# Result: One Document per page
print(f"Loaded {len(documents)} pages")
print(f"First page: {documents[0].page_content[:200]}")
print(f"Metadata: {documents[0].metadata}")
```

**Output**:
```
Loaded 25 pages
First page: Executive Summary

This annual report provides an overview of our company's 
performance in 2024, including financial results, strategic 
initiatives, and future outlook...

Metadata: {
    'source': 'data/documents/annual_report.pdf',
    'file_name': 'annual_report.pdf',
    'loader_type': 'pdf',
    'page': 0,
    'ingestion_time': '2024-01-15T10:30:00'
}
```

### 2. Load All PDFs from a Directory

```python
# Load all PDFs in a folder
loader = PDFDocumentLoader(
    path="data/research_papers/",
    source_name="AI Research Collection"
)

documents = loader.load_documents()

# Get statistics
unique_files = set(doc.metadata['file_name'] for doc in documents)
print(f"Loaded {len(documents)} pages from {len(unique_files)} files")

# Group by file
from collections import defaultdict
by_file = defaultdict(list)
for doc in documents:
    by_file[doc.metadata['file_name']].append(doc)

for filename, pages in by_file.items():
    print(f"  {filename}: {len(pages)} pages")
```

**Output**:
```
Loaded 87 pages from 5 files
  attention_is_all_you_need.pdf: 15 pages
  bert_paper.pdf: 16 pages
  gpt3_paper.pdf: 24 pages
  resnet_paper.pdf: 16 pages
  transformer_xl.pdf: 16 pages
```

---

## Complete RAG Pipeline Example

Here's how to use the PDF loader in a full RAG workflow:

```python
from app.ingestion.pdf_loader import PDFDocumentLoader
from app.core.embeddings import EmbeddingManager
from app.core.llm import LLMFactory
from app.utils.config import get_settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Step 1: Load PDFs
print("Loading PDFs...")
loader = PDFDocumentLoader(
    path="data/knowledge_base/",
    source_name="Company Knowledge Base"
)
documents = loader.load_documents()
print(f"✓ Loaded {len(documents)} pages")

# Step 2: Split into chunks
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
print(f"✓ Created {len(chunks)} chunks")

# Step 3: Create embeddings
print("Creating embeddings...")
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.get_embeddings()

# Step 4: Build vector store
print("Building vector store...")
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"✓ Vector store created")

# Step 5: Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Step 6: Create QA chain
settings = get_settings()
llm = LLMFactory.create(settings)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Step 7: Query the system
query = "What are the main findings of the research?"
result = qa_chain({"query": query})

print(f"\nQuery: {query}")
print(f"Answer: {result['result']}")
print(f"\nSources:")
for doc in result['source_documents']:
    print(f"  - {doc.metadata['file_name']} (page {doc.metadata['page']})")
```

---

## Metadata Structure

Every loaded document includes standardized metadata:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `source` | str | Full path to PDF file | `"data/docs/report.pdf"` |
| `file_name` | str | Name of PDF file | `"report.pdf"` |
| `loader_type` | str | Always `"pdf"` | `"pdf"` |
| `page` | int | Page number (0-indexed) | `0` |
| `ingestion_time` | str | When document was loaded | `"2024-01-15T10:30:00"` |

### Accessing Metadata

```python
documents = loader.load_documents()

for doc in documents:
    print(f"File: {doc.metadata['file_name']}")
    print(f"Page: {doc.metadata['page'] + 1}")  # +1 for 1-indexed
    print(f"Content length: {len(doc.page_content)} chars")
    print(f"First 100 chars: {doc.page_content[:100]}")
    print("-" * 50)
```

---

## Error Handling

The loader provides clear error messages for common issues:

### 1. File Not Found

```python
try:
    loader = PDFDocumentLoader(path="nonexistent.pdf")
    documents = loader.load_documents()
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Output: Source does not exist: nonexistent.pdf
```

### 2. Not a PDF File

```python
try:
    loader = PDFDocumentLoader(path="document.txt")
    documents = loader.load_documents()
except ValueError as e:
    print(f"Error: {e}")
    # Output: File must have .pdf extension, got: .txt
```

### 3. Empty Directory

```python
try:
    loader = PDFDocumentLoader(path="empty_folder/")
    documents = loader.load_documents()
except ValueError as e:
    print(f"Error: {e}")
    # Output: No PDF files found in directory: empty_folder/
```

### Best Practice: Safe Loading

```python
from pathlib import Path

def safe_load_pdfs(path: str, source_name: str = None):
    """
    Safely load PDFs with comprehensive error handling.
    
    Returns:
        List[Document]: Loaded documents, or empty list on error
    """
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
        import traceback
        traceback.print_exc()
        return []

# Usage
documents = safe_load_pdfs("data/pdfs/", "Research Papers")
if documents:
    # Process documents
    pass
```

---

## Advanced Topics

### 1. Page-Level Retrieval

Because the loader returns one Document per page, you get natural page-level granularity:

```python
documents = loader.load_documents()

# Build vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# Search returns whole pages
results = vectorstore.similarity_search("machine learning", k=3)

for result in results:
    print(f"File: {result.metadata['file_name']}")
    print(f"Page: {result.metadata['page'] + 1}")
    print(f"Content: {result.page_content[:200]}")
```

**When to use**:
- Each page contains complete information
- You want to cite specific pages in responses
- PDFs have clear page-level organization

### 2. Chunk-Level Retrieval

For more granular retrieval, split pages into smaller chunks:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load pages
documents = loader.load_documents()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Smaller chunks
    chunk_overlap=100,
    length_function=len
)
chunks = splitter.split_documents(documents)

# Each chunk retains page metadata
for chunk in chunks[:3]:
    print(f"From: {chunk.metadata['file_name']}, page {chunk.metadata['page']}")
    print(f"Content: {chunk.page_content[:100]}")
```

**When to use**:
- Pages are very long
- You want precise retrieval
- Embedding model has token limits

### 3. Filtering by File

```python
# Load all PDFs
documents = loader.load_documents()

# Filter to specific file
report_pages = [
    doc for doc in documents 
    if doc.metadata['file_name'] == 'annual_report.pdf'
]

# Process only those pages
vectorstore = FAISS.from_documents(report_pages, embeddings)
```

### 4. Processing Large Collections

For very large PDF collections (1000+ files):

```python
import os
from pathlib import Path

def load_pdfs_in_batches(directory: str, batch_size: int = 100):
    """Load PDFs in batches to avoid memory issues."""
    
    # Get all PDF files
    pdf_files = list(Path(directory).glob("*.pdf"))
    
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch)} files")
        
        # Load this batch
        documents = []
        for pdf_path in batch:
            loader = PDFDocumentLoader(path=str(pdf_path))
            try:
                docs = loader.load_documents()
                documents.extend(docs)
            except Exception as e:
                print(f"  ✗ Failed to load {pdf_path.name}: {e}")
        
        # Process batch (embed, store, etc.)
        yield documents

# Usage
for batch_docs in load_pdfs_in_batches("data/large_collection/"):
    # Process each batch
    chunks = text_splitter.split_documents(batch_docs)
    # Add to vector store
    # ...
```

---

## Common Pitfalls & Solutions

### ❌ DON'T: Load PDFs multiple times

```python
# Bad: Creates loader repeatedly
for i in range(10):
    loader = PDFDocumentLoader(path="data/doc.pdf")
    documents = loader.load_documents()
    # Process...
```

### ✅ DO: Load once, reuse

```python
# Good: Load once, process multiple times
loader = PDFDocumentLoader(path="data/doc.pdf")
documents = loader.load_documents()

for i in range(10):
    # Process the same documents
    pass
```

### ❌ DON'T: Ignore metadata

```python
# Bad: Loses tracking information
for doc in documents:
    print(doc.page_content)  # No way to know which file/page
```

### ✅ DO: Use metadata for tracking

```python
# Good: Track source information
for doc in documents:
    source = f"{doc.metadata['file_name']}:page {doc.metadata['page']}"
    print(f"[{source}] {doc.page_content[:100]}")
```

### ❌ DON'T: Mix file and directory loading

```python
# Bad: Unclear intent
path = "data/something"  # File or directory?
loader = PDFDocumentLoader(path=path)
```

### ✅ DO: Be explicit

```python
# Good: Clear what you're loading
from pathlib import Path

path = Path("data/something")
if path.is_file():
    loader = PDFDocumentLoader(path=str(path), source_name=path.stem)
elif path.is_directory():
    loader = PDFDocumentLoader(path=str(path), source_name=path.name)
else:
    raise ValueError(f"Path must be file or directory: {path}")
```

---

## Testing

Run the test suite to verify your installation:

```bash
# Run all tests
python test_pdf_loader.py
```

The test suite validates:
1. Single file loading
2. Directory batch loading
3. Error handling (missing files, invalid paths)
4. Interface compliance with BaseDocumentLoader
5. Metadata standardization

---

## Future Enhancements (TODOs)

The loader includes extensive TODO comments for advanced features:

### 1. OCR Support for Scanned PDFs

```python
# For PDFs that are images of text (scanned documents)
from langchain_community.document_loaders import UnstructuredPDFLoader

def load_with_ocr(path: str):
    loader = UnstructuredPDFLoader(
        path,
        mode="elements",
        strategy="ocr_only"
    )
    return loader.load()

# Dependencies: pip install unstructured pytesseract pdf2image
```

### 2. Password-Protected PDFs

```python
# For encrypted PDFs
import pikepdf

def load_encrypted_pdf(path: str, password: str):
    with pikepdf.open(path, password=password) as pdf:
        # Extract text from decrypted PDF
        pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            pages.append(Document(
                page_content=text,
                metadata={"page": page_num, "source": path}
            ))
    return pages

# Dependencies: pip install pikepdf
```

### 3. Large PDF Optimization

```python
# For very large PDFs (100+ pages, 100+ MB)
def load_page_range(path: str, start: int, end: int):
    """Load only specific pages."""
    import fitz  # PyMuPDF
    
    doc = fitz.open(path)
    documents = []
    
    for page_num in range(start, min(end, len(doc))):
        page = doc[page_num]
        text = page.get_text()
        documents.append(Document(
            page_content=text,
            metadata={"page": page_num, "source": path}
        ))
    
    doc.close()
    return documents

# Dependencies: pip install PyMuPDF
```

### 4. Table Extraction

```python
# For PDFs with tables
import pdfplumber

def extract_tables(path: str):
    """Extract tables from PDF."""
    tables = []
    
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            for table in page_tables:
                tables.append({
                    "page": page_num,
                    "data": table,
                    "markdown": table_to_markdown(table)
                })
    
    return tables

# Dependencies: pip install pdfplumber
```

---

## Interview-Ready Talking Points

When discussing this implementation in interviews:

1. **Interface Design**: "I implemented the Template Method pattern by extending BaseDocumentLoader, ensuring consistent interface across all document types."

2. **Error Handling**: "The loader validates inputs early and provides actionable error messages, following the fail-fast principle."

3. **Metadata Standardization**: "Every document includes standardized metadata (source, file_name, loader_type, page, ingestion_time), enabling better tracking and retrieval."

4. **Flexibility**: "The loader automatically detects file vs directory paths and uses the appropriate LangChain loader (PyPDFLoader or PyPDFDirectoryLoader)."

5. **Production-Ready**: "I included comprehensive error handling, type hints, docstrings, and a full test suite."

6. **Extensibility**: "The TODO comments document future enhancements like OCR support, password-protected PDFs, and table extraction with working code examples."

---

## Related Documentation

- [BaseDocumentLoader Guide](BASE_LOADER_GUIDE.md) - Base interface documentation
- [Embedding Guide](EMBEDDING_GUIDE.md) - How to create embeddings from loaded PDFs
- [LLM Abstraction Guide](docs/LLM_ABSTRACTION_GUIDE.md) - Using LLMs for RAG

---

## Summary

The PDFDocumentLoader provides:
- ✅ Clean, production-ready PDF loading
- ✅ Single file and directory support
- ✅ Standardized metadata
- ✅ Clear error handling
- ✅ Page-level document splitting
- ✅ LangChain integration
- ✅ Comprehensive test suite
- ✅ Extensible design for future features

**Next steps**: Use the loader in your RAG pipeline by combining it with the EmbeddingManager and Vector Store.
