# Document Indexer - Quick Reference Guide

**Module:** `app.rag.indexer`  
**Purpose:** Orchestrate the complete RAG indexing pipeline from raw documents to searchable vector stores

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [API Reference](#api-reference)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Usage (Recommended)

```python
from app.rag.indexer import create_indexer
from app.ingestion.pdf_loader import PDFDocumentLoader

# 1. Load documents
pdf_loader = PDFDocumentLoader("./documents")
documents = pdf_loader.load_documents()

# 2. Create indexer (auto-configured from .env)
indexer = create_indexer()

# 3. Index documents
result = indexer.index_documents(documents)

if result['success']:
    print(f"✓ Indexed {result['num_chunks']} chunks")
else:
    print(f"✗ Error: {result['message']}")
```

### Manual Configuration

```python
from app.rag.indexer import DocumentIndexer
from app.core.text_splitter import TextChunker
from app.core.embeddings import EmbeddingManager
from app.core.vector_store import VectorStoreManager
from app.utils.config import get_settings

# Initialize components
settings = get_settings()
text_chunker = TextChunker(settings)
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.get_embeddings()
vector_store_manager = VectorStoreManager(settings, embeddings)

# Create indexer
indexer = DocumentIndexer(
    text_chunker=text_chunker,
    embedding_manager=embedding_manager,
    vector_store_manager=vector_store_manager
)

# Index documents
result = indexer.index_documents(documents)
```

---

## Core Concepts

### What is Document Indexing?

Indexing is the process of converting raw documents into a searchable format:

```
Raw Documents → Split into Chunks → Generate Embeddings → Store in Vector DB
```

**Pipeline Steps:**
1. **Validation:** Ensure documents are valid LangChain Document objects
2. **Chunking:** Split documents into smaller, retrievable pieces
3. **Embedding:** Convert text chunks into numerical vectors
4. **Storage:** Store vectors in a vector database (FAISS, Chroma, Pinecone)
5. **Persistence:** Save the vector store for later retrieval

### Why Separate Indexing from Retrieval?

**Indexing (Write Operations):**
- Happens infrequently (when adding new documents)
- CPU/GPU intensive (embedding generation)
- Can be batched and optimized

**Retrieval (Read Operations):**
- Happens frequently (every user query)
- Requires low latency
- Operates on pre-computed embeddings

Separating these concerns enables independent optimization.

---

## API Reference

### DocumentIndexer Class

```python
class DocumentIndexer:
    """Orchestrates the document indexing pipeline."""
    
    def __init__(
        self,
        text_chunker: TextChunker,
        embedding_manager: EmbeddingManager,
        vector_store_manager: VectorStoreManager
    ):
        """Initialize with configured components."""
```

#### Methods

##### `index_documents(documents, **kwargs) -> Dict[str, Any]`

Index a list of documents into the vector store.

**Parameters:**
- `documents` (List[Document]): LangChain Document objects to index
- `**kwargs`: Additional arguments for vector store creation

**Returns:**
```python
{
    'success': bool,           # Whether indexing succeeded
    'num_documents': int,      # Number of input documents
    'num_chunks': int,         # Number of chunks created
    'vector_store_type': str,  # Type of vector store used
    'message': str             # Status message
}
```

**Raises:**
- `ValueError`: If documents list is empty or invalid
- `RuntimeError`: If indexing pipeline fails

---

### Convenience Function

##### `create_indexer(chunk_size=None, chunk_overlap=None, vector_store_type=None) -> DocumentIndexer`

Create a fully configured DocumentIndexer with optional overrides.

**Parameters:**
- `chunk_size` (int, optional): Override default chunk size
- `chunk_overlap` (int, optional): Override default chunk overlap
- `vector_store_type` (str, optional): Override vector store type

**Returns:**
- `DocumentIndexer`: Ready-to-use indexer instance

**Example:**
```python
# Use defaults from .env
indexer = create_indexer()

# Override settings
indexer = create_indexer(
    chunk_size=500,
    chunk_overlap=50,
    vector_store_type='chroma'
)
```

---

## Usage Examples

### Example 1: Index PDF Documents

```python
from app.rag.indexer import create_indexer
from app.ingestion.pdf_loader import PDFDocumentLoader

# Load PDFs
pdf_loader = PDFDocumentLoader("./research_papers")
documents = pdf_loader.load_documents()

print(f"Loaded {len(documents)} PDF pages")

# Index with default settings
indexer = create_indexer()
result = indexer.index_documents(documents)

print(f"Created {result['num_chunks']} searchable chunks")
```

### Example 2: Index Web Pages

```python
from app.rag.indexer import create_indexer
from app.ingestion.web_loader import WebDocumentLoader

# Load web content
urls = [
    "https://python.langchain.com/docs/",
    "https://docs.openai.com/",
]
web_loader = WebDocumentLoader()
documents = web_loader.load_urls(urls)

# Index with custom chunk size
indexer = create_indexer(chunk_size=800, chunk_overlap=80)
result = indexer.index_documents(documents)

if result['success']:
    print(f"✓ Indexed {result['num_chunks']} chunks")
    print(f"  Vector store: {result['vector_store_type']}")
```

### Example 3: Index YouTube Transcripts

```python
from app.rag.indexer import create_indexer
from app.ingestion.youtube_loader import YouTubeDocumentLoader

# Load YouTube transcripts
video_urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/example123",
]
youtube_loader = YouTubeDocumentLoader()
documents = youtube_loader.load_videos(video_urls)

# Index
indexer = create_indexer()
result = indexer.index_documents(documents)

print(f"Indexed {result['num_documents']} video transcripts")
```

### Example 4: Index Multiple Document Types

```python
from app.rag.indexer import create_indexer
from app.ingestion.pdf_loader import PDFDocumentLoader
from app.ingestion.web_loader import WebDocumentLoader
from app.ingestion.youtube_loader import YouTubeDocumentLoader

# Load documents from multiple sources
pdf_docs = PDFDocumentLoader("./papers").load_documents()
web_docs = WebDocumentLoader().load_urls(["https://example.com"])
youtube_docs = YouTubeDocumentLoader().load_videos(["https://youtu.be/abc"])

# Combine all documents
all_documents = pdf_docs + web_docs + youtube_docs

# Index everything together
indexer = create_indexer()
result = indexer.index_documents(all_documents)

print(f"Indexed {result['num_documents']} documents from multiple sources")
print(f"Created {result['num_chunks']} searchable chunks")
```

### Example 5: Error Handling

```python
from app.rag.indexer import create_indexer

try:
    indexer = create_indexer()
    result = indexer.index_documents(documents)
    
    if result['success']:
        print(f"✓ Success: {result['message']}")
    else:
        print(f"✗ Failed: {result['message']}")
        
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Indexing error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Example 6: Custom Vector Store Parameters

```python
from app.rag.indexer import create_indexer

indexer = create_indexer(vector_store_type='chroma')

# Pass custom parameters to Chroma
result = indexer.index_documents(
    documents,
    persist_directory="./custom_chroma_db",
    collection_name="research_papers"
)

print(f"Stored in: {result['vector_store_type']}")
```

---

## Best Practices

### 1. Document Preparation

**✓ DO:**
- Clean and preprocess documents before indexing
- Remove unnecessary whitespace and formatting
- Ensure documents have meaningful metadata
- Validate document quality before indexing

**✗ DON'T:**
- Index documents with empty content
- Mix different languages without proper handling
- Index duplicate documents without deduplication

```python
# Good: Clean documents before indexing
from app.utils.document_cleaner import clean_document

cleaned_docs = [clean_document(doc) for doc in raw_docs]
result = indexer.index_documents(cleaned_docs)

# Bad: Index raw, unprocessed documents
result = indexer.index_documents(raw_messy_docs)  # ✗
```

### 2. Chunk Size Selection

**General Guidelines:**
- **Small chunks (200-500 chars):** Precise retrieval, but may lose context
- **Medium chunks (500-1500 chars):** Good balance (recommended)
- **Large chunks (1500-3000 chars):** More context, but less precise

```python
# For technical documentation (needs context)
indexer = create_indexer(chunk_size=1000, chunk_overlap=200)

# For Q&A or short-form content
indexer = create_indexer(chunk_size=500, chunk_overlap=50)

# For long-form content (articles, books)
indexer = create_indexer(chunk_size=1500, chunk_overlap=300)
```

### 3. Vector Store Selection

**FAISS:** Development, prototyping, small datasets
```python
indexer = create_indexer(vector_store_type='faiss')
```

**Chroma:** Small-medium production, local persistence
```python
indexer = create_indexer(vector_store_type='chroma')
```

**Pinecone:** Large-scale production, cloud deployment
```python
indexer = create_indexer(vector_store_type='pinecone')
```

### 4. Monitoring and Logging

```python
import logging

logging.basicConfig(level=logging.INFO)

result = indexer.index_documents(documents)

# Log results
logging.info(f"Indexing completed:")
logging.info(f"  Documents: {result['num_documents']}")
logging.info(f"  Chunks: {result['num_chunks']}")
logging.info(f"  Store: {result['vector_store_type']}")

if not result['success']:
    logging.error(f"Indexing failed: {result['message']}")
```

### 5. Batch Processing

For large document collections, process in batches:

```python
def index_in_batches(documents, batch_size=100):
    """Index large document collections in batches."""
    indexer = create_indexer()
    
    total_chunks = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        result = indexer.index_documents(batch)
        
        if result['success']:
            total_chunks += result['num_chunks']
            print(f"Batch {i//batch_size + 1}: {result['num_chunks']} chunks")
        else:
            print(f"Batch {i//batch_size + 1} failed: {result['message']}")
    
    return total_chunks

# Usage
total = index_in_batches(large_document_list, batch_size=50)
print(f"Total indexed: {total} chunks")
```

---

## Troubleshooting

### Issue 1: "Document list is empty"

**Cause:** No documents provided to `index_documents()`

**Solution:**
```python
# Check document list before indexing
if not documents:
    print("No documents to index!")
else:
    result = indexer.index_documents(documents)
```

### Issue 2: "All items must be LangChain Document objects"

**Cause:** Invalid document types in the list

**Solution:**
```python
from langchain.schema import Document

# Convert strings to Documents
documents = [
    Document(page_content=text, metadata={"source": "text"})
    for text in text_list
]

result = indexer.index_documents(documents)
```

### Issue 3: "Found documents with empty content"

**Cause:** Documents with empty or whitespace-only content

**Solution:**
```python
# Filter out empty documents
valid_docs = [
    doc for doc in documents
    if doc.page_content and doc.page_content.strip()
]

result = indexer.index_documents(valid_docs)
```

### Issue 4: "Text chunking produced zero chunks"

**Cause:** Chunk size larger than document content

**Solution:**
```python
# Use smaller chunk size
indexer = create_indexer(chunk_size=200, chunk_overlap=20)
result = indexer.index_documents(documents)
```

### Issue 5: Slow Indexing Performance

**Causes:**
- Large documents
- Large chunk overlap
- Slow embedding model

**Solutions:**
```python
# 1. Reduce chunk overlap
indexer = create_indexer(chunk_overlap=50)  # Instead of 200

# 2. Use faster embedding model
# In .env: EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# 3. Process in smaller batches
result = indexer.index_documents(documents[:100])  # First 100 only
```

### Issue 6: Out of Memory

**Cause:** Indexing too many documents at once

**Solution:**
```python
# Index in smaller batches
def safe_index(documents, batch_size=50):
    indexer = create_indexer()
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        result = indexer.index_documents(batch)
        print(f"Indexed batch {i//batch_size + 1}")

safe_index(large_document_list)
```

---

## Configuration Reference

### Environment Variables

Key settings in `.env`:

```bash
# Chunking
CHUNK_SIZE=1000           # Characters per chunk
CHUNK_OVERLAP=200         # Overlap between chunks

# Vector Store
VECTOR_STORE_TYPE=faiss   # faiss, chroma, pinecone
VECTOR_STORE_PATH=./data/vector_store
COLLECTION_NAME=documents

# Embeddings
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

### Component Dependencies

```
DocumentIndexer
├── TextChunker (app/core/text_splitter.py)
│   └── Settings (chunk_size, chunk_overlap)
├── EmbeddingManager (app/core/embeddings.py)
│   └── Settings (embedding_model_name)
└── VectorStoreManager (app/core/vector_store.py)
    └── Settings (vector_store_type, vector_store_path, collection_name)
```

---

## Next Steps

After indexing documents, you can:

1. **Retrieve documents** using the retriever:
   ```python
   from app.rag.retriever import create_retriever
   
   retriever = create_retriever()
   results = retriever.retrieve("What is RAG?")
   ```

2. **Query with LLM** using the RAG chain:
   ```python
   from app.rag.chain import create_rag_chain
   
   chain = create_rag_chain()
   answer = chain.query("Explain RAG systems")
   ```

3. **Monitor index health**:
   ```python
   # Check vector store statistics
   vector_store = vector_store_manager.load_vector_store()
   print(f"Total vectors: {vector_store.index.ntotal}")
   ```

---

## Summary

**Key Takeaways:**
- Use `create_indexer()` for quick setup
- Index documents in batches for large collections
- Choose appropriate chunk sizes for your content
- Monitor indexing results for errors
- Separate indexing (write) from retrieval (read)

**Common Pattern:**
```python
# 1. Load
documents = loader.load_documents()

# 2. Index
indexer = create_indexer()
result = indexer.index_documents(documents)

# 3. Retrieve
retriever = create_retriever()
results = retriever.retrieve(query)
```
