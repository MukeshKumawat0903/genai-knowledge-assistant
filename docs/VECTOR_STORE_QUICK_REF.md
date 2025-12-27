# Vector Store Manager - Quick Reference

## Overview

The `VectorStoreManager` is a clean, configuration-driven factory that abstracts multiple vector databases (FAISS, Chroma, Pinecone) behind a single interface for RAG systems.

## Quick Start

```python
from app.core.vector_store import VectorStoreManager
from app.utils.config import get_settings
from langchain.embeddings import OpenAIEmbeddings

# Load documents and chunk them
from app.core.text_splitter import TextChunker
documents = load_documents()  # Your document loading
chunker = TextChunker(get_settings())
chunks = chunker.split_documents(documents)

# Create embeddings
settings = get_settings()
embeddings = OpenAIEmbeddings()

# Create vector store
manager = VectorStoreManager(settings, embeddings)
vectorstore = manager.create_vector_store(chunks)

# Save (automatic for Chroma, manual for FAISS)
manager.save_vector_store(vectorstore)

# Later: Load and query
vectorstore = manager.load_vector_store()
results = vectorstore.similarity_search("What is RAG?", k=4)
```

## Features

✅ **Multi-Backend Support** - FAISS, Chroma, Pinecone (extensible)  
✅ **Configuration-Driven** - Switch stores via .env  
✅ **Consistent Interface** - Same API across all backends  
✅ **Automatic Persistence** - Chroma auto-saves, FAISS explicit  
✅ **Type Validation** - Clear errors for misconfigurations  
✅ **LangChain Compatible** - Works with all LangChain embeddings  
✅ **Interview-Ready** - Clean factory pattern implementation

## Configuration (.env)

```bash
# Vector Store Configuration
VECTOR_STORE_TYPE=faiss       # Options: faiss, chroma, pinecone
VECTOR_STORE_PATH=./data/vector_store
COLLECTION_NAME=documents
```

### Choosing a Vector Store

| Store | Best For | Pros | Cons |
|-------|---------|------|------|
| **FAISS** | Development, <1M vectors | Fastest, no deps, in-memory | No persistence, single-machine |
| **Chroma** | Local production, <10M vectors | Auto-persistence, filtering | Slower than FAISS, not scalable |
| **Pinecone** | Cloud production, >10M vectors | Fully managed, scalable | Requires API key, network latency |

## Usage Examples

### Create Vector Store (FAISS)

```python
from app.core.vector_store import VectorStoreManager
from langchain.embeddings import OpenAIEmbeddings

# Set VECTOR_STORE_TYPE=faiss in .env
settings = get_settings()
embeddings = OpenAIEmbeddings()

manager = VectorStoreManager(settings, embeddings)

# Create from documents
vectorstore = manager.create_vector_store(chunks)

# Save (required for FAISS)
manager.save_vector_store(vectorstore)

# Query
results = vectorstore.similarity_search("query", k=4)
```

### Create Vector Store (Chroma)

```python
# Set VECTOR_STORE_TYPE=chroma in .env
settings = get_settings()
embeddings = OpenAIEmbeddings()

manager = VectorStoreManager(settings, embeddings)

# Create from documents (auto-persists)
vectorstore = manager.create_vector_store(chunks)

# No manual save needed for Chroma
# manager.save_vector_store(vectorstore)  # No-op

# Query
results = vectorstore.similarity_search("query", k=4)
```

### Load Existing Vector Store

```python
# Works for both FAISS and Chroma
settings = get_settings()
embeddings = OpenAIEmbeddings()

manager = VectorStoreManager(settings, embeddings)

# Load previously created store
vectorstore = manager.load_vector_store()

# Query immediately
results = vectorstore.similarity_search("query", k=4)
```

### Complete RAG Pipeline

```python
from app.ingestion.pdf_loader import PDFDocumentLoader
from app.core.text_splitter import TextChunker
from app.core.vector_store import VectorStoreManager
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Load documents
pdf_loader = PDFDocumentLoader("./documents")
documents = pdf_loader.load_documents()

# 2. Chunk documents
settings = get_settings()
chunker = TextChunker(settings)
chunks = chunker.split_documents(documents)

# 3. Create embeddings
embeddings = OpenAIEmbeddings()

# 4. Create vector store
manager = VectorStoreManager(settings, embeddings)
vectorstore = manager.create_vector_store(chunks)

# 5. Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# 6. Create RAG chain
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 7. Query
result = qa_chain({"query": "What is RAG?"})
print(result["result"])
print(f"Sources: {len(result['source_documents'])}")
```

### Multi-Source Vector Store

```python
from app.ingestion.pdf_loader import PDFDocumentLoader
from app.ingestion.web_loader import WebDocumentLoader
from app.ingestion.youtube_loader import YouTubeDocumentLoader

# Load from multiple sources
all_docs = []

# PDFs
pdf_loader = PDFDocumentLoader("./documents")
all_docs.extend(pdf_loader.load_documents())

# Web pages
web_loader = WebDocumentLoader(["https://example.com"])
all_docs.extend(web_loader.load_documents())

# YouTube videos
yt_loader = YouTubeDocumentLoader("https://youtube.com/watch?v=VIDEO_ID")
all_docs.extend(yt_loader.load_documents())

# Chunk all documents
chunker = TextChunker(settings)
chunks = chunker.split_documents(all_docs)

# Create unified vector store
manager = VectorStoreManager(settings, embeddings)
vectorstore = manager.create_vector_store(chunks)

print(f"Indexed {len(all_docs)} documents from 3 sources")
print(f"Total chunks: {len(chunks)}")
```

## API Reference

### VectorStoreManager

```python
class VectorStoreManager:
    def __init__(self, settings: Settings, embeddings: Embeddings)
    
    def create_vector_store(
        self,
        documents: List[Document],
        **kwargs: Any
    ) -> VectorStore
    
    def load_vector_store(
        self,
        **kwargs: Any
    ) -> VectorStore
    
    def save_vector_store(
        self,
        vectorstore: Any
    ) -> None
    
    def get_vector_store_info(self) -> dict
```

### Methods

#### `__init__(settings, embeddings)`

Initialize manager with configuration and embeddings.

**Parameters:**
- `settings`: Settings object (from get_settings())
- `embeddings`: LangChain embeddings (OpenAIEmbeddings, HuggingFaceEmbeddings, etc.)

**Raises:**
- `ValueError`: If vector_store_type is unsupported

#### `create_vector_store(documents, **kwargs)`

Create new vector store from documents.

**Parameters:**
- `documents`: List[Document] - LangChain Document objects
- `**kwargs`: Backend-specific parameters

**Returns:**
- LangChain VectorStore instance

**Raises:**
- `ValueError`: If documents list is empty
- `NotImplementedError`: If Pinecone not implemented

#### `load_vector_store(**kwargs)`

Load existing vector store from persistence.

**Parameters:**
- `**kwargs`: Backend-specific loading parameters

**Returns:**
- LangChain VectorStore instance

**Raises:**
- `FileNotFoundError`: If vector store doesn't exist

#### `save_vector_store(vectorstore)`

Save vector store to persistence.

**Parameters:**
- `vectorstore`: Vector store instance to save

**Note:** Only required for FAISS; Chroma and Pinecone auto-persist.

#### `get_vector_store_info()`

Get current configuration information.

**Returns:**
- Dict with type, path, collection_name, embeddings

## Vector Store Comparison

### FAISS

**Strengths:**
- Fastest similarity search (in-memory)
- No external dependencies
- Best for development
- Simple API

**Limitations:**
- Requires manual save/load
- Not distributed
- Limited to memory size
- No built-in metadata filtering

**When to use:**
- Development and prototyping
- Small datasets (<1M vectors)
- Speed is critical
- Simple use cases

### Chroma

**Strengths:**
- Automatic persistence
- Built-in metadata filtering
- Easy to use
- Good for local production

**Limitations:**
- Slower than FAISS for large datasets
- Not horizontally scalable
- Limited to single machine

**When to use:**
- Local production deployments
- Small-medium datasets (<10M vectors)
- Need metadata filtering
- Want automatic persistence

### Pinecone

**Strengths:**
- Fully managed (no infrastructure)
- Horizontally scalable
- High availability
- Advanced filtering

**Limitations:**
- Requires API key and internet
- Usage-based pricing
- Network latency
- Vendor lock-in

**When to use:**
- Large-scale production (>10M vectors)
- Need scalability
- Global deployment
- Don't want to manage infrastructure

## Error Handling

### Empty Documents

```python
try:
    vectorstore = manager.create_vector_store([])
except ValueError as e:
    print(e)
    # "Cannot create vector store from empty document list..."
```

### Invalid Vector Store Type

```python
# If VECTOR_STORE_TYPE=invalid in .env
try:
    manager = VectorStoreManager(settings, embeddings)
except ValueError as e:
    print(e)
    # "Unsupported vector store type: 'invalid'. Supported types: ..."
```

### Vector Store Not Found

```python
try:
    vectorstore = manager.load_vector_store()
except FileNotFoundError as e:
    print(e)
    # "FAISS index not found at ./data/vector_store/index..."
```

### Pinecone Not Implemented

```python
# If VECTOR_STORE_TYPE=pinecone in .env
try:
    vectorstore = manager.create_vector_store(docs)
except NotImplementedError as e:
    print(e)
    # "Pinecone integration not yet implemented. To use Pinecone: ..."
```

## Best Practices

### ✅ DO

- **Choose appropriate store** for your use case and scale
- **Use consistent embeddings** when loading/creating
- **Save FAISS indexes** explicitly after creation
- **Validate documents** before indexing
- **Monitor memory usage** with FAISS (in-memory)
- **Use metadata** for better filtering and organization
- **Cache loaded stores** to avoid repeated loading

### ❌ DON'T

- **Don't hardcode vector store type** - use configuration
- **Don't mix embeddings** between save/load
- **Don't forget to save** FAISS indexes
- **Don't index empty documents** - validate first
- **Don't ignore errors** - handle FileNotFoundError gracefully
- **Don't use wrong store** for scale (FAISS for 100M vectors)

## Performance Considerations

### FAISS
- **Indexing**: Very fast (in-memory)
- **Search**: Fastest (~1ms for 100k vectors)
- **Memory**: ~4 bytes × dimension × num_vectors
- **Throughput**: 10k+ queries/sec (single machine)

### Chroma
- **Indexing**: Medium (writes to disk)
- **Search**: Medium (~10-50ms)
- **Disk**: Depends on collection size
- **Throughput**: 1k-10k queries/sec

### Pinecone
- **Indexing**: Slow (network API calls)
- **Search**: Medium-Fast (~50-100ms with network)
- **Storage**: Cloud-managed
- **Throughput**: Scales horizontally

## Troubleshooting

### FAISS Index Not Found

**Problem**: FileNotFoundError when loading

**Solutions:**
- Create index first with `create_vector_store()`
- Check `VECTOR_STORE_PATH` in .env
- Verify you called `save_vector_store()`

### Chroma Collection Empty

**Problem**: Loading returns empty collection

**Causes:**
- Wrong collection name
- Wrong persist directory
- Index was cleared

**Solutions:**
- Check `COLLECTION_NAME` in .env
- Verify `VECTOR_STORE_PATH` points to correct directory
- Recreate index if necessary

### Different Embeddings Error

**Problem**: Loading fails with dimension mismatch

**Cause:** Using different embedding models for create vs. load

**Solution:** Use same embedding model consistently:
```python
# Create
embeddings = OpenAIEmbeddings()  # 1536 dimensions
manager.create_vector_store(docs)

# Load - use SAME embeddings
embeddings = OpenAIEmbeddings()  # Must be same
manager.load_vector_store()
```

## Advanced Features (TODOs)

The implementation includes comprehensive TODO comments for:

1. **Hybrid Search** - Combine vector + keyword (BM25)
2. **Metadata Filtering** - Filter by document type, date, source
3. **Index Versioning** - Track changes, A/B test, rollback
4. **Incremental Updates** - Add/delete without full rebuild
5. **Performance Monitoring** - Track latency, QPS, cache hits
6. **Pinecone Integration** - Full cloud-native implementation

See [vector_store.py](../app/core/vector_store.py) for detailed examples.

## Testing

Run the test suite:

```bash
python tests/test_vector_store.py
```

Tests cover:
- Initialization and configuration
- FAISS creation and loading
- Chroma creation
- Empty document handling
- Invalid type validation
- Pinecone placeholder
- Configuration info retrieval

## Dependencies

Already in requirements.txt:
```text
langchain>=0.1.0
faiss-cpu>=1.7.4
chromadb>=0.4.20  # Optional, for Chroma
pinecone-client>=3.0.0  # Optional, for Pinecone
```

## Integration Points

**Upstream** (Inputs):
- Text Chunker (chunked documents)
- Embeddings Factory (embedding models)

**Downstream** (Outputs):
- RAG Retriever (similarity search)
- Q&A Chains (retrieval-augmented generation)

## Architecture

```
Document Loading → Chunking → Embedding → Vector Storage → Retrieval
                                           ^^^^^^^^^^^^^^
                                         (VectorStoreManager)
```

## Next Steps

- **RAG Retriever**: Build retriever layer on top of vector store
- **Q&A Chain**: Connect retriever + LLM for question answering
- **Evaluation**: Measure retrieval quality (recall@k, MRR)
- **Monitoring**: Add performance metrics and logging

## Support

For issues or questions:
1. Check TODO comments in [vector_store.py](../app/core/vector_store.py)
2. Review test cases in [test_vector_store.py](../tests/test_vector_store.py)
3. Refer to [LangChain vector store docs](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
