# Text Chunker - Quick Reference

## Overview

The `TextChunker` is a clean, configuration-driven text splitting engine that converts LangChain Document objects into smaller, retrievable chunks optimized for RAG (Retrieval-Augmented Generation) systems.

## Quick Start

```python
from app.core.text_splitter import TextChunker
from app.utils.config import get_settings
from app.ingestion.pdf_loader import PDFDocumentLoader

# Load documents
pdf_loader = PDFDocumentLoader("./documents")
documents = pdf_loader.load_documents()

# Split into chunks
settings = get_settings()
chunker = TextChunker(settings)
chunks = chunker.split_documents(documents)

print(f"Original: {len(documents)} documents")
print(f"Chunks: {len(chunks)} chunks")
```

## Features

✅ **Configuration-Driven** - Reads chunk_size and chunk_overlap from Settings  
✅ **Metadata Preservation** - Keeps all original document metadata  
✅ **Chunk Tracking** - Adds chunk_index and total_chunks metadata  
✅ **Input Validation** - Graceful handling of edge cases  
✅ **Semantic Splitting** - Uses RecursiveCharacterTextSplitter (paragraphs → sentences → words)  
✅ **Estimation** - Pre-calculate chunk counts before splitting  
✅ **Interview-Ready** - Clean code with comprehensive docstrings

## Configuration

Set chunking parameters in your `.env` file:

```bash
# Chunking configuration
CHUNK_SIZE=1000          # Characters per chunk (typical: 500-2000)
CHUNK_OVERLAP=200        # Overlap between chunks (typical: 10-20% of chunk_size)
```

### Choosing Chunk Size

| Use Case | Chunk Size | Reasoning |
|----------|-----------|-----------|
| Short-form QA | 500-800 | Precise retrieval, less noise |
| General RAG | 1000-1500 | Balanced context and precision |
| Long-context | 1500-2000 | More context per chunk |
| Code | 300-600 | Function/class level granularity |

### Choosing Overlap

- **10-15%**: Minimal redundancy, lower costs
- **15-20%**: Recommended for most cases
- **20-30%**: Maximum context preservation

**Why overlap?** Prevents important information from being split across chunk boundaries.

## Usage Examples

### Basic Splitting

```python
from app.core.text_splitter import TextChunker
from app.utils.config import get_settings

settings = get_settings()
chunker = TextChunker(settings)

# Split documents
chunks = chunker.split_documents(documents)
```

### Without Chunk Metadata

```python
# Don't add chunk_index (lighter metadata)
chunks = chunker.split_documents(documents, add_chunk_metadata=False)
```

### Estimate Chunk Count

```python
# Get estimate before splitting (for progress bars, cost estimation)
estimate = chunker.get_chunk_count_estimate(documents)
print(f"Will create approximately {estimate} chunks")

# Then split
chunks = chunker.split_documents(documents)
print(f"Actually created {len(chunks)} chunks")
```

### Multi-Source Pipeline

```python
from app.ingestion.pdf_loader import PDFDocumentLoader
from app.ingestion.web_loader import WebDocumentLoader
from app.ingestion.youtube_loader import YouTubeDocumentLoader
from app.core.text_splitter import TextChunker
from app.utils.config import get_settings

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

# Split all documents with consistent chunking
settings = get_settings()
chunker = TextChunker(settings)
chunks = chunker.split_documents(all_docs)

print(f"Total documents: {len(all_docs)}")
print(f"Total chunks: {len(chunks)}")
```

### RAG Pipeline Integration

```python
from app.core.text_splitter import TextChunker
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 1. Load documents (from any source)
documents = load_your_documents()

# 2. Split into chunks
chunker = TextChunker(get_settings())
chunks = chunker.split_documents(documents)

# 3. Create embeddings
embeddings = OpenAIEmbeddings()

# 4. Create vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Query
results = vectorstore.similarity_search("your query", k=4)
```

## Document Structure

### Input (LangChain Document)

```python
Document(
    page_content="Full document text...",
    metadata={
        "source": "example.pdf",
        "page": 5,
        "author": "John Doe"
    }
)
```

### Output (Chunked Documents)

```python
[
    Document(
        page_content="First chunk (≤1000 chars)...",
        metadata={
            "source": "example.pdf",      # Original metadata preserved
            "page": 5,
            "author": "John Doe",
            "chunk_index": 0,              # Added by chunker
            "total_chunks": 5,             # Added by chunker
            "chunk_size": 987              # Added by chunker
        }
    ),
    Document(
        page_content="Second chunk (≤1000 chars)...",
        metadata={
            "source": "example.pdf",
            "page": 5,
            "author": "John Doe",
            "chunk_index": 1,
            "total_chunks": 5,
            "chunk_size": 1000
        }
    ),
    # ... more chunks
]
```

## How RecursiveCharacterTextSplitter Works

The chunker uses a hierarchy of separators to preserve semantic coherence:

1. **Paragraphs** (`\n\n`) - Try to split on paragraph breaks first
2. **Sentences** (`\n`) - If chunks still too large, split on sentences
3. **Words** (` `) - If still too large, split on words
4. **Characters** - Last resort: split anywhere

This ensures chunks are as semantically coherent as possible.

## Error Handling

### Empty Document List

```python
chunks = chunker.split_documents([])  # Returns []
# No error, gracefully returns empty list
```

### Invalid Configuration

```python
# If CHUNK_SIZE=0 or negative in .env
try:
    chunker = TextChunker(settings)
except ValueError as e:
    print(e)
    # "chunk_size must be positive, got 0. Check CHUNK_SIZE in your .env configuration."
```

### Overlap >= Chunk Size

```python
# If CHUNK_OVERLAP >= CHUNK_SIZE in .env
try:
    chunker = TextChunker(settings)
except ValueError as e:
    print(e)
    # "chunk_overlap (1000) must be less than chunk_size (1000). ..."
```

### Invalid Document Objects

```python
# If documents don't have page_content attribute
invalid_docs = [{"text": "content"}]  # Not a Document object
try:
    chunks = chunker.split_documents(invalid_docs)
except ValueError as e:
    print(e)
    # "Document at index 0 is missing 'page_content' attribute..."
```

## Best Practices

### ✅ DO

- **Use consistent chunk sizes** across your entire RAG system
- **Experiment with chunk size** for your specific use case
- **Monitor chunk distribution** (min, max, average sizes)
- **Preserve metadata** - it helps with retrieval and debugging
- **Estimate costs** using `get_chunk_count_estimate()` before splitting
- **Cache chunks** to avoid re-splitting same documents

### ❌ DON'T

- **Don't hardcode chunk parameters** - use Settings
- **Don't ignore validation errors** - fix your .env configuration
- **Don't skip metadata** - it's lightweight and valuable
- **Don't use huge chunks** (>2000 chars) - retrieval precision suffers
- **Don't use tiny chunks** (<300 chars) - loses context
- **Don't forget overlap** - helps with boundary cases

## Performance Characteristics

- **Time Complexity**: O(n × m) where n = number of docs, m = avg chunks per doc
- **Space Complexity**: O(total_text_size)
- **Throughput**: 1-10 MB/sec (depends on text structure)
- **Memory**: ~2x total text size during splitting

## Troubleshooting

### Too Many Chunks

**Problem**: Getting more chunks than expected

**Solutions**:
- Increase `CHUNK_SIZE` in .env
- Reduce `CHUNK_OVERLAP`
- Filter out short/irrelevant documents before chunking

### Poor Retrieval Quality

**Problem**: RAG returns irrelevant chunks

**Solutions**:
- Adjust chunk size (try 800-1500 range)
- Increase overlap to 20-25%
- Consider semantic chunking (see TODOs)
- Improve query formulation

### Chunks Too Large

**Problem**: Chunks exceed desired size

**Causes**: RecursiveCharacterTextSplitter is a guideline, not strict limit

**Solutions**:
- Reduce `CHUNK_SIZE` by 10-20%
- Use token-based splitting (see TODOs)
- Pre-process documents to add more paragraph breaks

### Metadata Missing

**Problem**: Original metadata not in chunks

**Cause**: Not using LangChain Document objects

**Solution**: Ensure input documents are `langchain.schema.Document` objects

## Advanced Features (TODOs)

The implementation includes extensive TODO comments for:

1. **Token-Based Splitting** - Use tiktoken for precise token counts
2. **Adaptive Chunk Sizes** - Different sizes for different document types
3. **Overlap Optimization** - A/B testing framework for optimal overlap
4. **Semantic Chunking** - Split at topic boundaries using embeddings
5. **Chunk Quality Metrics** - Analyze chunk distribution and quality

See [text_splitter.py](../app/core/text_splitter.py) for detailed implementation examples.

## Testing

Run the comprehensive test suite:

```bash
python tests/test_text_splitter.py
```

Tests include:
- Basic splitting
- Metadata preservation
- Empty list handling
- Multiple documents
- Parameter validation
- Chunk count estimation
- Overlap behavior
- Small documents
- Chunk metadata tracking

## Dependencies

Already in [requirements.txt](../../requirements.txt):

```text
langchain>=0.1.0
```

## Integration Points

**Upstream** (Inputs):
- PDFDocumentLoader
- WebDocumentLoader
- YouTubeDocumentLoader
- Any source that produces LangChain Documents

**Downstream** (Outputs):
- Embedding generators
- Vector stores (FAISS, Chroma, Pinecone)
- RAG retrievers
- Analytics pipelines

## Architecture

```
Document Loading → Text Chunking → Embedding → Vector Storage → Retrieval
                    ^^^^^^^^^^^^
                   (TextChunker)
```

The TextChunker is the critical preprocessing step between raw documents and vectorization.

## Next Steps

- **Vector Store Factory**: Create embeddings and store chunks
- **RAG Retriever**: Semantic search over chunked documents
- **RAG Chain**: Connect retriever + LLM for question answering
- **Evaluation**: Measure chunking impact on retrieval quality

## Support

For issues or questions:
1. Check TODO comments in [text_splitter.py](../app/core/text_splitter.py)
2. Review test cases in [test_text_splitter.py](../tests/test_text_splitter.py)
3. Refer to [LangChain text splitting docs](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
