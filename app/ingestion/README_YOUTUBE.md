# YouTube Document Loader

Clean, production-ready loader for YouTube video transcripts.

## Quick Start

```python
from app.ingestion.youtube_loader import YouTubeDocumentLoader

# Load a video transcript
loader = YouTubeDocumentLoader(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    language="en"
)
documents = loader.load_documents()

print(f"Loaded: {len(documents)} document(s)")
print(f"Video ID: {documents[0].metadata['video_id']}")
print(f"Content: {documents[0].page_content[:200]}...")
```

## Features

✅ **Simple API** - Extends `BaseDocumentLoader` interface  
✅ **Robust Validation** - Validates URLs and extracts video IDs  
✅ **Multiple URL Formats** - Supports youtube.com, youtu.be, embed, mobile  
✅ **Language Support** - Specify transcript language  
✅ **Rich Metadata** - Adds source, video_id, loader_type, timestamps  
✅ **Clear Error Messages** - Helpful debugging for missing transcripts  
✅ **LangChain Integration** - Uses `YoutubeLoader` under the hood

## Installation

```bash
pip install langchain langchain-community youtube-transcript-api
```

## Usage

### Basic Loading

```python
loader = YouTubeDocumentLoader(video_url="https://www.youtube.com/watch?v=VIDEO_ID")
docs = loader.load_documents()
```

### With Language Selection

```python
loader = YouTubeDocumentLoader(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    language="es"  # Spanish transcript
)
docs = loader.load_documents()
```

### With Custom Source Name

```python
loader = YouTubeDocumentLoader(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    source_name="Machine Learning Tutorial Series"
)
docs = loader.load_documents()
```

### Batch Loading

```python
video_urls = [
    "https://www.youtube.com/watch?v=VIDEO_ID_1",
    "https://www.youtube.com/watch?v=VIDEO_ID_2",
    "https://www.youtube.com/watch?v=VIDEO_ID_3",
]

all_docs = []
for url in video_urls:
    try:
        loader = YouTubeDocumentLoader(video_url=url)
        docs = loader.load_documents()
        all_docs.extend(docs)
    except ValueError as e:
        print(f"Failed to load {url}: {e}")

print(f"Successfully loaded {len(all_docs)} videos")
```

## Document Structure

Each document contains:

```python
{
    "page_content": "Full video transcript...",
    "metadata": {
        "source": "https://www.youtube.com/watch?v=VIDEO_ID",
        "video_id": "VIDEO_ID",
        "loader_type": "youtube",
        "ingestion_time": "2024-01-15T10:30:00",
        "source_name": "Optional custom name"
    }
}
```

## Supported URL Formats

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`
- `https://www.youtube.com/v/VIDEO_ID`
- `https://m.youtube.com/watch?v=VIDEO_ID`

## Error Handling

```python
try:
    loader = YouTubeDocumentLoader(video_url=url)
    docs = loader.load_documents()
except ValueError as e:
    if "transcript" in str(e).lower():
        print("Video has no transcript available")
    else:
        print(f"Invalid URL or video: {e}")
```

## Common Issues

### "Transcript not available"

**Causes:**
- Video has captions disabled
- Wrong language specified
- Video is private/age-restricted
- Video is too new (transcripts processing)

**Solutions:**
- Check if video has captions enabled
- Try language="en" for English
- Use publicly available videos
- Wait for transcript generation

### "Invalid video ID"

**Causes:**
- Malformed URL
- Video ID not 11 characters
- Invalid characters in video ID

**Solutions:**
- Use standard YouTube URLs
- Verify video ID is correct
- Check URL formatting

## RAG Integration

```python
from app.ingestion.youtube_loader import YouTubeDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 1. Load transcript
loader = YouTubeDocumentLoader(video_url="...")
docs = loader.load_documents()

# 2. Split for better retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Query
results = vectorstore.similarity_search("What did the speaker say about AI?", k=3)
```

## Multi-Source RAG

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

# Create unified vector store
vectorstore = create_vectorstore(all_docs)
```

## Testing

Run the test suite:

```bash
python tests/test_youtube_loader.py
```

Tests include:
- Video ID extraction from various URL formats
- Invalid URL handling
- Interface compliance with BaseDocumentLoader
- Error message clarity
- Interactive testing with custom URLs

## Documentation

- **Quick Reference**: [YOUTUBE_LOADER_QUICK_REF.md](../../docs/YOUTUBE_LOADER_QUICK_REF.md)
- **Implementation Details**: [YOUTUBE_LOADER_IMPLEMENTATION_SUMMARY.md](../../docs/reference/YOUTUBE_LOADER_IMPLEMENTATION_SUMMARY.md)
- **Source Code**: [youtube_loader.py](youtube_loader.py)

## TODO Features

The implementation includes extensive TODO comments for future enhancements:

1. **Chunking Long Transcripts** - Split with timestamp preservation
2. **Multi-Language Support** - Language fallback mechanism
3. **Caption Type Selection** - Choose manual vs auto-generated
4. **Playlist Support** - Batch load entire playlists
5. **Video Metadata Enrichment** - Add title, views, duration via YouTube API

See the source code for detailed implementation examples.

## Design Patterns

- **Template Method**: Extends BaseDocumentLoader interface
- **Adapter**: Wraps LangChain's YoutubeLoader
- **Strategy**: Configurable language and error handling

## Dependencies

```text
langchain>=0.1.0
langchain-community>=0.0.10
youtube-transcript-api>=0.6.1
```

## Best Practices

✅ **DO:**
- Validate URLs before loading
- Handle errors gracefully with try-except
- Add source names for better traceability
- Chunk long transcripts for better retrieval
- Cache transcripts to avoid repeated API calls

❌ **DON'T:**
- Load videos without checking for transcripts
- Ignore metadata - it improves retrieval
- Skip error handling
- Load too many videos at once (rate limits)

## Support

For issues or questions:
1. Check TODO comments in source code
2. Review test cases
3. Refer to LangChain documentation

## License

Part of GenAI Knowledge Assistant project.
