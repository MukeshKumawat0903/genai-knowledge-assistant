# YouTube Document Loader - Quick Reference

## Overview

The `YouTubeDocumentLoader` loads video transcripts from YouTube videos for use in RAG (Retrieval-Augmented Generation) systems. It extends the `BaseDocumentLoader` interface and uses LangChain's `YoutubeLoader` under the hood.

## Installation

Ensure the following packages are installed:

```bash
pip install langchain langchain-community youtube-transcript-api
```

## Basic Usage

### Load a Single Video

```python
from app.ingestion.youtube_loader import YouTubeDocumentLoader

# Create loader
loader = YouTubeDocumentLoader(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    language="en"
)

# Load transcript
documents = loader.load_documents()

# Access content and metadata
for doc in documents:
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")
```

### With Custom Source Name

```python
loader = YouTubeDocumentLoader(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    language="en",
    source_name="Tech Tutorial Series"
)

documents = loader.load_documents()
```

## Supported URL Formats

The loader accepts various YouTube URL formats:

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`
- `https://www.youtube.com/v/VIDEO_ID`
- `https://m.youtube.com/watch?v=VIDEO_ID` (mobile)

## Document Structure

Each document returned contains:

### Content
- Full video transcript as a single text string
- Includes all spoken words from the video

### Metadata
- `source`: The video URL
- `video_id`: 11-character YouTube video ID
- `loader_type`: Always "youtube"
- `ingestion_time`: ISO timestamp of when loaded
- `source_name`: Optional custom name (if provided)

Example:
```python
{
    "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "video_id": "dQw4w9WgXcQ",
    "loader_type": "youtube",
    "ingestion_time": "2024-01-15T10:30:00",
    "source_name": "Rick Astley"
}
```

## Language Support

Specify the transcript language using the `language` parameter:

```python
# English transcript
loader = YouTubeDocumentLoader(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    language="en"
)

# Spanish transcript
loader = YouTubeDocumentLoader(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    language="es"
)
```

**Common language codes:**
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `ja` - Japanese
- `zh` - Chinese
- `pt` - Portuguese
- `ru` - Russian

## Error Handling

### No Transcript Available

```python
try:
    loader = YouTubeDocumentLoader(video_url=url)
    documents = loader.load_documents()
except ValueError as e:
    if "transcript" in str(e).lower():
        print("Video has no transcripts available")
        print("Possible reasons:")
        print("  - Captions are disabled")
        print("  - Video is too new (transcripts pending)")
        print("  - Language not supported")
```

### Invalid Video URL

```python
try:
    loader = YouTubeDocumentLoader(video_url=invalid_url)
except ValueError as e:
    print(f"Invalid URL: {e}")
```

### Wrong Language

```python
try:
    loader = YouTubeDocumentLoader(
        video_url=url,
        language="unsupported_lang"
    )
    documents = loader.load_documents()
except ValueError as e:
    print(f"Language not available: {e}")
```

## Integration with RAG Pipeline

### Step 1: Load YouTube Transcripts

```python
from app.ingestion.youtube_loader import YouTubeDocumentLoader

video_urls = [
    "https://www.youtube.com/watch?v=VIDEO_ID_1",
    "https://www.youtube.com/watch?v=VIDEO_ID_2",
    "https://www.youtube.com/watch?v=VIDEO_ID_3",
]

all_documents = []
for url in video_urls:
    loader = YouTubeDocumentLoader(video_url=url)
    documents = loader.load_documents()
    all_documents.extend(documents)

print(f"Loaded {len(all_documents)} video transcripts")
```

### Step 2: Split Documents (if needed)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(all_documents)
print(f"Split into {len(chunks)} chunks")
```

### Step 3: Create Embeddings and Store

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

print("Vector store created successfully")
```

### Step 4: Query the System

```python
query = "What did the speaker say about machine learning?"
results = vectorstore.similarity_search(query, k=3)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Content: {result.page_content[:200]}...")
    print(f"Source: {result.metadata['source']}")
    print(f"Video ID: {result.metadata['video_id']}")
```

## Best Practices

### ✅ DO

- **Check for transcripts before loading**: Not all videos have transcripts
- **Use appropriate language codes**: Match the video's language
- **Handle errors gracefully**: Wrap load calls in try-except blocks
- **Add source names**: Help identify content in RAG responses
- **Chunk long transcripts**: Split for better retrieval accuracy
- **Cache loaded transcripts**: Avoid repeated API calls

### ❌ DON'T

- **Don't load videos without transcripts**: Will raise ValueError
- **Don't use invalid URLs**: Validate URLs before loading
- **Don't ignore metadata**: Rich metadata improves retrieval
- **Don't skip error handling**: Always catch ValueError exceptions
- **Don't load too many videos at once**: Process in batches
- **Don't forget rate limits**: YouTube API has usage limits

## Common Patterns

### Batch Loading with Error Recovery

```python
def load_multiple_videos(urls: list[str]) -> list[Document]:
    """Load multiple videos with error handling."""
    documents = []
    failed = []
    
    for url in urls:
        try:
            loader = YouTubeDocumentLoader(video_url=url)
            docs = loader.load_documents()
            documents.extend(docs)
            print(f"✓ Loaded: {url}")
        except ValueError as e:
            failed.append((url, str(e)))
            print(f"✗ Failed: {url} - {e}")
        except Exception as e:
            failed.append((url, str(e)))
            print(f"✗ Error: {url} - {e}")
    
    print(f"\nSuccessfully loaded: {len(documents)}")
    print(f"Failed: {len(failed)}")
    
    return documents
```

### Multi-Language Loading

```python
def load_with_fallback(url: str, languages: list[str]) -> list[Document]:
    """Try multiple languages in order."""
    for lang in languages:
        try:
            loader = YouTubeDocumentLoader(video_url=url, language=lang)
            return loader.load_documents()
        except ValueError as e:
            if "transcript" in str(e).lower():
                continue
            raise
    
    raise ValueError(f"No transcripts available in languages: {languages}")

# Try English, then Spanish, then French
documents = load_with_fallback(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    languages=["en", "es", "fr"]
)
```

### Integration with Other Loaders

```python
from app.ingestion.pdf_loader import PDFDocumentLoader
from app.ingestion.web_loader import WebDocumentLoader
from app.ingestion.youtube_loader import YouTubeDocumentLoader

# Multi-source loading
all_docs = []

# Load PDFs
pdf_loader = PDFDocumentLoader("./documents")
all_docs.extend(pdf_loader.load_documents())

# Load web pages
web_loader = WebDocumentLoader(["https://example.com"])
all_docs.extend(web_loader.load_documents())

# Load YouTube videos
yt_loader = YouTubeDocumentLoader("https://www.youtube.com/watch?v=VIDEO_ID")
all_docs.extend(yt_loader.load_documents())

print(f"Total documents: {len(all_docs)}")
```

## Performance Considerations

- **Network Latency**: Loading transcripts requires API calls to YouTube
- **Transcript Length**: Long videos produce large documents (consider chunking)
- **Rate Limits**: YouTube may throttle excessive requests
- **Caching**: Cache transcripts to avoid repeated API calls

## Troubleshooting

### "Transcript not available"

**Causes:**
- Video has captions disabled
- Video is private or age-restricted
- Wrong language specified
- Video is too new (transcripts processing)

**Solutions:**
- Check if video has captions enabled
- Try different language codes
- Use publicly available videos
- Wait for transcript generation

### "Invalid video ID"

**Causes:**
- Malformed YouTube URL
- Video ID is not 11 characters
- URL contains invalid characters

**Solutions:**
- Validate URL format
- Use standard YouTube URLs
- Check video ID length

### "Could not retrieve transcript"

**Causes:**
- Network connectivity issues
- YouTube API temporarily unavailable
- Rate limiting

**Solutions:**
- Check internet connection
- Retry after a delay
- Implement exponential backoff

## Next Steps

- See [YOUTUBE_LOADER_IMPLEMENTATION_SUMMARY.md](reference/YOUTUBE_LOADER_IMPLEMENTATION_SUMMARY.md) for implementation details
- Check [youtube_loader.py](../app/ingestion/youtube_loader.py) for TODO items and future enhancements
- Run [test_youtube_loader.py](../tests/test_youtube_loader.py) to validate functionality

## Support

For issues or questions:
1. Check the [TODO comments](../app/ingestion/youtube_loader.py) in the source code
2. Review test cases in [test_youtube_loader.py](../tests/test_youtube_loader.py)
3. Refer to [LangChain YoutubeLoader documentation](https://python.langchain.com/docs/integrations/document_loaders/youtube_transcript)
