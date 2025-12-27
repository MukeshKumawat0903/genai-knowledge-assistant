# Web Loader - Quick Reference

## One-Page Cheat Sheet

### Installation
```bash
pip install langchain langchain-community beautifulsoup4 requests
```

---

### Basic Usage

#### Load Single URL
```python
from app.ingestion.web_loader import WebDocumentLoader

loader = WebDocumentLoader(
    urls="https://example.com/article",
    source_name="Example Article"
)
documents = loader.load_documents()  # One Document per URL
```

#### Load Multiple URLs
```python
urls = [
    "https://blog.com/post1",
    "https://blog.com/post2",
    "https://blog.com/post3"
]
loader = WebDocumentLoader(urls=urls, source_name="Blog Posts")
documents = loader.load_documents()
```

---

### Complete RAG Pipeline
```python
from app.ingestion.web_loader import WebDocumentLoader
from app.core.embeddings import EmbeddingManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 1. Load web pages
urls = ["https://docs.example.com/intro", "https://docs.example.com/guide"]
loader = WebDocumentLoader(urls=urls)
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
results = retriever.get_relevant_documents("How do I get started?")
```

---

### Metadata Structure
| Field | Description | Example |
|-------|-------------|---------|
| `source` | URL of the page | `"https://example.com"` |
| `loader_type` | Always "web" | `"web"` |
| `ingestion_time` | Load timestamp | `"2024-12-21T10:30:00"` |
| `title` | Page title (if available) | `"Example Domain"` |

---

### Error Handling
```python
try:
    loader = WebDocumentLoader(urls="https://example.com")
    documents = loader.load_documents()
except ValueError as e:
    print(f"Error: {e}")
```

---

### Common Patterns

#### Group by Domain
```python
from urllib.parse import urlparse
from collections import defaultdict

documents = loader.load_documents()
by_domain = defaultdict(list)

for doc in documents:
    domain = urlparse(doc.metadata['source']).netloc
    by_domain[domain].append(doc)

for domain, docs in by_domain.items():
    print(f"{domain}: {len(docs)} documents")
```

#### Filter by URL Pattern
```python
documents = loader.load_documents()
blog_posts = [doc for doc in documents 
              if '/blog/' in doc.metadata['source']]
```

#### Extract Titles
```python
documents = loader.load_documents()
for doc in documents:
    title = doc.metadata.get('title', 'Untitled')
    url = doc.metadata['source']
    print(f"{title}: {url}")
```

---

### Do's and Don'ts

| ✅ DO | ❌ DON'T |
|-------|----------|
| Use https:// URLs when available | Use URLs without http/https |
| Handle fetch errors explicitly | Assume all URLs will load |
| Validate URLs before loading | Trust user-provided URLs |
| Add rate limiting for many URLs | Overwhelm servers with requests |
| Check robots.txt for crawling | Scrape without permission |

---

### Testing
```bash
python test_web_loader.py
```

---

### Interface Contract
WebDocumentLoader implements BaseDocumentLoader:
- **Required**: `load_documents() → List[Document]`
- **Inherited**: `_validate_source()`, `_add_metadata()`

---

### Future Enhancements (TODOs)

#### Rate Limiting
```python
import time
from collections import defaultdict

# Add delay between requests to same domain
def load_with_rate_limiting(delay_seconds=1.0):
    domain_last_request = defaultdict(float)
    for url in urls:
        domain = urlparse(url).netloc
        time_since_last = time.time() - domain_last_request[domain]
        if time_since_last < delay_seconds:
            time.sleep(delay_seconds - time_since_last)
        # Load URL...
```

#### JavaScript-Heavy Pages (Playwright)
```python
# Dependencies: pip install playwright && playwright install
from playwright.sync_api import sync_playwright

def load_with_playwright(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")
        text = page.inner_text("body")
        browser.close()
        return text
```

#### Web Crawling
```python
# Dependencies: Already in requirements
from langchain_community.document_loaders import RecursiveUrlLoader

def crawl_website(start_url, max_depth=2):
    loader = RecursiveUrlLoader(
        url=start_url,
        max_depth=max_depth,
        prevent_outside=True  # Stay on same domain
    )
    return loader.load()
```

---

### Related Resources
- Full Guide: `docs/WEB_LOADER_GUIDE.md`
- Base Loader: `docs/BASE_LOADER_GUIDE.md`
- PDF Loader: `PDF_LOADER_QUICK_REF.md`
- Embeddings: `EMBEDDING_QUICK_REF.md`

---

### Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| `ValueError: URL must start with http://` | Add http:// or https:// to URL |
| `ValueError: URL must contain domain` | Check URL format (e.g., "http://domain.com") |
| Failed to load content | Check internet connection, verify URL is accessible |
| Import errors | `pip install langchain langchain-community beautifulsoup4` |
| Empty content | Page might be JavaScript-heavy (see Playwright TODO) |
| Rate limited | Add delays between requests (see rate limiting TODO) |

---

### Safe Loading Function
```python
from typing import List

def safe_load_urls(urls: List[str], source_name: str = None):
    """Safely load URLs with error handling."""
    try:
        loader = WebDocumentLoader(urls=urls, source_name=source_name)
        documents = loader.load_documents()
        print(f"✓ Loaded {len(documents)} documents")
        return documents
    except ValueError as e:
        print(f"✗ Invalid URLs: {e}")
        return []
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return []

documents = safe_load_urls(["https://example.com"])
```

---

### Integration Example
```python
from app.ingestion.web_loader import WebDocumentLoader
from app.core.embeddings import EmbeddingManager
from app.core.llm import LLMFactory
from app.utils.config import get_settings

# Load web pages
urls = ["https://docs.example.com/getting-started"]
loader = WebDocumentLoader(urls=urls, source_name="Documentation")
documents = loader.load_documents()

# Embed
embedding_manager = EmbeddingManager()
vectorstore = FAISS.from_documents(documents, embedding_manager.get_embeddings())

# Query with LLM
settings = get_settings()
llm = LLMFactory.create(settings)
chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
result = chain({"query": "How do I install the product?"})
```

---

**Production-Ready**: URL validation, error handling, metadata tracking, and extensibility built-in.

**Note**: WebBaseLoader extracts static HTML content. For JavaScript-heavy sites, see Playwright/Selenium TODOs.
