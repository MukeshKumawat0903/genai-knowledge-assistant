# GenAI Knowledge Assistant (RAG & Agentic AI)

## Overview

GenAI Knowledge Assistant is an end-to-end Generative AI system that provides intelligent, source-aware answers across diverse knowledge sources. The system combines **Retrieval-Augmented Generation (RAG)** with **Agentic Workflows** powered by LangGraph to dynamically select the most appropriate tools and data sources for each query.

**Core Capabilities:**
- Answers questions using RAG over indexed documents and web content
- Employs LangGraph-backed agentic workflows for dynamic source selection and tool orchestration
- Supports multiple data sources: PDFs, web pages, Wikipedia, ArXiv, and YouTube videos
- Maintains session-based conversational context across multi-turn interactions
- Delivers grounded, source-attributed responses for transparency and trust

---

## Key Features

### Retrieval & Generation
- **RAG-Based Question Answering** – Semantic search over documents with grounding-enforced LLM responses (no hallucination outside provided context)
- **LangGraph Agentic Routing** – `create_agent` API (LangChain 1.2.0+) with LangGraph under the hood; ReAct-style reasoning with automatic tool fallback when tool calling fails
- **Source Attribution** – Every RAG response includes expandable source document references with metadata
- **Anti-Hallucination Grounding** – Prompt engineering forces LLM to answer only from retrieved context or say "I don't know"

### Data Ingestion
- **PDF Upload & Indexing** – Drag-and-drop PDF upload with auto-indexing support; stored per session under `data/raw/uploads/`
- **Web Page Scraping** – Batch URL ingestion via BeautifulSoup; one or multiple URLs per indexing run
- **YouTube Transcript Extraction** – Extracts and indexes video transcripts via `youtube-transcript-api`
- **Wikipedia & ArXiv** – Live lookup tools available to the agent (no pre-indexing needed)

### Runtime Configuration (No Restart Required)
- **Model Selection** – Switch between `llama-3.3-70b-versatile`, `llama-3.1-70b-versatile`, `llama-3.1-8b-instant` via UI
- **Temperature Control** – Slider from 0.0 (deterministic) to 1.0 (creative) applied per query
- **Vector Store Switching** – Toggle between FAISS and Chroma; each store is isolated in its own subdirectory to prevent data loss
- **Backend Mode Toggle** – Switch between Agent mode (multi-tool) and RAG mode (document retrieval) without restart

### Memory & Export
- **Session-Based Memory** – In-memory `ConversationMemoryManager` maintains per-session chat history for multi-turn context
- **Chat Export (JSON)** – Structured export with role, content, timestamp, and source metadata
- **Chat Export (Plain Text)** – Human-readable format with separators and source attribution
- **Session Info Panel** – Live session ID and message count in sidebar

### Evaluation
- **Precision@K** – Proportion of relevant documents in top-k retrieved results
- **Recall@K** – Proportion of all relevant documents appearing in top-k results
- **F1@K** – Harmonic mean of precision and recall at K
- **MRR (Mean Reciprocal Rank)** – Rank position of the first relevant document
- **Hit Rate@K** – Whether at least one relevant document was retrieved in top-k

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT UI LAYER                         │
│   (Chat Interface | Model Controls | Ingestion | Export)        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│            LANGCHAIN + LANGGRAPH ORCHESTRATION                  │
│      (KnowledgeAgent | RAGChain | ConversationMemory)           │
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   RAG PIPELINE  │   │  AGENT TOOLS    │   │  LLM ABSTRACTION│
│  - RAGRetriever │   │  - Web Search   │   │  - GroqLLM      │
│  - Indexer      │   │  - Wikipedia    │   │  - LLMFactory   │
│  - RAGEvaluator │   │  - ArXiv        │   │  - BaseLLM ABC  │
│  - RAGChain     │   │  - PDF Search   │   │  - Runtime Swap │
└─────────────────┘   └─────────────────┘   └─────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VECTOR STORE LAYER                         │
│          FAISS (data/vector_store/faiss/)                       │
│          Chroma (data/vector_store/chroma/)  — isolated dirs    │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA INGESTION LAYER                       │
│   PDFDocumentLoader | WebDocumentLoader | YouTubeDocumentLoader │
│   BaseDocumentLoader ABC | TextSplitter | EmbeddingManager      │
└─────────────────────────────────────────────────────────────────┘
```

**Query Flow:**
1. User submits a query through the Streamlit chat interface
2. Backend mode selector routes to Agent or RAG chain
3. **Agent mode:** `KnowledgeAgent` (LangGraph) reasons and selects among Web Search, Wikipedia, ArXiv, or PDF Knowledge Search
4. **RAG mode:** Documents are retrieved from the vector store → context assembled → LLM generates grounded response
5. Session memory maintains conversational context for follow-up questions
6. Response is returned with expandable source attribution

---

## Models & Technologies

### LLM Provider
| Model | Use Case |
|-------|----------|
| `llama-3.3-70b-versatile` | Default — best quality, recommended for complex queries |
| `llama-3.1-70b-versatile` | Balanced quality and speed |
| `llama-3.1-8b-instant` | Fastest inference, good for simple queries |

All models served via **Groq API** (ultra-fast inference, generous free tier).

### Embeddings
- **Hugging Face Sentence Transformers** – `all-MiniLM-L6-v2` for semantic similarity (local, no API cost)

### Vector Stores
| Store | Type | Best For |
|-------|------|----------|
| **FAISS** | In-memory | Development, fast prototyping |
| **Chroma** | Persistent on disk | Production, larger document sets |

### Frameworks & Libraries
| Component | Technology |
|-----------|------------|
| Agent Orchestration | LangChain 1.2.0 + LangGraph |
| UI Framework | Streamlit |
| Embeddings | Hugging Face Sentence Transformers |
| Vector Search | FAISS, ChromaDB |
| LLM Provider | Groq (ChatGroq) |
| PDF Processing | PyPDF |
| Web Scraping | BeautifulSoup4 |
| YouTube | youtube-transcript-api |
| Academic Search | ArXiv API, Wikipedia API |
| Web Search | DuckDuckGo (no API key needed) |

---

## Runtime Configuration

| Setting | Options | Notes |
|---------|---------|-------|
| **Backend Mode** | Agent / RAG | Agent uses tools dynamically; RAG requires indexed documents |
| **LLM Model** | 3 Groq models | Takes effect on next query after Apply Settings |
| **Temperature** | 0.0 – 1.0 (step 0.05) | Lower = more factual; Higher = more creative |
| **Vector Store** | FAISS / Chroma | Changing requires re-indexing documents |

All changes take effect immediately after clicking **Apply Settings** — no app restart needed.

---

## Supported Data Sources

| Source | Loader Class | Notes |
|--------|-------------|-------|
| **PDF Documents** | `PDFDocumentLoader` | Upload via UI; auto-saved and indexed per session |
| **Web Pages** | `WebDocumentLoader` | Batch URLs; BeautifulSoup-based extraction |
| **YouTube Videos** | `YouTubeDocumentLoader` | Transcript-based indexing via `youtube-transcript-api` |
| **Wikipedia** | Agent tool (live) | Real-time lookup; no pre-indexing required |
| **ArXiv Papers** | Agent tool (live) | Real-time academic search; no pre-indexing required |
| **DuckDuckGo Web** | Agent tool (live) | Real-time web search for current events and news |

---

## RAG Evaluation

The built-in `RAGEvaluator` class measures retrieval quality using manual relevance judgments:

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K retrieved docs that are relevant |
| **Recall@K** | Fraction of all relevant docs appearing in top-K results |
| **F1@K** | Harmonic mean of Precision@K and Recall@K |
| **MRR** | Mean Reciprocal Rank — position of first relevant document |
| **Hit Rate@K** | Binary: was at least one relevant doc retrieved? |

> Evaluation focuses on retrieval quality metrics. This project does not involve model training or fine-tuning.

---

## How to Run Locally

### Prerequisites
- Python 3.9+
- pip package manager
- Free Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/genai-knowledge-assistant.git
cd genai-knowledge-assistant

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Run the application
streamlit run run.py
```

The application will be available at `http://localhost:8501`

---

## Environment Variables

```env
# LLM Configuration
LLM_PROVIDER=groq
LLM_MODEL_NAME=llama-3.3-70b-versatile
GROQ_API_KEY=your_groq_api_key_here

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Store Configuration
VECTOR_STORE_TYPE=faiss
CHROMA_PERSIST_DIRECTORY=./data/vector_store/chroma
```

> **Security Note:** Never commit `.env` files with real API keys to version control.

---

## Export & Usability

| Format | Contents | Use Case |
|--------|----------|----------|
| **JSON** | role, content, timestamp, sources metadata | Programmatic analysis, logging, re-import |
| **Plain Text** | Formatted conversation with separators and source names | Documentation, sharing, quick review |

Exports are generated client-side in-browser — no server-side storage of conversation data.

---

## Use Cases

- **Research Assistance** – Query academic papers, documentation, and web sources from a unified interface
- **Document Analysis** – Upload and interrogate PDF reports, manuals, and technical documents
- **Video Content Q&A** – Extract and query YouTube video transcripts for educational content
- **Technical Learning** – Build a personal knowledge base from tutorials, documentation, and articles
- **Knowledge Exploration** – Discover connections across diverse sources with agentic search

---

## Current Limitations

- **Single LLM provider** – Only Groq is supported; OpenAI, Anthropic, and Google are stubbed but not implemented
- **In-memory session only** – Conversation history and session state reset on page reload (no persistent sessions)
- **No streaming** – Responses are delivered all-at-once after full generation (no token-by-token streaming)
- **Agent ↔ PDF retrieval gap** – In Agent mode, the PDF knowledge tool only works if a retriever is explicitly passed; the current UI wires Agent mode without a retriever, so the PDF tool returns "not initialized"
- **Manual evaluation only** – `RAGEvaluator` requires manual relevance judgments; no automated LLM-as-judge evaluation
- **No document management** – Indexed documents cannot be listed, previewed, or deleted through the UI
- **No reranking** – Retrieved chunks are not re-scored before being passed to the LLM

---

## What This Project Does NOT Do

- **No Model Training** – Uses pre-trained LLMs via API; no training is performed
- **No Fine-Tuning** – Models are used as-is without parameter updates
- **No RLHF** – No reinforcement learning from human feedback is implemented
- **No Custom Model Development** – Focus is on orchestration, retrieval, and application logic

This is an **application-layer project** demonstrating RAG and agentic AI patterns using existing foundation models.

---

## Project Structure

```
genai-knowledge-assistant/
├── app/
│   ├── agents/
│   │   ├── agent_router.py   # KnowledgeAgent (LangGraph create_agent)
│   │   └── tools.py          # PDF, Web, Wikipedia, ArXiv tool wrappers
│   ├── core/
│   │   ├── llm.py            # LLMFactory, GroqLLM, BaseLLM ABC
│   │   ├── embeddings.py     # EmbeddingManager (HuggingFace)
│   │   ├── memory.py         # ConversationMemoryManager (session-based)
│   │   ├── vector_store.py   # VectorStoreManager (FAISS/Chroma)
│   │   └── text_splitter.py  # TextSplitter for document chunking
│   ├── ingestion/
│   │   ├── base_loader.py    # BaseDocumentLoader ABC
│   │   ├── pdf_loader.py     # PDFDocumentLoader
│   │   ├── web_loader.py     # WebDocumentLoader (BeautifulSoup)
│   │   └── youtube_loader.py # YouTubeDocumentLoader
│   ├── rag/
│   │   ├── chain.py          # RAGChain (retrieval + grounded generation)
│   │   ├── retriever.py      # RAGRetriever
│   │   ├── indexer.py        # Document indexer
│   │   └── evaluator.py      # RAGEvaluator (Precision, Recall, F1, MRR)
│   ├── ui/
│   │   └── chat_app.py       # Streamlit UI (chat, sidebar, export)
│   └── utils/
│       ├── config.py         # Settings (pydantic-based config)
│       └── logger.py         # Logging setup
├── data/
│   ├── raw/                  # Source documents (PDFs, etc.)
│   ├── processed/            # Processed chunks
│   └── vector_store/
│       ├── faiss/            # FAISS index (isolated directory)
│       └── chroma/           # Chroma collections (isolated directory)
├── demos/                    # Component demonstration scripts
├── docs/                     # Architecture and implementation guides
├── tests/                    # Unit and integration tests
├── run.py                    # Application entry point
├── requirements.txt          # Python dependencies
└── future_plan.md            # Roadmap and planned improvements
```

---

## Resume-Ready Summary

> **GenAI Knowledge Assistant** – Designed and implemented an end-to-end Retrieval-Augmented Generation (RAG) system with LangGraph-backed agentic workflows using LangChain 1.2.0 and Streamlit. Supports multi-source ingestion (PDFs, web scraping, YouTube transcripts, ArXiv, Wikipedia), runtime model switching between Groq-hosted LLaMA 3 models, session-based conversational memory, and source-attributed grounded responses. Features a modular architecture with a clean LLM provider abstraction layer (Factory pattern), isolated dual vector store support (FAISS/Chroma), and a built-in retrieval evaluation framework (Precision@K, Recall@K, F1, MRR). Demonstrates production-ready patterns for LLM orchestration, semantic vector search, and intelligent agentic tool selection.

---

## Documentation

Detailed documentation is available in the `/docs` directory:

- [Chain Implementation](docs/reference/CHAIN_IMPLEMENTATION_SUMMARY.md)
- [Retriever Guide](docs/reference/RETRIEVER_IMPLEMENTATION_SUMMARY.md)
- [Evaluator Reference](docs/EVALUATOR_QUICK_REF.md)
- [YouTube Loader Guide](docs/YOUTUBE_LOADER_QUICK_REF.md)
- [Embedding Guide](docs/EMBEDDING_GUIDE.md)
- [Future Roadmap](future_plan.md)

---

## License

This project is for educational and portfolio purposes.

---

*Built with LangChain, LangGraph, Streamlit, Groq, and Hugging Face*
