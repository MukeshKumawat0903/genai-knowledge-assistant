# GenAI Knowledge Assistant (RAG & Agentic AI)

## 📌 Overview

GenAI Knowledge Assistant is an end-to-end Generative AI system designed to provide intelligent, source-aware answers across diverse knowledge sources. The system combines **Retrieval-Augmented Generation (RAG)** with **Agentic Workflows** to dynamically select the most appropriate tools and data sources for each query.

**Core Capabilities:**
- Answers questions using RAG over indexed documents and web content
- Employs agentic workflows for dynamic source selection and tool orchestration
- Supports multiple data sources: PDFs, web pages, Wikipedia, ArXiv, and YouTube videos
- Maintains conversational context across multi-turn interactions
- Delivers grounded, source-attributed responses for transparency and trust

---

## ✨ Key Features

- **RAG-Based Question Answering** – Semantic search over documents and web content with context-aware responses
- **YouTube Transcript Ingestion** – Extracts video transcripts for indexing and Q&A
- **Agentic Tool Selection** – Automatically routes queries to Web Search, Wikipedia, ArXiv, or local PDFs
- **Conversational Memory** – Maintains chat history for coherent multi-turn conversations
- **Runtime Model Selection** – Switch between LLM models without restarting the application
- **Temperature Control** – Adjust response creativity/determinism on the fly
- **Vector Database Switching** – Choose between FAISS (in-memory) or Chroma (persistent) at runtime
- **Chat Export** – Export conversation history as JSON or plain text
- **Source Attribution** – Every response includes references to source documents for transparency

---

## 🏗️ System Architecture

The system follows a modular, layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT UI LAYER                         │
│              (Chat Interface, Settings, Export)                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANGCHAIN ORCHESTRATION                      │
│         (Agent Router, Chain Management, Memory)                │
└─────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   RAG PIPELINE  │   │  AGENT TOOLS    │   │  LLM ABSTRACTION│
│  - Retriever    │   │  - Web Search   │   │  - Groq         │
│  - Indexer      │   │  - Wikipedia    │   │  - Model Switch │
│  - Evaluator    │   │  - ArXiv        │   │  - Temperature  │
└─────────────────┘   └─────────────────┘   └─────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VECTOR STORE LAYER                         │
│               (FAISS - In Memory | Chroma - Persistent)         │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA INGESTION LAYER                       │
│        (PDF Loader | Web Loader | YouTube | Wikipedia | ArXiv)  │
└─────────────────────────────────────────────────────────────────┘
```

**Flow Summary:**
1. User submits a query through the Streamlit UI
2. The Agent Router determines whether to use RAG retrieval or external tools
3. For RAG queries: documents are retrieved from the vector store, context is assembled, and the LLM generates a response
4. For tool-based queries: the appropriate agent tool (Web, Wikipedia, ArXiv) is invoked
5. Conversational memory maintains context across interactions
6. Response is returned with source attribution

---

## 🧠 Models & Technologies

### LLMs
- **Groq-Hosted Models** – LLaMA 3, Mixtral, and other models via Groq API (fast inference)


### Embeddings
- **Hugging Face Sentence Transformers** – `all-MiniLM-L6-v2` and other models for semantic similarity

### Vector Stores
- **FAISS** – In-memory vector store for fast prototyping and development
- **Chroma** – Persistent vector store for production deployments

### Frameworks & Libraries
| Component | Technology |
|-----------|------------|
| Orchestration | LangChain |
| UI Framework | Streamlit |
| Embeddings | Hugging Face Transformers |
| Vector Search | FAISS, ChromaDB |
| Data Processing | PyPDF, BeautifulSoup, youtube-transcript-api |

---

## 🔄 Runtime Configuration

Users can modify system behavior at runtime without restarting the application:

| Setting | Description |
|---------|-------------|
| **LLM Model** | Switch between available models (e.g., LLaMA 3 70B, Mixtral) |
| **Temperature** | Adjust from 0.0 (deterministic) to 1.0 (creative) |
| **Vector Store** | Toggle between FAISS and Chroma |
| **Retrieval Parameters** | Adjust top-k results and similarity thresholds |

All configuration changes take effect immediately and apply to subsequent queries.

---

## 📄 Supported Data Sources

| Source | Description |
|--------|-------------|
| **PDF Documents** | Upload and index local PDF files for Q&A |
| **Web Pages** | Scrape and process content from any URL |
| **Wikipedia** | Direct integration for encyclopedic knowledge |
| **ArXiv Papers** | Access academic papers and research documents |
| **YouTube Videos** | Extract video transcripts for indexing and Q&A |

Each source is processed through the ingestion pipeline, chunked appropriately, embedded, and stored in the vector database for retrieval.

---

## 🧪 RAG Evaluation

The project includes a built-in evaluation framework for assessing retrieval quality:

- **Precision@k** – Measures the proportion of relevant documents in the top-k retrieved results
- **Recall@k** – Measures the proportion of all relevant documents that appear in top-k results
- **Configuration Comparison** – Compare retrieval performance across different chunk sizes, overlap settings, and embedding models

> **Note:** Evaluation focuses on retrieval quality metrics. This project does not involve model training or fine-tuning.

---

## ▶️ How to Run Locally

### Prerequisites
- Python 3.9+
- pip package manager
- API key for LLM provider (Groq recommended)

### Installation Steps

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

## 🔐 Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# LLM Configuration
LLM_PROVIDER=groq
LLM_MODEL_NAME=llama3-70b-8192
GROQ_API_KEY=your_groq_api_key_here

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Store Configuration
VECTOR_STORE_TYPE=faiss
CHROMA_PERSIST_DIRECTORY=./data/vector_store/chroma
```

> **Security Note:** Never commit `.env` files with real API keys to version control.

---

## 📤 Export & Usability

The chat interface supports exporting conversation history for analysis and debugging:

| Format | Use Case |
|--------|----------|
| **JSON** | Structured export for programmatic analysis, logging, and integration |
| **Plain Text** | Human-readable format for documentation and sharing |

Exports include timestamps, queries, responses, and source references.

---

## 🎯 Use Cases

- **Research Assistance** – Query academic papers, documentation, and web sources from a unified interface
- **Document Analysis** – Upload and interrogate PDF reports, manuals, and technical documents
- **Video Content Q&A** – Extract and query YouTube video transcripts for educational content
- **Technical Learning** – Build a personal knowledge base from tutorials, documentation, and articles
- **Knowledge Exploration** – Discover connections across diverse sources with agentic search

---

## 🚫 What This Project Does NOT Do

For clarity and honesty:

- **No Model Training** – This project uses pre-trained LLMs via API; no training is performed
- **No Fine-Tuning** – Models are used as-is without parameter updates
- **No RLHF** – No reinforcement learning from human feedback is implemented
- **No Custom Model Development** – Focus is on orchestration, retrieval, and application logic

This is an **application-layer project** demonstrating RAG and agentic AI patterns using existing foundation models.

---

## 📁 Project Structure

```
genai-knowledge-assistant/
├── app/
│   ├── agents/          # Agent tools and routing logic
│   ├── core/            # LLM, embeddings, memory, vector store
│   ├── ingestion/       # Data loaders (PDF, Web, YouTube)
│   ├── rag/             # Retriever, chain, evaluator
│   ├── ui/              # Streamlit chat interface
│   └── utils/           # Configuration and logging
├── data/
│   ├── raw/             # Source documents
│   ├── processed/       # Processed chunks
│   └── vector_store/    # Persisted indexes
├── demos/               # Component demonstration scripts
├── docs/                # Architecture and implementation guides
├── tests/               # Unit and integration tests
├── run.py               # Application entry point
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 📌 Resume-Ready Summary

> **GenAI Knowledge Assistant** – Designed and implemented an end-to-end Retrieval-Augmented Generation (RAG) system with agentic workflows using LangChain and Streamlit. The application supports multi-source ingestion (PDFs, web, YouTube, ArXiv), runtime model switching, conversational memory, and source-attributed responses. Demonstrates production-ready patterns for LLM orchestration, vector search, and intelligent tool selection.

---

## 📚 Documentation

Detailed documentation is available in the `/docs` directory:

- [Architecture Diagram](docs/ARCHITECTURE_DIAGRAM.md)
- [Chain Implementation](docs/reference/CHAIN_IMPLEMENTATION_SUMMARY.md)
- [Retriever Guide](docs/reference/RETRIEVER_IMPLEMENTATION_SUMMARY.md)
- [Evaluator Reference](docs/EVALUATOR_QUICK_REF.md)
- [YouTube Loader Guide](docs/YOUTUBE_LOADER_QUICK_REF.md)

---

## License

This project is for educational and portfolio purposes.

---

*Built with LangChain, Streamlit, and Hugging Face*
