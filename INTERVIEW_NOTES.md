# GenAI Knowledge Assistant - Interview Notes

## 🎯 Project Overview

**GenAI Knowledge Assistant** is an end-to-end Retrieval-Augmented Generation (RAG) system with Agentic AI workflows. It provides intelligent, source-aware answers across diverse knowledge sources including PDFs, web pages, Wikipedia, ArXiv, and YouTube videos.

**Project Type**: Production-grade RAG System with Multi-Agent Architecture  
**Purpose**: Question-answering system with dynamic source selection and conversational memory  
**Architecture**: Modular, layered design with clear separation of concerns

---

## 🏗️ System Architecture

### **High-Level Architecture**
```
┌─────────────────────────────────────────┐
│         Streamlit UI Layer              │
│    (Chat Interface, Settings)           │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      LangChain Orchestration            │
│   (Agent Router, Chain, Memory)         │
└──────────┬──────────────────────────────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
┌────────┬────────┬────────┐
│ RAG    │ Agent  │  LLM   │
│Pipeline│ Tools  │Abstract│
└───┬────┴────────┴────────┘
    │
┌───▼─────────────────────┐
│   Vector Store Layer    │
│   (FAISS / Chroma)      │
└───┬─────────────────────┘
    │
┌───▼─────────────────────┐
│  Data Ingestion Layer   │
│ (PDF/Web/YouTube/etc)   │
└─────────────────────────┘
```

---

## 🧠 Technologies & Tools

### **1. Large Language Models (LLMs)**

#### **Groq API**
- **Purpose**: Cloud-hosted LLM inference with ultra-fast performance
- **Models Supported**: 
  - `llama-3.1-70b-versatile`
  - `llama-3.3-70b-versatile`
  - `mixtral-8x7b-32768`
  - `gemma2-9b-it`
- **Key Features**:
  - Fast inference (lower latency than OpenAI)
  - Free tier available
  - Support for temperature control
  - Runtime model switching
- **Implementation**: `app/core/llm.py` using `ChatGroq` from `langchain-groq`
- **Interview Points**:
  - Factory pattern for LLM abstraction (prevents vendor lock-in)
  - BaseLLM interface for easy provider switching
  - Configuration-driven initialization

---

### **2. LangChain Framework**

#### **Core Package (`langchain==1.2.0`)**
- **Purpose**: Orchestration framework for LLM applications
- **Key Components Used**:
  - **Chains**: Sequential logic for RAG pipeline
  - **Agents**: ReAct-pattern agents for tool selection
  - **Memory**: Conversation history management
  - **Document Loaders**: PDF, web, YouTube ingestion
  - **Vector Store Integration**: FAISS and Chroma connectors

#### **LangChain Community (`langchain-community==0.4.1`)**
- **Tools Used**:
  - `WikipediaQueryRun` - Wikipedia search
  - `ArxivQueryRun` - Academic paper search
  - `DuckDuckGoSearchRun` - Web search
  - `PyPDFLoader` - PDF document loading
  - `YoutubeLoader` - YouTube transcript extraction
  - `ChatMessageHistory` - Conversation memory

#### **LangChain Groq (`langchain-groq==1.1.1`)**
- Integration layer for Groq LLMs
- Full LangGraph support for modern agent workflows

#### **LangGraph (`langgraph>=1.0.2`)**
- **Purpose**: Modern agent framework (part of LangChain ecosystem)
- **Features**: State machines for complex agent workflows
- **Usage**: Powers `create_agent` in agent router

#### **Interview Points**:
- Why LangChain? Abstracts away LLM provider differences
- Agent design: ReAct pattern (Reasoning + Acting)
- Chain design: Retrieval → Context Assembly → Generation
- Memory management: Session-based with `ChatMessageHistory`

---

### **3. Embeddings**

#### **Hugging Face Transformers (`sentence-transformers>=2.3.0`)**
- **Purpose**: Convert text to dense vector representations for semantic search
- **Default Model**: `all-MiniLM-L6-v2`
  - 384-dimensional vectors
  - Fast inference on CPU
  - Optimized for semantic similarity
  - ~90MB model size
- **Why Embeddings?**: Enable semantic search (meaning-based) vs keyword search
- **Implementation**: `app/core/embeddings.py` using `HuggingFaceEmbeddings`
- **Key Design**: Lazy initialization with caching for efficiency

#### **Interview Points**:
- Embeddings transform text into vectors where similar meanings = close vectors
- Example: "What is AI?" and "Explain artificial intelligence" → similar vectors
- Local/offline execution (no API keys needed)
- Compatible with LangChain ecosystem

---

### **4. Vector Databases**

#### **FAISS (Facebook AI Similarity Search) (`faiss-cpu>=1.7.4`)**
- **Purpose**: In-memory vector store for fast similarity search
- **Pros**:
  - Fastest search performance
  - No external dependencies
  - Great for development/prototyping
- **Cons**:
  - Not persistent (must save/load manually)
  - Limited to single machine memory
- **Use Case**: Development, datasets <1M vectors

#### **ChromaDB (`chromadb>=0.4.20`)**
- **Purpose**: Persistent vector database with built-in storage
- **Pros**:
  - Auto-persistence (no manual save/load)
  - Metadata filtering support
  - Easy API
- **Cons**:
  - Slower than FAISS for large datasets
  - Not horizontally scalable
- **Use Case**: Production (small-medium datasets), local deployments

#### **Implementation**: `app/core/vector_store.py`
- Factory pattern for easy switching
- Unified interface for both stores
- Runtime selection via configuration

#### **Interview Points**:
- Vector stores enable semantic retrieval in RAG
- Different stores = different trade-offs (speed vs persistence vs scale)
- Abstraction allows experimentation without code changes

---

### **5. Document Ingestion**

#### **PDF Processing (`pypdf>=4.0.0`)**
- **Purpose**: Extract text from PDF documents
- **Implementation**: `app/ingestion/pdf_loader.py`
- **Features**:
  - Single file or directory batch loading
  - Automatic metadata tagging
  - Uses `PyPDFLoader` from LangChain

#### **Web Scraping (`beautifulsoup4>=4.12.0`, `requests>=2.31.0`)**
- **Purpose**: Extract content from web pages
- **Implementation**: `app/ingestion/web_loader.py`
- **Features**:
  - Clean HTML parsing
  - Automatic text extraction
  - Metadata preservation

#### **YouTube Transcripts (`youtube-transcript-api>=0.6.1`)**
- **Purpose**: Extract video transcripts as searchable text
- **Implementation**: `app/ingestion/youtube_loader.py`
- **Features**:
  - Video URL support (youtube.com and youtu.be)
  - Language selection
  - Automatic video ID extraction
  - Uses `YoutubeLoader` from LangChain

#### **Interview Points**:
- All loaders extend `BaseDocumentLoader` (Template Method pattern)
- Standardized metadata enables better tracking
- Directory loading enables batch processing

---

### **6. Text Processing**

#### **Text Chunking**
- **Implementation**: `app/core/text_splitter.py`
- **Purpose**: Split large documents into smaller chunks for embedding
- **Why?**: 
  - Embedding models have token limits
  - Smaller chunks = more precise retrieval
  - Enables semantic search at paragraph/section level
- **Strategy**: 
  - RecursiveCharacterTextSplitter (respects document structure)
  - Configurable chunk size and overlap
  - Preserves context with overlap

#### **Tokenization (`tiktoken>=0.5.2`)**
- **Purpose**: Token counting for prompt management
- **Usage**: Ensure prompts fit within LLM context windows

---

### **7. RAG Pipeline**

#### **RAG Chain (`app/rag/chain.py`)**
- **Workflow**:
  1. Retrieve relevant documents using retriever
  2. Format documents into context string
  3. Create grounded prompt (context + question)
  4. Generate answer using LLM
  5. Return answer with source documents
- **Key Features**:
  - Source grounding (reduces hallucinations)
  - Chat history integration
  - Source attribution in responses

#### **RAG Retriever (`app/rag/retriever.py`)**
- **Purpose**: Fetch relevant documents from vector store
- **Strategy**: Similarity search with configurable top-k
- **Features**: Metadata filtering, score thresholds

#### **RAG Indexer (`app/rag/indexer.py`)**
- **Purpose**: Process and index documents into vector store
- **Workflow**:
  1. Load documents (PDF, web, YouTube)
  2. Split into chunks
  3. Generate embeddings
  4. Store in vector database

#### **RAG Evaluator (`app/rag/evaluator.py`)**
- **Purpose**: Evaluate RAG pipeline quality
- **Metrics**: Retrieval accuracy, answer relevance, source grounding

---

### **8. Agentic AI Workflows**

#### **Agent Router (`app/agents/agent_router.py`)**
- **Pattern**: ReAct (Reasoning + Acting)
- **Purpose**: Dynamically select appropriate tools based on query
- **Architecture**:
  ```
  Question → Agent Thinks → Chooses Tool → Gets Result → Reasons → Answers
  ```
- **Tools Available**:
  1. **PDF Knowledge Search** - Search indexed documents
  2. **Web Search (DuckDuckGo)** - Current events, general web info
  3. **Wikipedia Search** - Encyclopedic knowledge
  4. **ArXiv Search** - Academic papers and research
- **Implementation**: Uses `create_agent` with LangGraph

#### **Agent Tools (`app/agents/tools.py`)**
- **Design**: Each tool wrapped as LangChain `BaseTool`
- **Input Schemas**: Pydantic models for validation
- **Error Handling**: Graceful fallbacks for unavailable resources

#### **Interview Points**:
- Agents provide dynamic routing (vs hardcoded logic)
- ReAct pattern: alternates thinking and acting
- Tool selection based on query semantics
- Example routing:
  - "What is AI?" → Wikipedia
  - "Latest AI news" → Web Search
  - "Research on transformers" → ArXiv
  - "Explain from our docs" → PDF Search

---

### **9. User Interface**

#### **Streamlit (`streamlit>=1.30.0`)**
- **Purpose**: Interactive web UI for chat interface
- **Implementation**: `app/ui/chat_app.py`
- **Features**:
  - Real-time chat interface
  - Message history display
  - Session-based conversations
  - Runtime settings (model selection, temperature control)
  - Vector store switching (FAISS ↔ Chroma)
  - Chat export (JSON/text)
  - Source document display

#### **Interview Points**:
- Session state management for conversation persistence
- Backend integration (RAG chain or agent)
- User-friendly parameter tuning
- Source attribution for transparency

---

### **10. Conversational Memory**

#### **Implementation**: `app/core/memory.py`
- **Purpose**: Maintain chat history across multi-turn conversations
- **Design**:
  - Session-based isolation (each user = separate history)
  - LangChain `ChatMessageHistory` integration
  - In-memory storage (extendable to Redis/PostgreSQL)
  - Stateless API (session_id passed explicitly)

#### **Features**:
- Per-session memory management
- Human/AI message tracking
- Memory clearing per session
- Compatible with RAG chains

#### **Interview Points**:
- Enables context-aware follow-up questions
- Session isolation for multi-user support
- Future: Memory windowing, summarization, persistent storage

---

### **11. Configuration & Utilities**

#### **Configuration (`app/utils/config.py`)**
- **Purpose**: Centralized settings management
- **Implementation**: Pydantic Settings with environment variables
- **Features**:
  - LLM configuration (model, temperature, API keys)
  - Embedding configuration (model name)
  - Vector store selection (FAISS/Chroma)
  - Chunking parameters
  - File paths

#### **Logging (`app/utils/logger.py`)**
- **Purpose**: Structured logging for debugging and monitoring
- **Features**: Configurable log levels, file/console output

#### **Environment Variables (`.env`)**
- `GROQ_API_KEY` - Groq API authentication
- Model selection, paths, etc.

---

### **12. Testing**

#### **Pytest (`pytest>=7.4.0`)**
- **Coverage**: Unit tests for all core modules
- **Test Files**:
  - `test_embeddings.py` - Embedding generation
  - `test_llm.py` - LLM factory and abstraction
  - `test_vector_store.py` - Vector store operations
  - `test_chain.py` - RAG chain workflow
  - `test_agent_tools.py` - Agent tool functionality
  - `test_pdf_loader.py` - PDF document loading
  - `test_youtube_loader.py` - YouTube transcript extraction
  - `test_retriever.py` - Document retrieval
  - `test_memory.py` - Conversation memory

#### **Test Configuration**: `pytest.ini`, `conftest.py`

---

### **13. Additional Tools & Libraries**

#### **DuckDuckGo Search (`ddgs>=1.0.0`)**
- Web search tool for agents
- No API key required

#### **Wikipedia (`wikipedia>=1.4.0`)**
- Wikipedia API wrapper for knowledge queries

#### **ArXiv (`arxiv>=2.0.0`)**
- Academic paper search

#### **Pydantic (`pydantic>=2.5.0`)**
- Data validation and settings management
- Type-safe configuration

#### **NumPy (`numpy>=1.24.0`)**
- Vector operations and numerical computing

#### **Python-dotenv (`python-dotenv>=1.0.0`)**
- Environment variable management

---

## 📊 Design Patterns Used

### **1. Factory Pattern**
- **Where**: `LLMFactory`, `VectorStoreManager`
- **Why**: Centralized object creation, easy provider switching

### **2. Template Method Pattern**
- **Where**: `BaseDocumentLoader` with `PDFDocumentLoader`, `YouTubeDocumentLoader`
- **Why**: Consistent interface, reusable loading logic

### **3. Strategy Pattern**
- **Where**: Vector store selection, text chunking strategies
- **Why**: Runtime algorithm selection

### **4. Singleton Pattern**
- **Where**: `EmbeddingManager` (cached embedding model)
- **Why**: Avoid reloading heavy models

### **5. Adapter Pattern**
- **Where**: LangChain tool wrappers
- **Why**: Uniform interface for diverse external APIs

---

## 🔑 Key Interview Talking Points

### **RAG Architecture**
- "I implemented a production-grade RAG system that combines retrieval from vector stores with LLM generation"
- "The system uses FAISS for fast in-memory search and ChromaDB for persistent storage"
- "Documents are chunked using RecursiveCharacterTextSplitter with configurable overlap to preserve context"

### **Agentic AI**
- "Built multi-agent system using ReAct pattern where LLM decides which tool to use"
- "Agents dynamically route queries to PDF search, web search, Wikipedia, or ArXiv based on query semantics"
- "Used LangGraph for state-based agent orchestration"

### **LLM Integration**
- "Abstracted LLM providers using Factory pattern for easy switching between Groq, OpenAI, Anthropic"
- "Implemented runtime model selection and temperature control"
- "Used Groq for fast inference with models like LLaMA 3.3 and Mixtral"

### **Vector Databases**
- "Implemented dual vector store support: FAISS for development, ChromaDB for production"
- "Embeddings generated using Hugging Face Sentence Transformers (all-MiniLM-L6-v2)"
- "Vector stores enable semantic search based on meaning rather than keywords"

### **Document Ingestion**
- "Built modular ingestion pipeline for PDFs, web pages, and YouTube transcripts"
- "All loaders extend BaseDocumentLoader interface for consistency"
- "YouTube loader extracts transcripts making video content searchable"

### **Conversation Memory**
- "Implemented session-based conversation memory for multi-turn dialogues"
- "Each user session maintains isolated chat history"
- "Integrated with LangChain's ChatMessageHistory for compatibility"

### **UI/UX**
- "Built interactive Streamlit interface with real-time chat"
- "Users can switch models, adjust temperature, and export conversations"
- "Source attribution displayed for transparency and trust"

### **Testing & Quality**
- "Comprehensive pytest suite covering all core modules"
- "RAG evaluator for measuring retrieval accuracy and answer quality"
- "Modular design enables easy unit testing"

---

## 🎤 Sample Interview Questions & Answers

### **Q: Explain RAG architecture**
**A**: "RAG combines retrieval and generation. When a user asks a question, we first retrieve relevant document chunks from a vector database using semantic similarity. These chunks provide context to the LLM, which generates a grounded answer. This reduces hallucinations since the LLM answers based on actual documents, not just its training data."

### **Q: Why use vector databases?**
**A**: "Vector databases enable semantic search. Traditional keyword search misses synonyms and context. With embeddings, 'What is AI?' and 'Explain artificial intelligence' are understood as similar queries. Vector databases like FAISS and ChromaDB efficiently find the most semantically similar documents to the query."

### **Q: How do agents work?**
**A**: "I used the ReAct pattern where the agent alternates between reasoning and acting. The LLM receives a question and available tools (Wikipedia, ArXiv, web search, PDF search). It thinks about which tool to use, takes an action, observes the result, and repeats until it can answer the question. This enables dynamic routing instead of hardcoded if-else logic."

### **Q: How did you handle different document types?**
**A**: "I created a BaseDocumentLoader interface and implemented specific loaders for PDFs (using PyPDF), web pages (BeautifulSoup), and YouTube (transcript API). Each loader returns standardized LangChain Document objects with consistent metadata, making them interchangeable in the indexing pipeline."

### **Q: What's your approach to conversation memory?**
**A**: "I implemented session-based memory where each user session has an isolated ChatMessageHistory. Messages are stored in-memory (can be extended to Redis/PostgreSQL). When a user asks a follow-up, the RAG chain receives the full conversation history, enabling context-aware responses."

### **Q: How do you ensure answer quality?**
**A**: "Three approaches: (1) Source grounding - LLM must answer from retrieved context, (2) Source attribution - responses include document references, (3) RAG evaluator - measures retrieval accuracy and answer relevance. Plus comprehensive pytest suite for code quality."

---

## 💡 Advanced Topics to Discuss

### **Prompt Engineering**
- Designed prompts that enforce source grounding
- ReAct prompt format for agent reasoning

### **Chunking Strategies**
- RecursiveCharacterTextSplitter respects document structure
- Chunk overlap preserves context across boundaries

### **Model Selection**
- Groq chosen for speed and free tier
- Factory pattern enables easy provider switching

### **Scalability Considerations**
- FAISS for <1M vectors
- ChromaDB for persistent storage
- Future: Pinecone for distributed scale

### **Error Handling**
- Graceful degradation when tools unavailable
- Parsing error handling in agents
- Validation with Pydantic schemas

---

## 📚 Key Files to Review Before Interview

1. **`requirements.txt`** - Full dependency list
2. **`app/core/llm.py`** - LLM abstraction and factory
3. **`app/core/embeddings.py`** - Embedding management
4. **`app/core/vector_store.py`** - Vector database abstraction
5. **`app/agents/agent_router.py`** - Agent orchestration
6. **`app/agents/tools.py`** - Tool definitions
7. **`app/rag/chain.py`** - RAG pipeline
8. **`app/core/memory.py`** - Conversation memory
9. **`app/ui/chat_app.py`** - Streamlit interface
10. **`README.md`** - Project overview

---

## 🚀 Deployment & Production Considerations

### **Current Setup**
- Local development with Streamlit
- FAISS/ChromaDB for vector storage
- Groq cloud API for LLM inference

### **Production Enhancements**
- **Vector Store**: Migrate to Pinecone or Weaviate for scale
- **Memory**: Add Redis/PostgreSQL for persistent conversations
- **Authentication**: Add user authentication
- **API Layer**: Wrap in FastAPI for programmatic access
- **Monitoring**: Add logging, metrics, tracing (LangSmith)
- **Caching**: Cache embeddings and LLM responses
- **Rate Limiting**: Prevent API abuse

---

## 📈 Performance Metrics

### **Latency**
- Embedding generation: ~100ms for 512 tokens
- Vector search: <50ms (FAISS), <200ms (Chroma)
- LLM inference: 500-2000ms depending on model
- End-to-end RAG: 1-3 seconds

### **Accuracy**
- Evaluated with RAG evaluator module
- Metrics: Retrieval precision, answer relevance, source grounding

---

## 🎓 Learning Outcomes

### **Technical Skills Demonstrated**
✅ LangChain framework mastery  
✅ Vector database implementation  
✅ Embedding generation and semantic search  
✅ Multi-agent AI systems  
✅ RAG architecture  
✅ LLM integration and prompt engineering  
✅ Document processing pipelines  
✅ Streamlit UI development  
✅ Design patterns (Factory, Template Method, Strategy)  
✅ Testing with pytest  
✅ Configuration management  
✅ API integration (Groq, Wikipedia, ArXiv, DuckDuckGo)  

---

## 📝 Summary

This GenAI Knowledge Assistant project demonstrates **production-grade RAG architecture** with **agentic AI workflows**, combining multiple modern technologies:

- **LangChain** for orchestration
- **Groq/LLaMA** for fast LLM inference
- **FAISS/ChromaDB** for vector storage
- **Hugging Face** for embeddings
- **Streamlit** for UI
- **Multi-agent architecture** for dynamic tool selection
- **Comprehensive testing** with pytest

The system showcases **clean architecture**, **design patterns**, **modularity**, and **scalability** - all critical for production AI systems.

---

**Good luck with your interview! 🚀**
