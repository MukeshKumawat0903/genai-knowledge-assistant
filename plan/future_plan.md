# Future Plan — GenAI Knowledge Assistant

This document tracks planned improvements, known gaps, and the phased roadmap for evolving the project beyond its current state.

---

## Current Gaps (What Needs Fixing First)

These are bugs or missing wires in the existing code — not new features.

| Gap | Location | Impact |
|-----|----------|--------|
| Agent mode does not connect the RAG retriever | `chat_app.py → call_knowledge_agent()` | PDF knowledge tool always returns "not initialized" in Agent mode |
| Conversation memory not wired into RAGChain | `chat_app.py → call_rag_chain()` | History is passed but `ConversationMemoryManager` is not persisted across re-renders |
| Model list is hardcoded in the UI | `chat_app.py` line ~469 | Breaking if Groq deprecates a model; should load from config |
| README model name mismatch | `.env.example` vs UI defaults | `.env.example` uses `llama3-70b-8192` (old ID); UI uses `llama-3.3-70b-versatile` |

---

## Phase 1 — Quick Wins (1–2 weeks)

High-impact, low-effort improvements that fix real usability issues.

### 1.1 Fix Agent ↔ PDF Retriever Connection
Connect the existing RAG retriever to the agent's PDF knowledge tool in `call_knowledge_agent()`.
The wiring is already designed (`create_knowledge_agent(retriever=retriever)`) but the retriever is never passed from the UI layer.

**Files:** `app/ui/chat_app.py`, `app/agents/agent_router.py`
**Effort:** Small — load the vector store and pass the retriever, same pattern as `call_rag_chain()`

### 1.2 Streaming Responses
Add token-by-token streaming to the Streamlit chat using `st.write_stream()` and LangChain's `stream()` API on ChatGroq.
Users currently wait silently with a spinner; streaming improves perceived speed significantly.

**Files:** `app/ui/chat_app.py → handle_user_input()`, `app/rag/chain.py`
**Effort:** Medium — LangChain's `stream()` returns a generator; Streamlit supports it natively

### 1.3 Load Model List from Config
Replace the hardcoded model list in `render_sidebar()` with a config-driven list so model additions/deprecations don't require code changes.

**Files:** `app/utils/config.py`, `app/ui/chat_app.py`
**Effort:** Small

### 1.4 Auto-index Status Badge
Show the current index state (document count, last indexed timestamp, vector store type) in the sidebar so users know whether the knowledge base is ready before switching to RAG mode.

**Files:** `app/ui/chat_app.py`, `app/rag/indexer.py`
**Effort:** Small

---

## Phase 2 — Retrieval Quality Improvements (2–4 weeks)

Better retrieval = better answers. These improvements make the RAG pipeline more accurate without changing the UI.

### 2.1 Cross-Encoder Reranking
After vector retrieval, apply a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score and reorder the top-k chunks before passing them to the LLM.
Reranking consistently improves answer quality for ambiguous queries.

**Files:** `app/rag/retriever.py` — add a `rerank()` step after `similarity_search()`
**Effort:** Medium — add `sentence-transformers` cross-encoder; no API cost

### 2.2 Multi-Query Retrieval
For each user query, generate 3–5 paraphrased variants using the LLM, retrieve documents for each variant, and merge/deduplicate results.
Catches relevant chunks that use different vocabulary than the original query.

**Files:** `app/rag/retriever.py` — new `MultiQueryRetriever` wrapper
**Effort:** Medium — LangChain has `MultiQueryRetriever` built-in

### 2.3 HyDE (Hypothetical Document Embeddings)
Before retrieval, ask the LLM to generate a hypothetical answer to the query. Embed that hypothetical answer and use it as the search vector instead of the raw query.
Improves retrieval for questions where the phrasing differs from how answers are written in documents.

**Files:** `app/rag/retriever.py`
**Effort:** Medium

### 2.4 Contextual Chunking
Replace fixed-size chunking with semantic/contextual chunking that respects paragraph and section boundaries.
Reduces mid-sentence splits that cause incoherent retrieved chunks.

**Files:** `app/core/text_splitter.py`
**Effort:** Medium — use `langchain_experimental.text_splitter.SemanticChunker`

### 2.5 Metadata Filtering
Allow users to filter retrieval by source type (PDF only, web only), filename, or date range.
Useful when the knowledge base contains documents from many different sources.

**Files:** `app/rag/retriever.py`, `app/ui/chat_app.py`
**Effort:** Medium

---

## Phase 3 — Multi-Provider LLM Support (2–3 weeks)

The `LLMFactory` / `BaseLLM` abstraction is already designed for this. These are the stubbed providers waiting to be implemented.

### 3.1 OpenAI / Azure OpenAI
Add `OpenAILLM` class using `langchain_openai.ChatOpenAI`.
Enables GPT-4o, GPT-4-turbo, and o1 models.

**Files:** `app/core/llm.py` — uncomment and implement `OpenAILLM`
**Config:** Add `OPENAI_API_KEY` to `.env`

### 3.2 Anthropic (Claude)
Add `AnthropicLLM` class using `langchain_anthropic.ChatAnthropic`.
Enables Claude Sonnet 4.6, Claude Opus 4.8 — best-in-class for long document analysis.

**Files:** `app/core/llm.py` — implement `AnthropicLLM`
**Config:** Add `ANTHROPIC_API_KEY` to `.env`

### 3.3 Google Gemini
Add `GoogleLLM` class using `langchain_google_genai.ChatGoogleGenerativeAI`.
Enables Gemini 1.5 Pro/Flash with 1M token context window — excellent for large document ingestion.

**Files:** `app/core/llm.py` — implement `GoogleLLM`
**Config:** Add `GOOGLE_API_KEY` to `.env`

### 3.4 Provider Selection in UI
Extend the Model Controls sidebar to show a provider dropdown (Groq / OpenAI / Anthropic / Google) and update the model list dynamically based on provider selection.

**Files:** `app/ui/chat_app.py`, `app/utils/config.py`

---

## Phase 4 — Persistent Sessions & User Management (3–4 weeks)

Currently all state is in-memory and resets on page reload. This phase adds durability.

### 4.1 Persistent Chat History (Redis or SQLite)
Store conversation history in Redis (production) or SQLite (local) so sessions survive page reloads.
Wire `ConversationMemoryManager` with a persistent backend instead of in-memory `ChatMessageHistory`.

**Files:** `app/core/memory.py` — add `RedisChatMessageHistory` or `SQLiteChatMessageHistory` backend
**Dependencies:** `redis` or `sqlite3` (stdlib)

### 4.2 Session Restore on Reload
On page load, check for an existing session cookie and restore the conversation history from persistent storage.

**Files:** `app/ui/chat_app.py → initialize_session_state()`

### 4.3 Document Management Panel
Add a sidebar panel listing all indexed documents with:
- Source name, type (PDF/web/YouTube), and index timestamp
- Document count and total chunk count
- Delete individual documents or clear the entire index

**Files:** `app/ui/chat_app.py`, `app/rag/indexer.py` — add `list_documents()` and `delete_document()` methods

### 4.4 Chat Import
Implement the "Import Chat History" feature already marked as "Coming Soon" in the UI.
Allow re-loading a previously exported JSON file to restore a conversation.

**Files:** `app/ui/chat_app.py → render_sidebar()`

---

## Phase 5 — Advanced Evaluation & Observability (2–3 weeks)

Make the system measurable and debuggable.

### 5.1 LLM-as-Judge Evaluation (RAGAS)
Integrate [RAGAS](https://github.com/explodinggradients/ragas) for automated evaluation without manual relevance labels:
- **Faithfulness** — Is the answer supported by the context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision/Recall** — How good is the retrieval?

**Files:** `app/rag/evaluator.py` — add `RAGASEvaluator` class
**Dependencies:** `ragas`

### 5.2 Retrieval Debug Mode
Add an expandable debug panel in the UI showing:
- The chunks actually retrieved (with similarity scores)
- Which tool the agent chose and why (agent reasoning trace)
- Token usage estimate

**Files:** `app/ui/chat_app.py`, `app/rag/retriever.py`

### 5.3 Query Analytics Dashboard
Track and visualize per-session metrics:
- Response latency per query
- Tool usage frequency (which agent tools are called most)
- Source type distribution
- User satisfaction signals (thumbs up/down per response)

**Files:** New `app/ui/analytics_app.py` — Streamlit second page

### 5.4 Feedback Loop (Thumbs Up/Down)
Add a thumbs-up / thumbs-down button on each assistant response.
Store feedback with the query, retrieved context, and response for future evaluation and fine-tuning datasets.

**Files:** `app/ui/chat_app.py → render_chat_message()`

---

## Phase 6 — Additional Data Sources (1–2 weeks each)

### 6.1 Notion Integration
Ingest Notion pages and databases via the Notion API.
Enables personal knowledge management workflows.

**Files:** New `app/ingestion/notion_loader.py`
**Dependencies:** `notion-client`

### 6.2 Google Drive / OneDrive
Ingest PDFs and docs from cloud storage.
**Files:** New `app/ingestion/gdrive_loader.py`

### 6.3 GitHub Repository Ingestion
Index code repositories (Python, JS, etc.) for code-aware Q&A.
**Files:** New `app/ingestion/github_loader.py`
**Dependencies:** `GitPython` or LangChain's `GitLoader`

### 6.4 Confluence / Jira
Enterprise knowledge base ingestion for workplace deployments.

---

## Phase 7 — Production Readiness (4–6 weeks)

### 7.1 FastAPI Backend
Separate the backend (RAG chain, agent) from the Streamlit UI into a FastAPI REST API.
Enables integration with other frontends (React, mobile apps).

```
POST /api/chat          — Query the assistant
POST /api/index         — Index new documents
GET  /api/documents     — List indexed documents
DELETE /api/documents/{id}  — Remove a document
GET  /api/sessions/{id} — Get session history
```

**New:** `api/main.py`, `api/routes/`

### 7.2 Docker Containerization
Add `Dockerfile` and `docker-compose.yml` to run the full stack (app + Chroma + Redis) with a single command.

**New:** `Dockerfile`, `docker-compose.yml`

### 7.3 Pinecone / Weaviate / Qdrant Support
Add cloud-hosted vector store options for production scale (millions of documents).

**Files:** `app/core/vector_store.py` — extend `VectorStoreManager` with additional backends

### 7.4 Rate Limiting & Cost Tracking
Track API token usage per session and display estimated cost in the sidebar.
Add configurable rate limits to prevent runaway API spend.

**Files:** `app/core/llm.py`, `app/ui/chat_app.py`

### 7.5 CI/CD Pipeline
Add GitHub Actions workflow for:
- Automated test runs on PRs (`pytest`)
- Linting (`ruff`, `black`)
- Docker image build and push

**New:** `.github/workflows/ci.yml`

---

## Priority Matrix

| Improvement | Impact | Effort | Priority |
|-------------|--------|--------|----------|
| Fix Agent ↔ PDF retriever wiring | High | Low | P0 |
| Streaming responses | High | Medium | P0 |
| Cross-encoder reranking | High | Medium | P1 |
| Multi-query retrieval | High | Low | P1 |
| Persistent chat history | Medium | Medium | P1 |
| Anthropic / OpenAI provider | Medium | Low | P1 |
| Document management panel | Medium | Medium | P2 |
| RAGAS automated evaluation | High | Medium | P2 |
| Feedback loop (thumbs up/down) | Medium | Low | P2 |
| FastAPI backend | High | High | P3 |
| Docker containerization | Medium | Medium | P3 |
| Multi-modal support (images) | Medium | High | P3 |
| Pinecone / Qdrant support | Low | Medium | P3 |

---

## Technical Debt

- The `evaluator.py` docstring block contains ASCII diagram content mixed with Python code — needs cleanup
- `chat_app.py` is a single large file (~1045 lines); split into `sidebar.py`, `chat_panel.py`, `export.py`
- Agent `run()` method has TODO items: query preprocessing, response streaming, citation formatting
- `ConversationMemoryManager` TODO items: Redis persistence, memory windowing, TTL expiration
- Tests cover unit behavior but no integration test exercises the full RAG pipeline end-to-end
- `.env.example` model name (`llama3-70b-8192`) is a deprecated Groq model ID — update to `llama-3.3-70b-versatile`

---

*Last updated: 2026-06-01*
