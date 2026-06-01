"""
Streamlit Chat Interface for GenAI Knowledge Assistant

A clean, modular UI that provides a conversational chat experience
with backend RAG/Agent integration.

Key Features:
- Chat message display (user and assistant)
- Session-based chat history
- Backend integration (RAG chain or agent)
- Source display for retrieved context
- Clear/Reset functionality

Architecture:
┌──────────────────────┐
│   Streamlit UI       │
│   (chat_app.py)      │
└──────────┬───────────┘
           │
           ├─────────────────┐
           ▼                 ▼
    ┌──────────┐      ┌──────────┐
    │ RAGChain │  OR  │  Agent   │
    └──────────┘      └──────────┘
           │                 │
           └────────┬────────┘
                    ▼
              ┌──────────┐
              │ Response │
              └──────────┘

Usage:
    $ streamlit run app/ui/chat_app.py
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# =============================================================================
# Configuration
# =============================================================================

# App configuration
APP_TITLE = "🤖 GenAI Knowledge Assistant"
APP_DESCRIPTION = """
Ask questions and get intelligent answers powered by RAG and AI agents.
The assistant can search PDFs, web, Wikipedia, and ArXiv.
"""

# Session state keys
SESSION_ID_KEY = "session_id"
MESSAGES_KEY = "messages"
BACKEND_MODE_KEY = "backend_mode"  # "rag" or "agent"

# Ingestion/indexing session keys
INDEX_STATUS_KEY = "index_status"  # Stores last indexing result/status


# =============================================================================
# Session State Management
# =============================================================================

def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    
    This function sets up:
    - Unique session ID for tracking conversations
    - Empty message history list
    - Backend mode selection
    
    Called once when the app loads.
    """
    # Generate unique session ID if not exists
    if SESSION_ID_KEY not in st.session_state:
        st.session_state[SESSION_ID_KEY] = str(uuid.uuid4())
        print(f"[INFO] New session created: {st.session_state[SESSION_ID_KEY]}")
    
    # Initialize message history
    if MESSAGES_KEY not in st.session_state:
        st.session_state[MESSAGES_KEY] = []
        print(f"[INFO] Message history initialized")
    
    # Set default backend mode
    if BACKEND_MODE_KEY not in st.session_state:
        st.session_state[BACKEND_MODE_KEY] = "agent"  # Default to agent mode


def reset_chat():
    """
    Clear chat history and reset session.
    
    This function:
    - Clears all messages
    - Generates new session ID
    - Resets backend state
    
    Called when user clicks "Clear Chat" button.
    """
    st.session_state[MESSAGES_KEY] = []
    st.session_state[SESSION_ID_KEY] = str(uuid.uuid4())
    print(f"[INFO] Chat reset. New session: {st.session_state[SESSION_ID_KEY]}")
    st.rerun()  # Force UI refresh


def get_session_id() -> str:
    """Get current session ID."""
    return st.session_state.get(SESSION_ID_KEY, "unknown")


def get_chat_history() -> List[Dict[str, str]]:
    """
    Get current chat history.
    
    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    return st.session_state.get(MESSAGES_KEY, [])


def add_message(role: str, content: str, sources: Optional[List[Dict]] = None):
    """
    Add a message to chat history.
    
    Args:
        role: "user" or "assistant"
        content: Message text
        sources: Optional list of source documents (for assistant messages)
    """
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add sources if provided (for assistant responses)
    if sources:
        message["sources"] = sources
    
    st.session_state[MESSAGES_KEY].append(message)


# =============================================================================
# Backend Integration Layer
# =============================================================================

def call_rag_chain(query: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Call RAG chain backend for document-based Q&A.
    
    This function integrates with the RAG chain to:
    1. Retrieve relevant documents from vector store
    2. Generate answers using LLM with retrieved context
    3. Return answer with source citations
    
    Args:
        query: User's question
        chat_history: Previous conversation turns
    
    Returns:
        Dictionary with:
            - answer: Generated response text
            - sources: List of source documents
            - metadata: Additional info
    """
    try:
        from app.rag.chain import create_rag_chain
        from app.rag.retriever import RAGRetriever
        from app.core.vector_store import VectorStoreManager
        from app.core.embeddings import EmbeddingManager
        from app.utils.config import get_settings
        from app.core.llm import LLMFactory

        settings = get_settings()

        # Check for runtime overrides from UI controls
        runtime_model = st.session_state.get("runtime_model_override")
        runtime_temperature = st.session_state.get("runtime_temperature_override")
        runtime_provider = st.session_state.get("runtime_provider_override", settings.llm_provider)

        # Create LLM with runtime overrides when the user has applied settings
        llm = None
        if runtime_model is not None and runtime_temperature is not None:
            llm = LLMFactory.create_from(
                provider=runtime_provider,
                model=runtime_model,
                temperature=runtime_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        # If no overrides, LLM will be auto-created by create_rag_chain() with .env defaults
        
        # Check for runtime vector store override and use type-specific subdirectory
        # This ensures we load from the correct isolated directory (faiss/ or chroma/)
        runtime_vector_store = st.session_state.get("runtime_vector_store_override")
        if runtime_vector_store:
            # User has changed vector store type - use subdirectory
            original_path = settings.vector_store_path
            settings.vector_store_path = original_path / runtime_vector_store
            settings.vector_store_type = runtime_vector_store
        else:
            # Use default from .env with type-specific subdirectory for isolation
            original_path = settings.vector_store_path
            settings.vector_store_path = original_path / settings.vector_store_type
        
        # Initialize embeddings
        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.get_embeddings()
        
        # Initialize vector store manager and load existing index
        vector_manager = VectorStoreManager(settings, embeddings)
        
        # Try to load existing vector store
        try:
            vector_store = vector_manager.load_vector_store()
        except Exception as load_error:
            return {
                "answer": (
                    "⚠️ **No document index found.**\n\n"
                    "Please index content first using the sidebar controls (PDF upload / URLs / YouTube).\n"
                    "Then switch to **RAG** mode and ask your question again.\n\n"
                    f"Technical details: {str(load_error)}"
                ),
                "sources": [],
                "metadata": {"mode": "rag", "error": "no_index"}
            }
        
        # Create retriever
        retriever = RAGRetriever(
            vector_store=vector_store,
            settings=settings
        )
        
        # Create RAG chain (with custom LLM if runtime overrides applied)
        chain = create_rag_chain(retriever=retriever, llm=llm)
        
        # Convert chat history format if needed
        history_tuples = None
        if chat_history:
            # Extract alternating user/assistant pairs
            history_tuples = []
            for i in range(0, len(chat_history) - 1, 2):
                if chat_history[i]["role"] == "user" and i + 1 < len(chat_history):
                    user_msg = chat_history[i]["content"]
                    assistant_msg = chat_history[i + 1]["content"]
                    history_tuples.append((user_msg, assistant_msg))
        
        # Run the chain
        result = chain.run(query, chat_history=history_tuples)
        
        # Format sources for display
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "answer": result["answer"],
            "sources": sources,
            "metadata": {"mode": "rag", **result.get("metadata", {})}
        }
        
    except ImportError as e:
        return {
            "answer": f"⚠️ **Module import error:** {str(e)}\n\nPlease ensure all dependencies are installed.",
            "sources": [],
            "metadata": {"mode": "rag", "error": "import_error"}
        }
    except Exception as e:
        return {
            "answer": f"❌ **Error processing request:** {str(e)}\n\nPlease try again or rephrase your question.",
            "sources": [],
            "metadata": {"mode": "rag", "error": str(type(e).__name__)}
        }


def call_knowledge_agent(query: str) -> Dict[str, Any]:
    """
    Call knowledge agent backend for multi-tool Q&A.
    
    This function integrates with the KnowledgeAgent to:
    1. Reason about which tools to use (PDF search, web, Wikipedia, ArXiv)
    2. Execute tools based on reasoning
    3. Synthesize information from multiple sources
    4. Return comprehensive answer
    
    Args:
        query: User's question
    
    Returns:
        Dictionary with:
            - answer: Generated response text
            - sources: List of source documents (if available)
            - metadata: Tool usage info
    """
    try:
        from app.agents.agent_router import create_knowledge_agent
        from app.utils.config import get_settings
        from app.core.llm import LLMFactory

        settings = get_settings()

        # Check for runtime overrides from UI controls
        runtime_model = st.session_state.get("runtime_model_override")
        runtime_temperature = st.session_state.get("runtime_temperature_override")
        runtime_provider = st.session_state.get("runtime_provider_override", settings.llm_provider)

        # Create LLM with runtime overrides when the user has applied settings
        llm = None
        if runtime_model is not None and runtime_temperature is not None:
            llm = LLMFactory.create_from(
                provider=runtime_provider,
                model=runtime_model,
                temperature=runtime_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        # If no overrides, agent will be created with .env defaults
        
        # Create agent with verbose mode for debugging
        # Note: Set verbose=False in production for cleaner output
        agent = create_knowledge_agent(llm=llm, verbose=False)
        
        # Run the agent
        response = agent.run(query)
        
        return {
            "answer": response,
            "sources": [],  # Agent synthesizes from multiple tools
            "metadata": {
                "mode": "agent",
                "tools_available": agent.get_available_tools()
            }
        }
        
    except ImportError as e:
        return {
            "answer": (
                f"⚠️ **Module import error:** {str(e)}\n\n"
                "Please ensure all dependencies are installed:\n"
                "```\npip install -r requirements.txt\n```"
            ),
            "sources": [],
            "metadata": {"mode": "agent", "error": "import_error"}
        }
    except ValueError as e:
        return {
            "answer": (
                f"⚠️ **Configuration error:** {str(e)}\n\n"
                "Please check your `.env` file and ensure API keys are set."
            ),
            "sources": [],
            "metadata": {"mode": "agent", "error": "config_error"}
        }
    except Exception as e:
        error_type = type(e).__name__
        return {
            "answer": (
                f"❌ **Agent error ({error_type}):** {str(e)}\n\n"
                "The agent encountered an issue. Please try:\n"
                "- Rephrasing your question\n"
                "- Checking your internet connection\n"
                "- Verifying API keys in .env file"
            ),
            "sources": [],
            "metadata": {"mode": "agent", "error": error_type}
        }


def get_assistant_response(query: str) -> Dict[str, Any]:
    """
    Get response from backend (RAG chain or agent).
    
    This function routes the query to the appropriate backend
    based on the current mode setting.
    
    Args:
        query: User's question
    
    Returns:
        Response dictionary with answer, sources, and metadata
    """
    # Get backend mode from session state
    mode = st.session_state.get(BACKEND_MODE_KEY, "agent")
    
    # Route to appropriate backend
    if mode == "rag":
        chat_history = get_chat_history()
        return call_rag_chain(query, chat_history)
    else:  # mode == "agent"
        return call_knowledge_agent(query)


# =============================================================================
# UI Rendering Functions
# =============================================================================

def render_header():
    """
    Render app header with title and description.
    """
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
    st.divider()


def render_sidebar():
    """
    Render sidebar with controls and settings.
    
    TODO:
    - Add model selection dropdown (GPT-4, Claude, etc.)
    - Add vector DB selection (Chroma, Pinecone, etc.)
    - Add temperature slider for LLM
    - Add max tokens control
    - Add file upload for PDFs
    - Add settings persistence
    """
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Backend mode selector
        st.session_state[BACKEND_MODE_KEY] = st.radio(
            "Backend Mode",
            options=["agent", "rag"],
            index=0 if st.session_state.get(BACKEND_MODE_KEY, "agent") == "agent" else 1,
            help="Agent: Uses tools dynamically. RAG: Direct retrieval + generation."
        )
        
        st.divider()
        
        # Runtime Model Controls - Collapsed by default for cleaner UI
        with st.expander("🎛️ Model Controls", expanded=False):
            from app.utils.config import get_settings
            settings = get_settings()

            # ---- initialize session state defaults ----
            if "selected_provider" not in st.session_state:
                st.session_state["selected_provider"] = settings.llm_provider
            if "selected_model" not in st.session_state:
                st.session_state["selected_model"] = settings.llm_model_name
            if "temperature" not in st.session_state:
                st.session_state["temperature"] = settings.llm_temperature
            if "vector_store_type" not in st.session_state:
                st.session_state["vector_store_type"] = settings.vector_store_type

            # ---- Provider Selection ----
            provider_options = list(settings.available_models.keys())
            current_provider = st.session_state["selected_provider"]
            provider_index = provider_options.index(current_provider) if current_provider in provider_options else 0

            selected_provider = st.selectbox(
                "LLM Provider",
                options=provider_options,
                index=provider_index,
                format_func=lambda p: {
                    "groq": "Groq (free, fast)",
                    "openai": "OpenAI (GPT-4o)",
                    "anthropic": "Anthropic (Claude)",
                    "google": "Google (Gemini)",
                }.get(p, p.title()),
                help="Choose the AI provider. Each requires its own API key in .env.",
            )
            # Reset model selection when provider changes
            if selected_provider != st.session_state["selected_provider"]:
                st.session_state["selected_provider"] = selected_provider
                st.session_state["selected_model"] = settings.available_models[selected_provider][0]

            # ---- Model Selection (dynamic per provider) ----
            model_options = settings.available_models.get(selected_provider, [settings.llm_model_name])
            current_model = st.session_state["selected_model"]
            model_index = model_options.index(current_model) if current_model in model_options else 0

            st.session_state["selected_model"] = st.selectbox(
                "LLM Model",
                options=model_options,
                index=model_index,
                help="Models available for the selected provider.",
            )

            # ---- Temperature ----
            st.session_state["temperature"] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state["temperature"],
                step=0.05,
                help="0.0–0.3 = factual/focused. 0.7–1.0 = creative/diverse.",
            )

            # ---- Vector Store ----
            vector_store_options = ["faiss", "chroma"]
            current_vector_store = st.session_state["vector_store_type"]
            vs_index = vector_store_options.index(current_vector_store) if current_vector_store in vector_store_options else 0

            st.session_state["vector_store_type"] = st.selectbox(
                "Vector Store",
                options=vector_store_options,
                index=vs_index,
                help="Choose the vector database. Switching requires re-indexing.",
            )
            st.info("ℹ️ Switching vector stores requires re-indexing documents.")

            # ---- Status indicators ----
            if not st.session_state.get("settings_applied", False):
                st.warning("⚠️ Click 'Apply Settings' to activate your selection.")
            elif st.session_state.get("runtime_model_override") != st.session_state["selected_model"]:
                st.warning("⚠️ Model changed. Click 'Apply Settings' to use it.")

            # ---- Apply Settings Button ----
            if st.button("✅ Apply Settings", use_container_width=True, type="primary"):
                vector_store_changed = (
                    st.session_state.get("runtime_vector_store_override", settings.vector_store_type)
                    != st.session_state["vector_store_type"]
                )

                st.session_state["runtime_provider_override"] = st.session_state["selected_provider"]
                st.session_state["runtime_model_override"] = st.session_state["selected_model"]
                st.session_state["runtime_temperature_override"] = st.session_state["temperature"]
                st.session_state["runtime_vector_store_override"] = st.session_state["vector_store_type"]

                for key in ("cached_llm", "cached_vector_store", "cached_retriever"):
                    st.session_state.pop(key, None)

                st.session_state["settings_applied"] = True
                st.success(
                    f"✅ Applied: {st.session_state['selected_provider'].title()} / "
                    f"{st.session_state['selected_model']}"
                )
                if vector_store_changed:
                    st.warning(
                        "⚠️ Vector store changed. Re-index your documents with 'Load + Index'."
                    )
        
        st.divider()

        # Content ingestion + indexing (for RAG)
        st.subheader("📥 Index Content")
        st.caption("Upload PDFs or paste URLs to build the RAG knowledge base.")

        # PDF uploader
        uploaded_pdfs = st.file_uploader(
            "📄 Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Uploaded PDFs are saved under data/raw and auto-indexed into the vector store."
        )

        # Website URLs (one per line)
        website_urls_text = st.text_area(
            "🌐 Website URL(s) (one per line)",
            value="",
            height=90,
            help="Example: https://example.com/article"
        )

        # YouTube URLs (one per line)
        youtube_urls_text = st.text_area(
            "▶️ YouTube URL(s) (one per line)",
            value="",
            height=90,
            help="Example: https://www.youtube.com/watch?v=..."
        )

        def _parse_lines(text: str) -> List[str]:
            return [line.strip() for line in (text or "").splitlines() if line.strip()]

        website_urls = _parse_lines(website_urls_text)
        youtube_urls = _parse_lines(youtube_urls_text)

        # Auto-index toggle
        auto_index_enabled = st.checkbox(
            "⚡ Auto-index on upload",
            value=True,
            help="Automatically index PDFs when uploaded. Disable if you want to add URLs/YouTube first."
        )
        
        # Detect if PDFs were just uploaded (compare with previous state)
        prev_pdf_count = st.session_state.get("_prev_pdf_count", 0)
        current_pdf_count = len(uploaded_pdfs) if uploaded_pdfs else 0
        pdfs_just_uploaded = auto_index_enabled and current_pdf_count > prev_pdf_count
        st.session_state["_prev_pdf_count"] = current_pdf_count
        
        # Manual index button (or triggered by auto-index)
        index_clicked = st.button("🔎 Load + Index", use_container_width=True) or pdfs_just_uploaded
        
        # Show auto-index indicator
        if pdfs_just_uploaded:
            st.info("⚡ Auto-indexing uploaded PDFs...")

        if index_clicked:
            try:
                from app.utils.config import get_settings
                from app.rag.indexer import create_indexer
                from app.ingestion.pdf_loader import PDFDocumentLoader
                from app.ingestion.web_loader import WebDocumentLoader
                from app.ingestion.youtube_loader import YouTubeDocumentLoader
                from langchain_core.documents import Document

                settings = get_settings()
                
                # ========================================================================
                # Check for runtime vector store override from UI controls
                # ========================================================================
                # Use separate directories for each vector store type to avoid data loss:
                # - data/vector_store/faiss/    (FAISS indices)
                # - data/vector_store/chroma/   (Chroma collections)
                #
                # Why this matters:
                # 1. FAISS and Chroma have incompatible storage formats
                # 2. Switching stores without separate dirs would overwrite existing data
                # 3. Users can experiment with different stores without losing work
                # 4. Enables easy rollback if new store has issues
                #
                # Example: User indexes 1000 PDFs in FAISS, then tries Chroma.
                # Without isolation, FAISS index would be lost. With isolation,
                # both indices persist and user can switch back anytime.
                runtime_vector_store = st.session_state.get("runtime_vector_store_override")
                if runtime_vector_store:
                    active_vector_store_type = runtime_vector_store
                else:
                    active_vector_store_type = settings.vector_store_type
                
                # Update vector store path to include type-specific subdirectory
                original_path = settings.vector_store_path
                settings.vector_store_path = original_path / active_vector_store_type

                documents: List[Document] = []

                # Save uploaded PDFs and load them as Documents
                if uploaded_pdfs:
                    uploads_dir = settings.raw_data_dir / "uploads" / get_session_id()
                    uploads_dir.mkdir(parents=True, exist_ok=True)

                    for uploaded in uploaded_pdfs:
                        # Streamlit provides an in-memory file-like object
                        filename = Path(uploaded.name).name
                        dest_path = uploads_dir / filename
                        with open(dest_path, "wb") as f:
                            f.write(uploaded.getbuffer())

                        pdf_loader = PDFDocumentLoader(path=str(dest_path), source_name=filename)
                        documents.extend(pdf_loader.load_documents())

                # Load website content
                if website_urls:
                    web_loader = WebDocumentLoader(urls=website_urls, source_name="web")
                    documents.extend(web_loader.load_documents())

                # Load YouTube transcripts
                if youtube_urls:
                    for url in youtube_urls:
                        yt_loader = YouTubeDocumentLoader(video_url=url, source_name="youtube")
                        documents.extend(yt_loader.load_documents())

                if not documents:
                    st.warning("Nothing to index. Upload PDFs and/or provide URLs.")
                else:
                    with st.spinner(f"Indexing {len(documents)} document(s) into {active_vector_store_type.upper()}…"):
                        # Create indexer with vector store type override
                        # This ensures documents are indexed into the correct store
                        indexer = create_indexer(vector_store_type=active_vector_store_type)
                        result = indexer.index_documents(documents)

                    st.session_state[INDEX_STATUS_KEY] = {
                        "timestamp": datetime.now().isoformat(),
                        **result,
                    }

                    if result.get("success"):
                        st.success(result.get("message", "Indexing complete."))
                    else:
                        st.error(result.get("message", "Indexing failed."))

            except Exception as e:
                st.session_state[INDEX_STATUS_KEY] = {
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "message": f"Indexing failed: {str(e)}",
                }
                st.error(st.session_state[INDEX_STATUS_KEY]["message"])

        # Show last indexing status (if any)
        last_status = st.session_state.get(INDEX_STATUS_KEY)
        if last_status:
            ok = bool(last_status.get("success"))
            label = "Last index: success" if ok else "Last index: failed"
            st.caption(f"{label} • {last_status.get('timestamp', '')}")
        
        # Session info
        st.subheader("📊 Session Info")
        st.text(f"Session ID: {get_session_id()[:8]}...")
        st.text(f"Messages: {len(get_chat_history())}")
        
        st.divider()
        
        # ========================================================================
        # Chat Export Section
        # ========================================================================
        st.subheader("💾 Export Chat")
        
        # Get current chat history
        chat_messages = get_chat_history()
        
        if len(chat_messages) == 0:
            st.caption("No messages to export yet.")
        else:
            # Format selection
            export_format = st.radio(
                "Export Format",
                options=["JSON", "Plain Text"],
                horizontal=True,
                help="Choose the format for exporting your conversation"
            )
            
            # Prepare export data based on selected format
            if export_format == "JSON":
                # ================================================================
                # JSON Export Format
                # ================================================================
                # Structured export including:
                # - role: "user" or "assistant"
                # - content: message text
                # - timestamp: ISO format timestamp (if available)
                # - sources: document sources for assistant messages (if available)
                #
                # This format is machine-readable and preserves all metadata
                # for potential re-import or analysis.
                import json
                
                export_data = {
                    "session_id": get_session_id(),
                    "export_timestamp": datetime.now().isoformat(),
                    "message_count": len(chat_messages),
                    "messages": []
                }
                
                for msg in chat_messages:
                    message_entry = {
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                        "timestamp": msg.get("timestamp", "")
                    }
                    # Include sources if present (for assistant messages with RAG)
                    if "sources" in msg and msg["sources"]:
                        message_entry["sources"] = msg["sources"]
                    
                    export_data["messages"].append(message_entry)
                
                # Convert to pretty-printed JSON string
                file_content = json.dumps(export_data, indent=2, ensure_ascii=False)
                file_name = f"chat_export_{get_session_id()[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                mime_type = "application/json"
            
            else:  # Plain Text
                # ================================================================
                # Plain Text Export Format
                # ================================================================
                # Human-readable conversation format with:
                # - Clear role labels (User: / Assistant:)
                # - Timestamps in readable format
                # - Separator lines for readability
                # - Source information for RAG responses
                #
                # This format is ideal for:
                # - Sharing conversations with others
                # - Copying into documents/reports
                # - Quick review without technical tools
                lines = []
                lines.append("=" * 80)
                lines.append("GenAI Knowledge Assistant - Chat Export")
                lines.append("=" * 80)
                lines.append(f"Session ID: {get_session_id()}")
                lines.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append(f"Total Messages: {len(chat_messages)}")
                lines.append("=" * 80)
                lines.append("")
                
                for idx, msg in enumerate(chat_messages, 1):
                    role = msg.get("role", "unknown").upper()
                    content = msg.get("content", "")
                    timestamp = msg.get("timestamp", "")
                    
                    # Format timestamp if available
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            time_str = timestamp
                    else:
                        time_str = "N/A"
                    
                    lines.append(f"[{idx}] {role} ({time_str})")
                    lines.append("-" * 80)
                    lines.append(content)
                    
                    # Include source information for assistant messages
                    if role == "ASSISTANT" and "sources" in msg and msg["sources"]:
                        lines.append("")
                        lines.append(f"📚 Sources ({len(msg['sources'])}):")
                        for src_idx, source in enumerate(msg["sources"], 1):
                            metadata = source.get("metadata", {})
                            source_name = metadata.get("source", "Unknown")
                            lines.append(f"   {src_idx}. {source_name}")
                    
                    lines.append("")
                    lines.append("=" * 80)
                    lines.append("")
                
                file_content = "\n".join(lines)
                file_name = f"chat_export_{get_session_id()[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                mime_type = "text/plain"
            
            # ====================================================================
            # Download Button
            # ====================================================================
            # Provides in-browser download without server-side persistence.
            # The file is generated on-the-fly and sent directly to the user.
            # No data is stored on the server - fully client-side operation.
            st.download_button(
                label=f"📥 Download {export_format}",
                data=file_content,
                file_name=file_name,
                mime=mime_type,
                use_container_width=True
            )
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            reset_chat()
        
        st.divider()
        
        # ====================================================================
        # Document Management Panel
        # ====================================================================
        with st.expander("📂 Document Management", expanded=False):
            st.caption("Documents currently in the knowledge base index.")
            try:
                from app.rag.indexer import create_indexer
                from app.utils.config import get_settings as _gs

                _settings = _gs()
                _runtime_vs = st.session_state.get("runtime_vector_store_override")
                _active_vs = _runtime_vs or _settings.vector_store_type
                # Apply type-specific subdirectory (same pattern as call_rag_chain)
                _settings.vector_store_path = (
                    _settings.vector_store_path
                    if str(_settings.vector_store_path).endswith(_active_vs)
                    else _settings.vector_store_path / _active_vs
                )

                _mgr_indexer = create_indexer(vector_store_type=_active_vs)
                _docs = _mgr_indexer.list_documents()

                if not _docs:
                    st.info("No documents indexed yet. Use 'Index Content' above to add some.")
                else:
                    st.caption(f"{len(_docs)} document(s) indexed in {_active_vs.upper()}:")
                    for _doc in _docs:
                        _col1, _col2 = st.columns([4, 1])
                        with _col1:
                            _type_icon = {"pdf": "📄", "web": "🌐", "youtube": "▶️"}.get(
                                _doc.get("source_type", ""), "📁"
                            )
                            _src = _doc.get("source", "Unknown")
                            _chunks = _doc.get("num_chunks", "?")
                            _ts = _doc.get("indexed_at", "")[:10]
                            _has_sep = ("/" in _src) or ("\\" in _src)
                            _display = _src[-40:] if _has_sep else Path(_src).name
                            st.markdown(
                                f"{_type_icon} **{_display}**  \n"
                                f"<small>{_chunks} chunks · {_ts}</small>",
                                unsafe_allow_html=True,
                            )
                        with _col2:
                            if st.button("🗑️", key=f"del_{_src}", help=f"Remove {_src}"):
                                _del_result = _mgr_indexer.delete_document(_src)
                                if _del_result["success"]:
                                    st.success(_del_result["message"])
                                else:
                                    st.error(_del_result["message"])
                                st.rerun()

                    if st.button("🗑️ Clear Entire Index", use_container_width=True):
                        for _d in list(_docs):
                            _mgr_indexer.delete_document(_d["source"])
                        st.success("Index cleared.")
                        st.rerun()

            except Exception as _e:
                st.warning(f"Could not load document registry: {_e}")

        # ====================================================================
        # RAGAS Automated Evaluation
        # ====================================================================
        with st.expander("🧪 RAGAS Evaluation", expanded=False):
            st.caption(
                "LLM-as-judge evaluation of RAG quality. "
                "Scores the last assistant response on faithfulness, "
                "answer relevancy, and context precision."
            )

            _last_q = ""
            _last_a = ""
            _last_ctx: List[str] = []

            # Pre-fill from last exchange in history
            _history = get_chat_history()
            for _i in range(len(_history) - 1, -1, -1):
                if _history[_i]["role"] == "assistant":
                    _last_a = _history[_i].get("content", "")
                    _raw_sources = _history[_i].get("sources", [])
                    _last_ctx = [s.get("content", "") for s in _raw_sources if s.get("content")]
                    # find the preceding user message
                    if _i > 0 and _history[_i - 1]["role"] == "user":
                        _last_q = _history[_i - 1].get("content", "")
                    break

            _eval_question = st.text_input(
                "Question",
                value=_last_q,
                help="The user question to evaluate against.",
                key="ragas_question",
            )
            _eval_answer = st.text_area(
                "Answer (assistant response)",
                value=_last_a,
                height=80,
                help="The answer to evaluate for faithfulness and relevancy.",
                key="ragas_answer",
            )
            _eval_context = st.text_area(
                "Context (retrieved chunks, one per line)",
                value="\n\n---\n\n".join(_last_ctx) if _last_ctx else "",
                height=100,
                help="The retrieved document chunks used to generate the answer.",
                key="ragas_context",
            )

            if st.button("▶️ Run RAGAS Evaluation", use_container_width=True):
                if not _eval_question.strip() or not _eval_answer.strip():
                    st.warning("Please provide at least a question and an answer.")
                else:
                    _contexts = [
                        c.strip()
                        for c in _eval_context.split("---")
                        if c.strip()
                    ] if _eval_context.strip() else []

                    with st.spinner("Evaluating with LLM-as-judge…"):
                        try:
                            from app.rag.evaluator import RAGASEvaluator
                            _ragas = RAGASEvaluator()
                            _result = _ragas.evaluate(
                                question=_eval_question,
                                answer=_eval_answer,
                                contexts=_contexts,
                            )
                            st.markdown("**Evaluation Results**")
                            _r1, _r2, _r3 = st.columns(3)
                            _r1.metric("Faithfulness", f"{_result['faithfulness']:.2f}")
                            _r2.metric("Answer Relevancy", f"{_result['answer_relevancy']:.2f}")
                            _r3.metric("Context Precision", f"{_result['context_precision']:.2f}")
                            st.metric("Overall Score", f"{_result['overall']:.2f}")
                            with st.expander("Reasoning"):
                                st.write(f"**Faithfulness:** {_result.get('faithfulness_reason', '')}")
                                st.write(f"**Answer Relevancy:** {_result.get('answer_relevancy_reason', '')}")
                                st.write(f"**Context Precision:** {_result.get('context_precision_reason', '')}")
                        except Exception as _eval_err:
                            st.error(f"Evaluation failed: {_eval_err}")

        with st.expander("🚀 Coming Soon"):
            st.markdown("""
            - 🔄 Import Chat History
            - 📧 Email Export
            - 📊 Chat Analytics
            """)


def render_chat_message(message: Dict[str, str]):
    """
    Render a single chat message.
    
    Args:
        message: Dictionary with 'role' and 'content' keys
    """
    role = message["role"]
    content = message["content"]
    
    # Render message with appropriate avatar
    with st.chat_message(role):
        st.markdown(content)
        
        # Display sources if available (for assistant messages)
        if role == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources and len(sources) > 0:
                with st.expander(f"📚 View Sources ({len(sources)})"):
                    for idx, source in enumerate(sources, 1):
                        st.markdown(f"**Source {idx}:**")
                        st.text(source.get("content", "")[:200] + "...")
                        if "metadata" in source:
                            st.caption(f"Metadata: {source['metadata']}")
                        st.divider()


def render_chat_history():
    """
    Render all messages in chat history.
    """
    messages = get_chat_history()
    
    # Show welcome message if no history
    if not messages:
        with st.chat_message("assistant"):
            st.markdown("""
            👋 **Welcome!** I'm your GenAI Knowledge Assistant.
            
            I can help you with:
            - 📄 Searching through PDF documents
            - 🌐 Finding information on the web
            - 📖 Looking up Wikipedia articles
            - 🔬 Searching academic papers on ArXiv
            
            Ask me anything to get started!
            """)
        return
    
    # Render each message
    for message in messages:
        render_chat_message(message)


def handle_user_input():
    """
    Handle user input from chat box.
    
    This function:
    1. Gets user input from chat_input widget
    2. Validates input (non-empty)
    3. Adds user message to history
    4. Calls backend to get response
    5. Adds assistant response to history
    6. Triggers UI refresh
    """
    # Get user input
    user_input = st.chat_input("Ask a question...")
    
    if user_input:
        # Validate input
        user_input = user_input.strip()
        if not user_input:
            st.warning("Please enter a valid question.")
            return
        
        # Add user message to history
        add_message("user", user_input)
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call backend
                    result = get_assistant_response(user_input)
                    
                    # Extract response
                    answer = result.get("answer", "I couldn't generate a response.")
                    sources = result.get("sources", [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources if available
                    if sources and len(sources) > 0:
                        with st.expander(f"📚 View Sources ({len(sources)})"):
                            for idx, source in enumerate(sources, 1):
                                st.markdown(f"**Source {idx}:**")
                                st.text(source.get("content", "")[:200] + "...")
                                if "metadata" in source:
                                    st.caption(f"Metadata: {source['metadata']}")
                                st.divider()
                    
                    # Add assistant message to history
                    add_message("assistant", answer, sources)
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    add_message("assistant", error_msg)
        
        # Rerun to update display
        st.rerun()


# =============================================================================
# Main Application
# =============================================================================

def main():
    """
    Main application entry point.
    
    This function:
    1. Sets page configuration
    2. Initializes session state
    3. Renders UI components
    4. Handles user interaction
    """
    # Configure Streamlit page
    st.set_page_config(
        page_title="GenAI Knowledge Assistant",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render UI components
    render_header()
    render_sidebar()
    render_chat_history()
    
    # Handle user input
    handle_user_input()


class ChatApp:
    """Compatibility wrapper used by run.py.

    Everything is implemented as functions, so this class simply exposes a
    `run` method that delegates to the functional entry point.
    """

    def run(self) -> None:
        main()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
