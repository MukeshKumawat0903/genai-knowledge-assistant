"""
RAG Retriever for Retrieval-Augmented Generation Systems

This module provides a clean, configuration-driven retriever that fetches
relevant context documents from a vector store for RAG question answering.

Purpose:
    - Wrap LangChain vector store retrievers
    - Support configurable top-k retrieval
    - Enable history-aware query rewriting for conversational RAG
    - Read all configuration from centralized Settings
    - Keep retrieval logic isolated from generation logic

Why This Matters:
    - Retrieval quality directly impacts RAG answer quality
    - Poor retrieval = irrelevant context = wrong answers
    - Configuration-driven approach enables easy experimentation
    - Modular design allows swapping retrieval strategies

Design Pattern:
    - Wrapper Pattern: Wraps LangChain retriever with clean interface
    - Strategy Pattern: Different retrieval strategies (similarity, MMR, etc.)
    - Dependency Injection: Accepts vector store and configuration

Usage Example:
    ```python
    from app.rag.retriever import RAGRetriever, create_retriever
    from app.core.vector_store import VectorStoreManager
    from app.core.embeddings import EmbeddingManager
    from app.utils.config import get_settings
    
    # 1. Load vector store
    settings = get_settings()
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.get_embeddings()
    vector_store_manager = VectorStoreManager(settings, embeddings)
    vector_store = vector_store_manager.load_vector_store()
    
    # 2. Create retriever
    retriever = RAGRetriever(vector_store, settings)
    
    # 3. Retrieve documents
    docs = retriever.retrieve("What is RAG?")
    for doc in docs:
        print(f"- {doc.page_content[:100]}...")
    
    # 4. With conversation history
    chat_history = [("What is RAG?", "RAG is...")]
    docs = retriever.retrieve("How does it work?", chat_history=chat_history)
    ```

Architecture:
    [Vector Store] → [RAGRetriever] → [Relevant Documents] → [LLM Chain]
                      ^^^^^^^^^^^^
                    (This Module)
    
    Ingestion:  Load → Chunk → Embed → Index
    Retrieval:  Query → Fetch relevant docs (THIS MODULE)
    Generation: Docs → LLM → Answer
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

# Imports are kept at module scope so tests can monkeypatch
# app.rag.retriever.get_settings / EmbeddingManager / VectorStoreManager.
from app.utils.config import Settings, get_settings
from app.core.embeddings import EmbeddingManager
from app.core.vector_store import VectorStoreManager


class RAGRetriever:
    """
    Retriever for fetching relevant context documents from a vector store.
    
    This class wraps a LangChain-compatible vector store and provides a clean
    interface for document retrieval in RAG systems. It supports configurable
    top-k retrieval, different search strategies, and optional history-aware
    query rewriting for conversational RAG.
    
    Responsibilities:
        - Wrap vector store with retriever interface
        - Configure retrieval parameters (top-k, search type)
        - Validate queries and handle edge cases
        - Optionally rewrite queries based on chat history
        - Return ranked, relevant documents
    
    NOT responsible for:
        - Generating answers (handled by LLM chain)
        - Implementing prompts (handled by chain/agent)
        - Document chunking (handled by indexer)
        - Storing embeddings (handled by vector store)
    
    Design Philosophy:
        - Configuration over hard-coding
        - Simple, focused interface
        - Clear separation from generation
        - Easy to test and mock
    
    Attributes:
        vector_store (VectorStore): LangChain-compatible vector store
        settings (Settings): Configuration object
        top_k (int): Number of documents to retrieve
        search_type (str): Type of search ("similarity", "mmr", "similarity_score_threshold")
        retriever: LangChain retriever object
    
    Example:
        ```python
        # Create retriever
        settings = get_settings()
        retriever = RAGRetriever(vector_store, settings)
        
        # Simple retrieval
        docs = retriever.retrieve("What is machine learning?")
        print(f"Found {len(docs)} relevant documents")
        
        # With custom k
        docs = retriever.retrieve("Explain neural networks", k=10)
        
        # Conversational retrieval
        history = [("What is ML?", "ML is...")]
        docs = retriever.retrieve("How does it work?", chat_history=history)
        ```
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        settings: Settings,
        search_type: str = "similarity"
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            vector_store: LangChain-compatible vector store instance
            settings: Settings object with retrieval configuration
            search_type: Search strategy to use
                - "similarity": Standard similarity search
                - "mmr": Maximum Marginal Relevance (diverse results)
                - "similarity_score_threshold": Filter by similarity score
                
        Raises:
            ValueError: If vector_store is None or invalid
        """
        # Validate inputs
        if vector_store is None:
            raise ValueError("vector_store cannot be None")
        
        if not hasattr(vector_store, 'as_retriever'):
            raise ValueError(
                "vector_store must be a LangChain-compatible VectorStore "
                "with as_retriever() method"
            )
        
        self.vector_store = vector_store
        self.settings = settings
        self.top_k = settings.retriever_top_k
        self.search_type = search_type
        
        # Create the retriever with configuration
        self.retriever = self._create_retriever()
    
    def _create_retriever(self):
        """
        Create a configured LangChain retriever from the vector store.
        
        This method wraps the vector store's as_retriever() method and
        configures it with the appropriate search parameters based on
        the search_type.
        
        Returns:
            Configured LangChain retriever
        """
        retriever_kwargs = {
            "search_type": self.search_type,
            "search_kwargs": {"k": self.top_k}
        }
        
        # Add search type-specific parameters
        if self.search_type == "similarity_score_threshold":
            retriever_kwargs["search_kwargs"]["score_threshold"] = \
                self.settings.similarity_threshold
        elif self.search_type == "mmr":
            # MMR-specific parameters
            retriever_kwargs["search_kwargs"]["fetch_k"] = self.top_k * 2
            retriever_kwargs["search_kwargs"]["lambda_mult"] = 0.5
        
        return self.vector_store.as_retriever(**retriever_kwargs)
    
    def get_retriever(self):
        """
        Get the underlying LangChain retriever.
        
        Useful for:
        - Passing to LangChain chains (e.g., RetrievalQA)
        - Direct access to retriever methods
        - Integration with LangChain agents
        
        Returns:
            LangChain retriever object
            
        Example:
            ```python
            rag_retriever = RAGRetriever(vector_store, settings)
            langchain_retriever = rag_retriever.get_retriever()
            
            # Use with RetrievalQA chain
            from langchain.chains import RetrievalQA
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=langchain_retriever
            )
            ```
        """
        return self.retriever
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        This is the main retrieval method. It validates the query, optionally
        rewrites it based on chat history, and returns the most relevant documents.
        
        Args:
            query: User query string
            k: Number of documents to retrieve (overrides default top_k)
            chat_history: Optional list of (question, answer) tuples for
                         context-aware retrieval
            **kwargs: Additional parameters to pass to retriever
                     (e.g., filter for metadata filtering)
        
        Returns:
            List of LangChain Document objects with page_content and metadata
            
        Raises:
            ValueError: If query is empty or None
            
        Example:
            ```python
            # Simple retrieval
            docs = retriever.retrieve("What is RAG?")
            
            # Override default k
            docs = retriever.retrieve("Explain embeddings", k=10)
            
            # Conversational retrieval
            history = [
                ("What is machine learning?", "ML is a subset of AI..."),
                ("What are neural networks?", "Neural networks are...")
            ]
            docs = retriever.retrieve("How do they learn?", chat_history=history)
            
            # With metadata filtering
            docs = retriever.retrieve(
                "Python tutorial",
                filter={"source": "python_docs"}
            )
            ```
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or None")
        
        # Rewrite query if chat history provided (history-aware retrieval)
        if chat_history:
            query = self._rewrite_query_with_history(query, chat_history)
        
        # Override k if provided
        if k is not None:
            # Temporarily update retriever's search_kwargs
            original_k = self.retriever.search_kwargs.get("k", self.top_k)
            self.retriever.search_kwargs["k"] = k
            
            try:
                documents = self._run_retrieval(query, **kwargs)
            finally:
                # Restore original k
                self.retriever.search_kwargs["k"] = original_k
        else:
            # Use default k
            documents = self._run_retrieval(query, **kwargs)
        
        return documents

    def _run_retrieval(self, query: str, **kwargs) -> List[Document]:
        """Run retrieval against the underlying LangChain retriever.

        Tests in this repo expect the retriever to call `get_relevant_documents`
        and forward kwargs (e.g., `filter={...}`). Newer LangChain retrievers
        also support `.invoke()`.
        """
        if hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.get_relevant_documents(query, **kwargs)
        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(query, **kwargs)
        # Last-resort fallback for callable retrievers
        if callable(self.retriever):
            return self.retriever(query, **kwargs)
        raise RuntimeError("Underlying retriever does not support retrieval methods")
    
    def _rewrite_query_with_history(
        self,
        query: str,
        chat_history: List[Tuple[str, str]]
    ) -> str:
        """
        Rewrite a follow-up query to be standalone using chat history.
        
        This method handles conversational queries like "How does it work?"
        that depend on previous context. It combines the chat history with
        the current query to create a self-contained query.
        
        For example:
            History: [("What is RAG?", "RAG combines retrieval with generation")]
            Query: "How does it work?"
            Rewritten: "How does RAG (Retrieval-Augmented Generation) work?"
        
        Note: This is a simple implementation. For production, consider using
        LangChain's create_history_aware_retriever with an LLM.
        
        Args:
            query: Current query that may reference chat history
            chat_history: List of (question, answer) tuples
        
        Returns:
            Rewritten standalone query
        """
        normalized = query.strip().lower()
        words = normalized.split()

        # Keep the heuristic intentionally conservative:
        # rewrite only when the query looks referential ("it/this/that") or
        # is a classic follow-up question (how/what/why/when/where/who).
        wh_starters = ("how", "what", "why", "when", "where", "who")
        pronoun_starters = ("it", "this", "that", "they", "them", "these", "those")

        starts_with_wh = normalized.startswith(wh_starters)
        starts_with_pronoun = any(
            normalized == p or normalized.startswith(p + " ")
            for p in pronoun_starters
        )

        is_short = len(words) <= 10
        is_follow_up = is_short and (starts_with_wh or starts_with_pronoun)
        
        if is_follow_up and chat_history:
            # Get the last question as context
            last_question, _ = chat_history[-1]
            # Combine last question with current query
            rewritten = f"Given the previous question '{last_question}', {query}"
            return rewritten
        
        # Return original query if no rewriting needed
        return query
    
    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with their similarity scores.
        
        Useful for:
        - Understanding retrieval quality
        - Filtering by score threshold
        - Debugging retrieval issues
        - Logging and monitoring
        
        Args:
            query: User query string
            k: Number of documents to retrieve
            **kwargs: Additional parameters
        
        Returns:
            List of (Document, score) tuples, sorted by score descending
            
        Example:
            ```python
            docs_with_scores = retriever.retrieve_with_scores("What is RAG?")
            
            for doc, score in docs_with_scores:
                print(f"Score: {score:.3f}")
                print(f"Content: {doc.page_content[:100]}...")
                print()
            ```
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or None")
        
        # Use vector store's similarity_search_with_score
        k_to_use = k if k is not None else self.top_k
        
        if hasattr(self.vector_store, 'similarity_search_with_score'):
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query,
                k=k_to_use,
                **kwargs
            )
            return docs_and_scores
        else:
            # Fallback: retrieve without scores
            docs = self.retrieve(query, k=k, **kwargs)
            # Return with dummy scores
            return [(doc, 1.0) for doc in docs]


# ============================================================================
# Convenience Functions
# ============================================================================

def create_retriever(
    vector_store: Optional[VectorStore] = None,
    top_k: Optional[int] = None,
    search_type: str = "similarity"
) -> RAGRetriever:
    """
    Convenience function to create a fully configured RAGRetriever.
    
    This function handles:
    1. Loading Settings
    2. Loading vector store (if not provided)
    3. Creating RAGRetriever with configuration
    
    Args:
        vector_store: Optional pre-loaded vector store. If None, will load
                     from configured path using VectorStoreManager
        top_k: Optional override for retriever_top_k from Settings
        search_type: Search strategy ("similarity", "mmr", "similarity_score_threshold")
    
    Returns:
        Fully configured RAGRetriever instance
        
    Raises:
        RuntimeError: If vector store cannot be loaded
        
    Example:
        ```python
        # Use defaults from .env
        retriever = create_retriever()
        docs = retriever.retrieve("What is RAG?")
        
        # With custom top_k
        retriever = create_retriever(top_k=10)
        
        # With pre-loaded vector store
        retriever = create_retriever(vector_store=my_vector_store)
        
        # With MMR search
        retriever = create_retriever(search_type="mmr")
        ```
    """
    # Load settings
    settings = get_settings()
    
    # Override top_k if provided
    if top_k is not None:
        settings.retriever_top_k = top_k
    
    # Load vector store if not provided
    if vector_store is None:
        # Initialize embedding manager
        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.get_embeddings()
        
        # Initialize vector store manager
        vector_store_manager = VectorStoreManager(settings, embeddings)
        
        # Load existing vector store
        try:
            vector_store = vector_store_manager.load_vector_store()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load vector store: {e}\n"
                "Make sure you have indexed documents first using DocumentIndexer."
            )
    
    # Create and return retriever
    return RAGRetriever(vector_store, settings, search_type=search_type)


# ============================================================================
# Quick Usage Example
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of the RAGRetriever.
    
    This example shows how to use the retriever to fetch relevant documents
    from a vector store.
    """
    print("=" * 80)
    print("RAG Retriever - Demo")
    print("=" * 80)
    print()
    
    try:
        print("🔧 Creating retriever...")
        retriever = create_retriever()
        print("✓ Retriever created")
        print()
        
        # Sample query
        query = "What is Retrieval-Augmented Generation?"
        print(f"📝 Query: {query}")
        print()
        
        print("🔍 Retrieving relevant documents...")
        docs = retriever.retrieve(query)
        print()
        
        print(f"✓ Found {len(docs)} relevant documents:")
        print("-" * 80)
        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. {doc.page_content[:200]}...")
            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        print()
        
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print()
        print("Note: This demo requires:")
        print("  1. An existing vector store (run indexer first)")
        print("  2. Configured .env file")
        print("  3. Required packages (langchain, sentence-transformers)")
        print()
        print("Quick setup:")
        print("  from app.rag.indexer import create_indexer")
        print("  indexer = create_indexer()")
        print("  indexer.index_documents(documents)")
