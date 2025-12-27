"""
Vector Store Manager for RAG Systems

This module provides a clean, extensible factory for managing vector databases
in Retrieval-Augmented Generation (RAG) systems. It abstracts multiple vector
store implementations behind a single interface.

Purpose:
    - Centralize vector store creation and management
    - Support multiple vector databases (FAISS, Chroma, Pinecone)
    - Enable easy switching between vector stores via configuration
    - Provide consistent interface for document indexing and loading

Why This Matters:
    - Vector stores are critical for semantic search in RAG
    - Different stores have different trade-offs (speed, scale, cost)
    - Abstraction allows experimentation without code changes
    - Production systems often need to migrate between stores

Vector Store Trade-offs:

    FAISS (Facebook AI Similarity Search):
        ✓ Fastest for similarity search
        ✓ Best for in-memory operations
        ✓ No external dependencies
        ✓ Good for development/prototyping
        ✗ Not distributed (single machine)
        ✗ Limited to memory size
        ✗ No built-in persistence (save/load manually)
        Use Case: Development, small datasets (<1M vectors)

    Chroma:
        ✓ Built-in persistence (no manual save/load)
        ✓ Metadata filtering
        ✓ Easy to use API
        ✓ Good for local production
        ✗ Slower than FAISS for large datasets
        ✗ Not horizontally scalable
        Use Case: Small-medium production, local deployments

    Pinecone:
        ✓ Cloud-native, fully managed
        ✓ Horizontally scalable
        ✓ High availability
        ✓ Advanced filtering and metadata
        ✓ No infrastructure management
        ✗ Requires API key and internet
        ✗ Usage-based pricing
        ✗ Latency (network calls)
        Use Case: Large-scale production (>10M vectors)

Usage Example:
    ```python
    from app.core.vector_store import VectorStoreManager
    from app.core.embeddings import EmbeddingFactory
    from app.core.text_splitter import TextChunker
    from app.utils.config import get_settings
    
    # Load and chunk documents
    documents = load_documents()
    chunker = TextChunker(get_settings())
    chunks = chunker.split_documents(documents)
    
    # Create embeddings
    settings = get_settings()
    embedding_factory = EmbeddingFactory(settings)
    embeddings = embedding_factory.create_embeddings()
    
    # Create vector store
    vector_manager = VectorStoreManager(settings, embeddings)
    vectorstore = vector_manager.create_vector_store(chunks)
    
    # Later: Load existing vector store
    vectorstore = vector_manager.load_vector_store()
    
    # Query
    results = vectorstore.similarity_search("What is RAG?", k=4)
    ```

Architecture:
    Document Loading → Chunking → Embedding → Vector Storage → Retrieval
                                               ^^^^^^^^^^^^^^
                                             (This Module)
"""

from typing import Optional, List, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Import vector store implementations (LangChain 1.x split packages)
from langchain_community.vectorstores import FAISS, Chroma
# Pinecone imported conditionally (requires API key setup)

# Import our configuration
from app.utils.config import Settings


class VectorStoreManager:
    """
    Vector store manager for RAG systems. Supports FAISS, Chroma, and Pinecone backends.
    
    This manager provides a unified interface for creating, loading, and managing
    vector databases across different implementations. It handles:
    - Vector store creation from document chunks
    - Persistence and loading of existing stores
    - Configuration-based backend selection
    - Directory management and validation
    
    Supported Backends:
        - FAISS: In-memory, fastest search, best for development (<1M vectors)
        - Chroma: Persistent, auto-saves, good for local production
        - Pinecone: Cloud-native, scalable, best for large production (>10M vectors)
    
    Example:
        manager = VectorStoreManager(settings, embeddings)
        
        # Create new vector store from documents
        vectorstore = manager.create_vector_store(documents)
        
        # Later: Load existing vector store
        vectorstore = manager.load_vector_store()
        
        # Query the store
        results = vectorstore.similarity_search("What is RAG?", k=4)
    """
    
    def __init__(self, settings: Settings, embeddings: Embeddings):
        """
        Initialize vector store manager with configuration and embedding model.
        
        Args:
            settings: Application settings containing vector store configuration:
                - vector_store_type: Type of vector store ("faiss", "chroma", "pinecone")
                - vector_store_path: Path for persistent storage (FAISS/Chroma)
                - collection_name: Collection identifier (Chroma/Pinecone)
            embeddings: LangChain embeddings instance for generating vectors
        
        Raises:
            ValueError: If vector_store_type is not supported
        """
        self.settings = settings
        self.embeddings = embeddings
        
        # Read vector store configuration from Settings
        self.vector_store_type: str = settings.vector_store_type.lower()
        # Treat Settings.vector_store_path as a *base* directory and isolate
        # storage per backend type (faiss/chroma/...). This prevents different
        # backends from overwriting each other's on-disk formats and keeps UI
        # indexing and loading paths consistent.
        self.vector_store_path: Path = self._resolve_store_path(
            base_path=settings.vector_store_path,
            vector_store_type=self.vector_store_type,
        )
        self.collection_name: str = settings.collection_name
        
        # Validate vector store type
        self._validate_vector_store_type()
        
        # Ensure persistence directory exists for local stores
        if self.vector_store_type in ["faiss", "chroma"]:
            self.vector_store_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_store_path(base_path: Path, vector_store_type: str) -> Path:
        """Resolve the on-disk directory for a given vector store type.

        If the caller already provided a type-scoped path (e.g., .../faiss),
        return it as-is. Otherwise, append the type as a subdirectory.
        """
        # Normalize without touching the filesystem.
        base = Path(base_path)
        # Avoid double-appending if already scoped.
        if base.name.lower() == vector_store_type.lower():
            return base
        return base / vector_store_type
    
    def create_vector_store(
        self,
        documents: List[Document],
        **kwargs: Any
    ):
        """
        Create a new vector store from documents.
        
        This method:
            1. Validates input documents
            2. Creates appropriate vector store based on configuration
            3. Indexes all documents with embeddings
            4. Returns LangChain-compatible vector store
        
        The returned vector store can be used for:
            - Similarity search
            - Retrieval in RAG pipelines
            - Metadata filtering (if supported by backend)
        
        Args:
            documents: List of LangChain Document objects to index.
                      Each document should have:
                      - page_content: Text content to embed
                      - metadata: Dict with source info, chunk info, etc.
            
            **kwargs: Additional backend-specific parameters
                     (e.g., distance_metric for FAISS)
        
        Returns:
            LangChain vector store instance (FAISS, Chroma, or Pinecone)
            with documents indexed and ready for querying
        
        Raises:
            ValueError: If documents list is empty or None
            ValueError: If vector store type is unsupported
        
        Example:
            ```python
            # Load and chunk documents
            documents = load_documents()
            chunks = chunk_documents(documents)
            
            # Create vector store
            manager = VectorStoreManager(settings, embeddings)
            vectorstore = manager.create_vector_store(chunks)
            
            # Query immediately
            results = vectorstore.similarity_search("query", k=4)
            ```
        
        Performance:
            - Time: O(n × d) where n = docs, d = embedding dimension
            - Memory: Proportional to number of documents
            - FAISS: Fastest creation (in-memory)
            - Chroma: Medium (writes to disk)
            - Pinecone: Slowest (network API calls)
        """
        # Validate input
        if not documents:
            raise ValueError(
                "Cannot create vector store from empty document list. "
                "Please provide at least one document with page_content."
            )
        
        # Create vector store based on type
        if self.vector_store_type == "faiss":
            return self._create_faiss_store(documents, **kwargs)
        
        elif self.vector_store_type == "chroma":
            return self._create_chroma_store(documents, **kwargs)
        
        elif self.vector_store_type == "pinecone":
            return self._create_pinecone_store(documents, **kwargs)
        
        else:
            # This should never happen due to validation in __init__
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
    
    def load_vector_store(self, **kwargs: Any):
        """
        Load an existing vector store from persistence.
        
        This method loads a previously created and saved vector store,
        allowing you to resume work without re-indexing documents.
        
        Behavior by vector store:
            - FAISS: Loads from .faiss index file
            - Chroma: Loads from persistent directory
            - Pinecone: Connects to existing cloud index
        
        Args:
            **kwargs: Backend-specific loading parameters
        
        Returns:
            LangChain vector store instance ready for querying
        
        Raises:
            FileNotFoundError: If vector store doesn't exist (FAISS, Chroma)
            ValueError: If vector store type is unsupported
        
        Example:
            ```python
            # Load existing vector store
            manager = VectorStoreManager(settings, embeddings)
            vectorstore = manager.load_vector_store()
            
            # Query immediately
            results = vectorstore.similarity_search("query", k=4)
            ```
        
        Note:
            For FAISS and Chroma, this requires that a vector store was
            previously created and saved at vector_store_path.
        """
        if self.vector_store_type == "faiss":
            return self._load_faiss_store(**kwargs)
        
        elif self.vector_store_type == "chroma":
            return self._load_chroma_store(**kwargs)
        
        elif self.vector_store_type == "pinecone":
            return self._load_pinecone_store(**kwargs)
        
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
    
    def save_vector_store(self, vectorstore: Any) -> None:
        """
        Save vector store to persistent storage.
        
        Behavior by vector store type:
        - FAISS: Explicitly saves index to disk (required)
        - Chroma: Auto-persists, no action needed
        - Pinecone: Cloud-based, no action needed
        
        Args:
            vectorstore: Vector store instance to save
        
        Note:
            Only call this for FAISS stores after creation or updates.
            Chroma and Pinecone handle persistence automatically.
        """
        if self.vector_store_type == "faiss":
            # FAISS requires explicit save
            save_path = str(self.vector_store_path / "index")
            vectorstore.save_local(save_path)
            print(f"✓ FAISS index saved to {save_path}")
        
        elif self.vector_store_type == "chroma":
            # Chroma auto-persists, no action needed
            print("✓ Chroma auto-persisted (no manual save needed)")
        
        elif self.vector_store_type == "pinecone":
            # Pinecone is cloud-based, no action needed
            print("✓ Pinecone index auto-saved (cloud-based)")
    
    # =========================================================================
    # FAISS Implementation
    # =========================================================================
    
    def _create_faiss_store(
        self,
        documents: List[Document],
        **kwargs: Any
    ) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        FAISS (Facebook AI Similarity Search) is an in-memory vector database
        optimized for fast similarity search. Best for development and
        small-to-medium datasets that fit in memory.
        
        Args:
            documents: List of Document objects to index
            **kwargs: Additional FAISS parameters
        
        Returns:
            FAISS vector store instance
        
        Performance:
            - Very fast indexing and search (in-memory)
            - Memory usage: ~4 bytes per dimension per vector
            - Best for: <1M vectors, development, prototyping
        """
        print(f"Creating FAISS vector store with {len(documents)} documents...")
        
        # Create FAISS index from documents
        # This will:
        #   1. Generate embeddings for all documents
        #   2. Build FAISS index with embeddings
        #   3. Store document texts and metadata
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
            **kwargs
        )
        
        print(f"✓ FAISS index created with {len(documents)} vectors")
        return vectorstore
    
    def _load_faiss_store(self, **kwargs: Any) -> FAISS:
        """
        Load existing FAISS vector store from disk.
        
        Args:
            **kwargs: Additional FAISS loading parameters
        
        Returns:
            FAISS vector store instance
        
        Raises:
            FileNotFoundError: If index file doesn't exist
        """
        index_path = str(self.vector_store_path / "index")

        # Check if index exists (new layout: <base>/<type>/index)
        if not Path(index_path).exists():
            # Backward-compatible fallback (legacy layout: <base>/index)
            legacy_index_path = None
            if self.vector_store_path.name.lower() == self.vector_store_type.lower():
                legacy_index_path = str(self.vector_store_path.parent / "index")

            if legacy_index_path and Path(legacy_index_path).exists():
                index_path = legacy_index_path
            else:
                raise FileNotFoundError(
                    f"FAISS index not found at {str(self.vector_store_path / 'index')}. "
                    f"Create a new index with create_vector_store() first."
                )
        
        print(f"Loading FAISS index from {index_path}...")
        
        # Load FAISS index
        # Note: allow_dangerous_deserialization=True is required for modern LangChain
        # This is safe for locally-created indexes but should not be used with untrusted sources
        vectorstore = FAISS.load_local(
            folder_path=index_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
            **kwargs
        )
        
        print(f"✓ FAISS index loaded successfully")
        return vectorstore
    
    # =========================================================================
    # Chroma Implementation
    # =========================================================================
    
    def _create_chroma_store(
        self,
        documents: List[Document],
        **kwargs: Any
    ) -> Chroma:
        """
        Create Chroma vector store from documents.
        
        Chroma is a persistent vector database with built-in metadata filtering.
        Good for local production deployments with automatic persistence.
        
        Args:
            documents: List of Document objects to index
            **kwargs: Additional Chroma parameters
        
        Returns:
            Chroma vector store instance
        
        Features:
            - Automatic persistence (no manual save needed)
            - Metadata filtering support
            - Easy to use API
            - Good for local production
        """
        print(f"Creating Chroma vector store with {len(documents)} documents...")
        
        # Create Chroma collection with persistence
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.vector_store_path),
            **kwargs
        )
        
        print(f"✓ Chroma collection '{self.collection_name}' created")
        print(f"  Persisted to: {self.vector_store_path}")
        return vectorstore
    
    def _load_chroma_store(self, **kwargs: Any) -> Chroma:
        """
        Load existing Chroma vector store from disk.
        
        Args:
            **kwargs: Additional Chroma loading parameters
        
        Returns:
            Chroma vector store instance
        
        Raises:
            FileNotFoundError: If collection doesn't exist
        """
        # Check if persist directory exists
        if not self.vector_store_path.exists():
            raise FileNotFoundError(
                f"Chroma persist directory not found at {self.vector_store_path}. "
                f"Create a new collection with create_vector_store() first."
            )
        
        print(f"Loading Chroma collection '{self.collection_name}'...")
        
        # Load Chroma collection
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.vector_store_path),
            **kwargs
        )
        
        print(f"✓ Chroma collection loaded successfully")
        return vectorstore
    
    # =========================================================================
    # Pinecone Implementation (Placeholder)
    # =========================================================================
    
    def _create_pinecone_store(
        self,
        documents: List[Document],
        **kwargs: Any
    ):
        """
        Create Pinecone vector store from documents.
        
        Pinecone is a cloud-native, fully managed vector database.
        Best for large-scale production deployments (>10M vectors).
        
        Args:
            documents: List of Document objects to index
            **kwargs: Additional Pinecone parameters
        
        Returns:
            Pinecone vector store instance
        
        Note:
            This is a placeholder implementation. Full integration requires:
            - Pinecone API key (PINECONE_API_KEY in .env)
            - Pinecone environment (PINECONE_ENVIRONMENT in .env)
            - pip install pinecone-client
        """
        # TODO: Implement Pinecone integration
        raise NotImplementedError(
            "Pinecone integration not yet implemented. "
            "To use Pinecone:\n"
            "1. Install: pip install pinecone-client\n"
            "2. Set PINECONE_API_KEY in .env\n"
            "3. Set PINECONE_ENVIRONMENT in .env\n"
            "4. Implement _create_pinecone_store() method\n\n"
            "See TODO comments in this file for implementation details."
        )
        
        # TODO: Full implementation (see detailed TODO section below)
    
    def _load_pinecone_store(self, **kwargs: Any):
        """
        Load existing Pinecone vector store.
        
        Args:
            **kwargs: Additional Pinecone loading parameters
        
        Returns:
            Pinecone vector store instance
        """
        # TODO: Implement Pinecone loading
        raise NotImplementedError(
            "Pinecone integration not yet implemented. "
            "See _create_pinecone_store() for setup instructions."
        )
    
    # =========================================================================
    # Validation and Utilities
    # =========================================================================
    
    def _validate_vector_store_type(self) -> None:
        """
        Validate that vector store type is supported.
        
        Raises:
            ValueError: If vector_store_type is not supported
        """
        supported_types = ["faiss", "chroma", "pinecone"]
        
        if self.vector_store_type not in supported_types:
            raise ValueError(
                f"Unsupported vector store type: '{self.vector_store_type}'. "
                f"Supported types: {supported_types}. "
                f"Check VECTOR_STORE_TYPE in your .env configuration."
            )
    
    def get_vector_store_info(self) -> dict:
        """
        Get information about current vector store configuration.
        
        Returns:
            Dictionary containing:
            - type: Vector store backend type
            - path: Filesystem path for persistence
            - collection_name: Collection identifier
            - embeddings: Embedding model class name
        
        Useful for:
        - Debugging configuration issues
        - Logging vector store details
        - Verifying setup before indexing
        
        Example:
            info = manager.get_vector_store_info()
            print(f"Using {info['type']} at {info['path']}")
        """
        return {
            "type": self.vector_store_type,
            "path": str(self.vector_store_path),
            "collection_name": self.collection_name,
            "embeddings": type(self.embeddings).__name__
        }
    
    # TODO: Hybrid Search
    # -------------------
    # Combine semantic search (vector) with keyword search (BM25) for better results.
    #
    # Why hybrid search?
    #   - Vector search: Great for semantic similarity
    #   - Keyword search: Great for exact matches, names, IDs
    #   - Hybrid: Best of both worlds
    #
    # Implementation approach:
    #   1. Perform vector similarity search
    #   2. Perform keyword search (BM25, TF-IDF)
    #   3. Combine results with weighted score
    #   4. Re-rank based on combined score
    #
    # Example:
    #   def hybrid_search(
    #       self,
    #       query: str,
    #       k: int = 4,
    #       vector_weight: float = 0.7,


