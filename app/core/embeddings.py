"""
Embedding Manager for RAG System

This module provides a clean, reusable embedding engine that converts text
into dense vector representations for semantic search in RAG systems.

Why Embeddings Matter in RAG:
------------------------------
In Retrieval-Augmented Generation, we need to find relevant documents based
on semantic similarity, not just keyword matching. Embeddings transform text
into high-dimensional vectors where semantically similar texts are close
together in vector space.

For example:
- "What is AI?" and "Explain artificial intelligence"
  → Very similar embeddings (close in vector space)
- "What is AI?" and "Recipe for chocolate cake"
  → Very different embeddings (far apart in vector space)

Key Design Principles:
----------------------
- Lazy Initialization: Create embedding model once, reuse many times
- Configuration-Driven: All settings from centralized Settings
- LangChain-Compatible: Returns objects that work with LangChain RAG chains
- Simple Interface: One method `get_embeddings()` does everything

Usage Example:
--------------
    from app.core.embeddings import EmbeddingManager
    
    # Initialize manager (reads config automatically)
    manager = EmbeddingManager()
    
    # Get embeddings object (initialized once, cached)
    embeddings = manager.get_embeddings()
    
    # Use in RAG pipeline
    vector_store = FAISS.from_documents(documents, embeddings)
"""

from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from app.utils.config import get_settings


class EmbeddingManager:
    """
    Manages embedding model initialization and provides LangChain-compatible embeddings.
    
    This class implements lazy initialization: the embedding model is only created
    when first requested, then cached for subsequent use. This is efficient because:
    1. Avoids loading the model if embeddings aren't needed
    2. Loads the model only once (can be 100-500MB)
    3. Reuses the same instance across the application
    
    Responsibilities:
    -----------------
    - Read embedding configuration from Settings
    - Initialize HuggingFace embedding model on first use
    - Cache the model instance for reuse
    - Validate configuration before initialization
    - Return LangChain-compatible embeddings object
    
    Why HuggingFace?
    ----------------
    - Free and open-source (no API keys required)
    - Runs locally (data privacy)
    - High-quality models available (all-MiniLM-L6-v2, etc.)
    - Fast inference on CPU or GPU
    - Compatible with LangChain ecosystem
    
    Thread Safety:
    --------------
    This class is NOT thread-safe. If using in multi-threaded environment,
    create separate instances per thread or add locking mechanisms.
    """
    
    def __init__(self):
        """
        Initialize the EmbeddingManager.
        
        Note: The actual embedding model is NOT loaded here. It will be loaded
        lazily when get_embeddings() is first called. This keeps initialization
        fast and avoids loading models that may not be used.
        """
        # Cache for the embedding model instance (lazy initialization)
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get or create the HuggingFace embeddings instance.
        
        This method implements lazy initialization pattern:
        - First call: Creates and caches the embedding model
        - Subsequent calls: Returns the cached instance
        
        The embedding model is loaded from HuggingFace Hub and runs locally.
        First load may take 10-30 seconds depending on model size and internet
        speed. Subsequent calls are instant.
        
        Returns:
            HuggingFaceEmbeddings: A LangChain-compatible embeddings object that
                can be used with vector stores, retrievers, and RAG chains.
                
        Raises:
            ValueError: If embedding_model_name is not set in configuration.
            RuntimeError: If model fails to load (e.g., invalid model name,
                network issues, insufficient memory).
                
        Example:
            manager = EmbeddingManager()
            embeddings = manager.get_embeddings()
            
            # Use with vector store
            vector_store = FAISS.from_texts(texts, embeddings)
            
            # Or use to embed queries
            query_vector = embeddings.embed_query("What is RAG?")
        """
        # If already initialized, return cached instance
        if self._embeddings is not None:
            return self._embeddings
        
        # Get configuration from Settings singleton
        settings = get_settings()
        
        # Validate that embedding model name is configured
        if not settings.embedding_model_name:
            raise ValueError(
                "Embedding model name is not configured. "
                "Please set EMBEDDING_MODEL_NAME in your .env file.\n"
                "Recommended: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)"
            )
        
        # Create HuggingFace embeddings instance
        # This downloads the model from HuggingFace Hub on first use
        # Subsequent runs will use the cached model from ~/.cache/huggingface/
        try:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                # Encode parameters for consistent embeddings
                encode_kwargs={
                    'normalize_embeddings': True,  # Normalize to unit vectors for cosine similarity
                    'batch_size': 32,  # Process multiple texts at once for efficiency
                }
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize embedding model '{settings.embedding_model_name}'. "
                f"Error: {str(e)}\n"
                f"Common causes:\n"
                f"  • Invalid model name (check HuggingFace Hub)\n"
                f"  • Network issues (model downloads from HuggingFace)\n"
                f"  • Insufficient memory (some models need 2-4GB RAM)\n"
                f"Recommended models:\n"
                f"  • all-MiniLM-L6-v2 (384 dim, fast, 80MB)\n"
                f"  • all-mpnet-base-v2 (768 dim, better quality, 420MB)"
            )
        
        return self._embeddings
#    import torch
#    device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 
# 2. Pass device to HuggingFaceEmbeddings:
#    HuggingFaceEmbeddings(
#        model_name=settings.embedding_model_name,
#        model_kwargs={'device': device},
#        encode_kwargs={'normalize_embeddings': True}
#    )
# 
# 3. Performance improvement:
#    CPU: ~100 texts/second
#    GPU: ~1000 texts/second (10x faster!)
# 
# 4. Requirements:
#    - CUDA-capable GPU
#    - PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118
