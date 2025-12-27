"""
Document Indexer for RAG Systems

This module provides a clean, modular indexing pipeline that orchestrates
the complete flow from raw documents to searchable vector stores.

Purpose:
    - Connect document preprocessing, embedding, and vector storage
    - Provide a single entry point for indexing operations
    - Keep indexing logic isolated from retrieval and LLM logic
    - Enable easy testing and maintenance of the indexing pipeline

Why This Matters:
    - Indexing is the foundation of RAG - poor indexing = poor retrieval
    - Separating indexing from retrieval enables independent optimization
    - Modular design allows swapping components (e.g., different chunkers)
    - Clear pipeline makes debugging and monitoring easier

Pipeline Flow:
    Raw Documents → TextChunker → Chunks → VectorStoreManager → Vector Store
                    ^^^^^^^^^^^^   ^^^^^^   ^^^^^^^^^^^^^^^^^^
                    Split text     Embed    Store vectors

Design Pattern:
    - Facade Pattern: Simplifies complex subsystem interactions
    - Dependency Injection: Accepts configured components
    - Single Responsibility: Only handles indexing orchestration

Usage Example:
    ```python
    from app.rag.indexer import DocumentIndexer
    from app.core.text_splitter import TextChunker
    from app.core.embeddings import EmbeddingManager
    from app.core.vector_store import VectorStoreManager
    from app.ingestion.pdf_loader import PDFDocumentLoader
    from app.utils.config import get_settings
    
    # 1. Load raw documents
    pdf_loader = PDFDocumentLoader("./documents")
    documents = pdf_loader.load_documents()
    
    # 2. Initialize components
    settings = get_settings()
    text_chunker = TextChunker(settings)
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.get_embeddings()
    vector_store_manager = VectorStoreManager(settings, embeddings)
    
    # 3. Create indexer and index documents
    indexer = DocumentIndexer(
        text_chunker=text_chunker,
        embedding_manager=embedding_manager,
        vector_store_manager=vector_store_manager
    )
    
    result = indexer.index_documents(documents)
    print(f"Indexed {result['num_chunks']} chunks from {result['num_documents']} documents")
    ```

Architecture:
    [Ingestion Layer] → [Indexing Layer] → [Storage Layer]
                         ^^^^^^^^^^^^^^
                       (This Module)
    
    Ingestion: Load raw documents (PDF, Web, YouTube, etc.)
    Indexing:  Split → Embed → Store (THIS MODULE)
    Retrieval: Query vector store, retrieve relevant chunks
    Generation: Pass chunks to LLM for answer generation
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.documents import Document

# Import our core components
from app.core.text_splitter import TextChunker
from app.core.embeddings import EmbeddingManager
from app.core.vector_store import VectorStoreManager
from app.utils.config import get_settings


class DocumentIndexer:
    """
    Orchestrates the document indexing pipeline for RAG systems.
    
    This class connects text chunking, embedding generation, and vector storage
    into a cohesive indexing pipeline. It provides a clean interface for converting
    raw documents into searchable vector stores.
    
    Responsibilities:
        - Validate input documents
        - Apply text chunking via TextChunker
        - Coordinate embedding generation (via EmbeddingManager)
        - Create or update vector stores via VectorStoreManager
        - Return indexing metadata for monitoring
    
    NOT responsible for:
        - Document loading (handled by ingestion layer)
        - Retrieval operations (handled by retriever layer)
        - LLM interactions (handled by chain layer)
        - Hard-coding configurations (uses Settings)
    
    Design Philosophy:
        - Each component (chunker, embeddings, vector store) is injected
        - No hard-coded configurations or magic values
        - Clear separation between indexing and retrieval
        - Simple, testable interface
        - Graceful error handling
    
    Attributes:
        text_chunker (TextChunker): Component for splitting documents
        embedding_manager (EmbeddingManager): Component for managing embeddings
        vector_store_manager (VectorStoreManager): Component for vector storage
    
    Example:
        ```python
        # Initialize components
        text_chunker = TextChunker(get_settings())
        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.get_embeddings()
        vector_store_manager = VectorStoreManager(get_settings(), embeddings)
        
        # Create indexer
        indexer = DocumentIndexer(
            text_chunker=text_chunker,
            embedding_manager=embedding_manager,
            vector_store_manager=vector_store_manager
        )
        
        # Index documents
        result = indexer.index_documents(documents)
        print(f"Success! Indexed {result['num_chunks']} chunks")
        ```
    """
    
    def __init__(
        self,
        text_chunker: TextChunker,
        embedding_manager: EmbeddingManager,
        vector_store_manager: VectorStoreManager
    ):
        """
        Initialize the document indexer with required components.
        
        Args:
            text_chunker: Configured TextChunker for document splitting
            embedding_manager: EmbeddingManager for embedding generation
            vector_store_manager: VectorStoreManager for vector storage
            
        Note:
            All components should be pre-configured before passing to the indexer.
            The indexer does not modify component configurations.
        """
        self.text_chunker = text_chunker
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager
    
    def index_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Index a list of documents into the vector store.
        
        This method orchestrates the complete indexing pipeline:
        1. Validates input documents
        2. Splits documents into chunks using TextChunker
        3. Creates or updates vector store using VectorStoreManager
           (embeddings are generated internally by the vector store)
        4. Returns indexing metadata
        
        Pipeline Steps:
            Raw Documents → Validate → Chunk → Create Vector Store → Metadata
        
        Args:
            documents: List of LangChain Document objects to index
            **kwargs: Additional arguments to pass to vector store creation
                     (e.g., persist_directory, collection_name)
        
        Returns:
            Dict containing indexing metadata:
                - num_documents: Number of input documents
                - num_chunks: Number of chunks created
                - vector_store_type: Type of vector store used
                - success: Whether indexing succeeded
                - message: Status message
                
        Raises:
            ValueError: If documents list is empty or contains invalid documents
            RuntimeError: If indexing pipeline fails
            
        Example:
            ```python
            documents = pdf_loader.load_documents()
            
            result = indexer.index_documents(documents)
            
            if result['success']:
                print(f"✓ Indexed {result['num_chunks']} chunks")
                print(f"  Vector store: {result['vector_store_type']}")
            else:
                print(f"✗ Indexing failed: {result['message']}")
            ```
        """
        # Step 1: Validate input documents
        validation_result = self._validate_documents(documents)
        if not validation_result['valid']:
            return {
                'success': False,
                'num_documents': len(documents),
                'num_chunks': 0,
                'vector_store_type': None,
                'message': validation_result['message']
            }
        
        try:
            # Step 2: Split documents into chunks
            print(f"📄 Splitting {len(documents)} documents into chunks...")
            chunks = self.text_chunker.split_documents(documents)
            print(f"✓ Created {len(chunks)} chunks")
            
            if not chunks:
                return {
                    'success': False,
                    'num_documents': len(documents),
                    'num_chunks': 0,
                    'vector_store_type': None,
                    'message': 'Text chunking produced zero chunks'
                }
            
            # Step 3: Create vector store with chunks
            # Note: The VectorStoreManager handles embedding generation internally
            print(f"🔢 Creating vector store and generating embeddings...")
            vector_store = self.vector_store_manager.create_vector_store(
                documents=chunks,
                **kwargs
            )
            print(f"✓ Vector store created successfully")
            
            # Step 4: Save vector store (if applicable)
            # FAISS requires explicit save, Chroma/Pinecone auto-persist
            print(f"💾 Saving vector store...")
            self.vector_store_manager.save_vector_store(vector_store)
            print(f"✓ Vector store saved")
            
            # Step 5: Return success metadata
            return {
                'success': True,
                'num_documents': len(documents),
                'num_chunks': len(chunks),
                'vector_store_type': self.vector_store_manager.vector_store_type,
                'message': f'Successfully indexed {len(chunks)} chunks from {len(documents)} documents'
            }
            
        except Exception as e:
            # Handle any errors in the indexing pipeline
            return {
                'success': False,
                'num_documents': len(documents),
                'num_chunks': 0,
                'vector_store_type': None,
                'message': f'Indexing failed: {str(e)}'
            }
    
    def _validate_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Validate input documents before indexing.
        
        Checks:
            - Documents list is not empty
            - All items are LangChain Document objects
            - Documents have non-empty page_content
        
        Args:
            documents: List of documents to validate
            
        Returns:
            Dict with validation results:
                - valid (bool): Whether documents are valid
                - message (str): Validation message
        """
        # Check if list is empty
        if not documents:
            return {
                'valid': False,
                'message': 'Document list is empty. Please provide at least one document.'
            }
        
        # Check if all items are Document objects
        if not all(isinstance(doc, Document) for doc in documents):
            return {
                'valid': False,
                'message': 'All items must be LangChain Document objects.'
            }
        
        # Check if documents have content
        empty_docs = [
            i for i, doc in enumerate(documents)
            if not doc.page_content or not doc.page_content.strip()
        ]
        
        if empty_docs:
            return {
                'valid': False,
                'message': f'Found {len(empty_docs)} documents with empty content at indices: {empty_docs[:5]}'
            }
        
        # All validations passed
        return {
            'valid': True,
            'message': 'All documents are valid'
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_indexer(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    vector_store_type: Optional[str] = None
) -> DocumentIndexer:
    """
    Convenience function to create a fully configured DocumentIndexer.
    
    This function handles all the setup:
    1. Loads Settings (with optional overrides)
    2. Initializes TextChunker
    3. Initializes EmbeddingManager
    4. Initializes VectorStoreManager
    5. Returns ready-to-use DocumentIndexer
    
    Args:
        chunk_size: Override default chunk size (optional)
        chunk_overlap: Override default chunk overlap (optional)
        vector_store_type: Override default vector store type (optional)
        
    Returns:
        Fully configured DocumentIndexer instance
        
    Example:
        ```python
        # Use default settings from .env
        indexer = create_indexer()
        
        # Override specific settings
        indexer = create_indexer(
            chunk_size=500,
            chunk_overlap=50,
            vector_store_type="chroma"
        )
        
        # Use the indexer
        result = indexer.index_documents(documents)
        ```
    """
    # Load settings
    settings = get_settings()
    
    # Apply overrides if provided
    if chunk_size is not None:
        settings.chunk_size = chunk_size
    if chunk_overlap is not None:
        settings.chunk_overlap = chunk_overlap
    if vector_store_type is not None:
        settings.vector_store_type = vector_store_type
    
    # Initialize components
    text_chunker = TextChunker(settings)
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.get_embeddings()
    vector_store_manager = VectorStoreManager(settings, embeddings)
    
    # Create and return indexer
    return DocumentIndexer(
        text_chunker=text_chunker,
        embedding_manager=embedding_manager,
        vector_store_manager=vector_store_manager
    )


# ============================================================================
# Quick Usage Example
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of the DocumentIndexer.
    
    This example shows a complete indexing pipeline from loading documents
    to creating a searchable vector store.
    """
    print("=" * 80)
    print("Document Indexer - Demo")
    print("=" * 80)
    print()
    
    # For demonstration, we'll create dummy documents
    # In real usage, load from PDFLoader, WebLoader, etc.
    from langchain_core.documents import Document
    
    sample_documents = [
        Document(
            page_content="RAG (Retrieval-Augmented Generation) combines retrieval and generation.",
            metadata={"source": "rag_intro.txt", "page": 1}
        ),
        Document(
            page_content="Vector databases store embeddings for semantic search.",
            metadata={"source": "vector_db.txt", "page": 1}
        ),
        Document(
            page_content="LangChain provides tools for building LLM applications.",
            metadata={"source": "langchain.txt", "page": 1}
        ),
    ]
    
    print(f"Sample documents: {len(sample_documents)}")
    print()
    
    try:
        # Create indexer using convenience function
        print("🔧 Creating indexer with default settings...")
        indexer = create_indexer()
        print("✓ Indexer created")
        print()
        
        # Index the documents
        print("📊 Indexing documents...")
        result = indexer.index_documents(sample_documents)
        print()
        
        # Display results
        if result['success']:
            print("✓ Indexing Successful!")
            print("-" * 80)
            print(f"  Documents processed: {result['num_documents']}")
            print(f"  Chunks created:      {result['num_chunks']}")
            print(f"  Vector store type:   {result['vector_store_type']}")
            print(f"  Message:             {result['message']}")
        else:
            print("✗ Indexing Failed")
            print("-" * 80)
            print(f"  Error: {result['message']}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        print()
        print("Note: This demo requires:")
        print("  1. .env file with configurations")
        print("  2. Required packages (langchain, sentence-transformers, etc.)")
        print("  3. Proper Settings configuration")
