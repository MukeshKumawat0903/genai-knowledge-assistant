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
import json
from datetime import datetime

# Import our core components
from app.core.text_splitter import TextChunker
from app.core.embeddings import EmbeddingManager
from app.core.vector_store import VectorStoreManager
from app.utils.config import get_settings


def _infer_source_type(source: str) -> str:
    """Guess a human-readable type label from a source string."""
    s = source.lower()
    if s.endswith(".pdf"):
        return "pdf"
    if "youtube.com" in s or "youtu.be" in s:
        return "youtube"
    if s.startswith("http://") or s.startswith("https://"):
        return "web"
    return "file"


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
            
            # Step 5: Update document registry
            self._update_registry(documents, len(chunks))

            # Step 6: Return success metadata
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
    
    # ------------------------------------------------------------------
    # Document Registry (metadata tracking for the management panel)
    # ------------------------------------------------------------------

    def _get_registry_path(self) -> Path:
        """Return the path to the JSON document registry file."""
        vs_path = self.vector_store_manager.settings.vector_store_path
        return Path(vs_path) / "document_registry.json"

    def _load_registry(self) -> List[Dict[str, Any]]:
        """Load the document registry from disk (empty list if not found)."""
        path = self._get_registry_path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_registry(self, registry: List[Dict[str, Any]]) -> None:
        """Persist the document registry to disk."""
        path = self._get_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")

    def _update_registry(self, documents: List[Document], num_chunks: int) -> None:
        """Add newly indexed documents to the registry."""
        registry = self._load_registry()

        # Group documents by their source metadata field
        sources: Dict[str, Dict[str, Any]] = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in sources:
                sources[source] = {
                    "source": source,
                    "source_type": doc.metadata.get("source_type", _infer_source_type(source)),
                    "num_pages": 0,
                    "num_chunks": 0,
                    "indexed_at": datetime.now().isoformat(),
                }
            sources[source]["num_pages"] += 1

        # Distribute chunks proportionally (simple estimate)
        if sources:
            per_source = max(1, num_chunks // len(sources))
            for entry in sources.values():
                entry["num_chunks"] = per_source

        # Merge: replace existing entry for same source, append new ones
        existing = {e["source"]: i for i, e in enumerate(registry)}
        for entry in sources.values():
            if entry["source"] in existing:
                registry[existing[entry["source"]]] = entry
            else:
                registry.append(entry)

        self._save_registry(registry)

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Return metadata for all indexed documents.

        Returns:
            List of dicts with keys: source, source_type, num_chunks, indexed_at
        """
        return self._load_registry()

    def delete_document(self, source: str) -> Dict[str, Any]:
        """
        Remove a document entry from the registry.

        For Chroma vector stores, also deletes the corresponding vectors.
        For FAISS, the vectors remain until the next full re-index (soft delete).

        Args:
            source: The 'source' metadata value identifying the document.

        Returns:
            Dict with 'success' bool and 'message' string.
        """
        registry = self._load_registry()
        original_len = len(registry)
        registry = [e for e in registry if e.get("source") != source]

        if len(registry) == original_len:
            return {"success": False, "message": f"Document '{source}' not found in registry."}

        self._save_registry(registry)

        # Attempt hard delete from Chroma (FAISS does not support per-document deletion)
        vs_type = getattr(self.vector_store_manager, "vector_store_type", "")
        chroma_deleted = False
        if vs_type == "chroma":
            try:
                vs = self.vector_store_manager.load_vector_store()
                collection = vs._collection  # Chroma internal
                collection.delete(where={"source": source})
                chroma_deleted = True
            except Exception as exc:
                return {
                    "success": True,
                    "message": (
                        f"Removed '{source}' from registry. "
                        f"Chroma vector deletion failed: {exc} — re-index to fully remove."
                    ),
                }

        msg = f"Removed '{source}' from registry."
        if vs_type == "faiss":
            msg += " FAISS index unchanged — re-index remaining documents to rebuild."
        elif chroma_deleted:
            msg += " Chroma vectors deleted."
        return {"success": True, "message": msg}

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
