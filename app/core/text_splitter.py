"""
Text Chunking Engine for RAG Systems

This module provides a clean, reusable text splitting layer that converts
raw LangChain Document objects into smaller, retrievable chunks optimized
for semantic search and retrieval.

Purpose:
    - Centralizes all document chunking logic for the RAG system
    - Reads chunking parameters from centralized configuration
    - Preserves document metadata across chunks
    - Provides a simple, testable interface for text preprocessing

Why This Matters:
    - Proper chunking is critical for RAG performance
    - Too large = poor retrieval precision, too small = lost context
    - Overlap prevents information loss at chunk boundaries
    - Consistent chunking strategy across all document types

Design Pattern:
    - Strategy Pattern: Encapsulates chunking algorithm
    - Dependency Injection: Accepts Settings for configuration
    - Single Responsibility: Only handles text splitting

Usage Example:
    ```python
    from app.core.text_splitter import TextChunker
    from app.utils.config import get_settings
    from app.ingestion.pdf_loader import PDFDocumentLoader
    
    # Load documents
    pdf_loader = PDFDocumentLoader("./documents")
    documents = pdf_loader.load_documents()
    
    # Split into chunks
    settings = get_settings()
    chunker = TextChunker(settings)
    chunks = chunker.split_documents(documents)
    
    print(f"Original: {len(documents)} documents")
    print(f"After chunking: {len(chunks)} chunks")
    ```

Architecture:
    Document Loading → Text Chunking → Embedding → Vector Storage → Retrieval
                        ^^^^^^^^^^^^
                        (This Module)
"""

from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Import our centralized configuration
from app.utils.config import Settings


class TextChunker:
    """
    Text chunking engine for RAG document preprocessing.
    
    This class wraps LangChain's RecursiveCharacterTextSplitter and provides
    a clean, configuration-driven interface for splitting documents into
    retrievable chunks.
    
    Features:
        - Configuration-driven chunk sizing (from Settings)
        - Metadata preservation across chunks
        - Chunk-level metadata (chunk_index) for tracking
        - Input validation (empty lists, invalid parameters)
        - Graceful error handling
    
    The RecursiveCharacterTextSplitter works by:
        1. Trying to split on paragraphs (\\n\\n)
        2. If chunks still too large, split on sentences (\\n)
        3. If still too large, split on words (spaces)
        4. As last resort, split on characters
    
    This hierarchy preserves semantic coherence as much as possible.
    
    Attributes:
        settings (Settings): Configuration object with chunk_size and chunk_overlap
        splitter (RecursiveCharacterTextSplitter): Underlying LangChain splitter
        chunk_size (int): Maximum characters per chunk
        chunk_overlap (int): Overlap between consecutive chunks
    
    Example:
        ```python
        from app.core.text_splitter import TextChunker
        from app.utils.config import get_settings
        
        settings = get_settings()
        chunker = TextChunker(settings)
        
        # Split documents
        chunks = chunker.split_documents(documents)
        
        # Inspect chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i}: {len(chunk.page_content)} chars")
            print(f"Metadata: {chunk.metadata}")
        ```
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the text chunker with configuration from Settings.
        
        Args:
            settings: Settings object containing chunk_size and chunk_overlap
        
        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap < 0
            ValueError: If chunk_overlap >= chunk_size
        
        Example:
            ```python
            from app.utils.config import get_settings
            
            settings = get_settings()
            chunker = TextChunker(settings)
            ```
        """
        self.settings = settings
        
        # Read chunking parameters from configuration
        self.chunk_size: int = settings.chunk_size
        self.chunk_overlap: int = settings.chunk_overlap
        
        # Validate chunking parameters
        self._validate_parameters()
        
        # Initialize LangChain's RecursiveCharacterTextSplitter
        # This splitter tries to split on semantic boundaries in this order:
        #   1. Paragraphs (\n\n)
        #   2. Sentences (\n)
        #   3. Words (spaces)
        #   4. Characters (fallback)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,  # Count characters, not tokens
            separators=["\n\n", "\n", ". ", " ", ""],  # Semantic boundaries
            is_separator_regex=False
        )
    
    def split_documents(
        self,
        documents: List[Document],
        add_chunk_metadata: bool = True
    ) -> List[Document]:
        """
        Split a list of LangChain Document objects into smaller chunks.
        
        This is the main public interface for text chunking. It:
            1. Validates input documents
            2. Splits each document using RecursiveCharacterTextSplitter
            3. Preserves original metadata in all chunks
            4. Optionally adds chunk-level metadata (chunk_index)
        
        Args:
            documents: List of LangChain Document objects to split.
                      Each document should have page_content (str) and
                      metadata (dict) attributes.
            
            add_chunk_metadata: If True, adds chunk_index to metadata.
                               Useful for tracking chunk order and debugging.
                               Default: True
        
        Returns:
            List of chunked Document objects. Each chunk:
                - Has page_content ≤ chunk_size characters
                - Retains all original document metadata
                - Has optional chunk_index in metadata
        
        Raises:
            ValueError: If documents list is None
            ValueError: If any document lacks page_content attribute
        
        Example:
            ```python
            # Load documents
            from app.ingestion.pdf_loader import PDFDocumentLoader
            pdf_loader = PDFDocumentLoader("./docs")
            documents = pdf_loader.load_documents()
            
            # Split documents
            chunker = TextChunker(settings)
            chunks = chunker.split_documents(documents)
            
            print(f"Original: {len(documents)} documents")
            print(f"Chunks: {len(chunks)} chunks")
            
            # Inspect first chunk
            first_chunk = chunks[0]
            print(f"Content length: {len(first_chunk.page_content)}")
            print(f"Metadata: {first_chunk.metadata}")
            print(f"Chunk index: {first_chunk.metadata.get('chunk_index')}")
            ```
        
        Performance:
            - Linear time complexity: O(n * m) where:
                n = number of documents
                m = average document length / chunk_size
            - Memory: Proportional to total text size
            - Typical throughput: 1-10 MB/sec depending on text structure
        """
        # Validate input
        if documents is None:
            raise ValueError("documents cannot be None")
        
        # Handle empty list gracefully (not an error, just return empty)
        if len(documents) == 0:
            return []
        
        # Validate that each document has page_content
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'page_content'):
                raise ValueError(
                    f"Document at index {i} is missing 'page_content' attribute. "
                    f"Expected LangChain Document object."
                )
        
        # Split documents using LangChain's splitter
        # This automatically preserves metadata from original documents
        chunks = self.splitter.split_documents(documents)
        
        # Add chunk-level metadata if requested
        if add_chunk_metadata:
            chunks = self._add_chunk_metadata(chunks)
        
        return chunks
    
    def _add_chunk_metadata(self, chunks: List[Document]) -> List[Document]:
        """
        Add chunk-level metadata to each chunk.
        
        This internal method enriches chunks with additional metadata:
            - chunk_index: Sequential index of chunk (0, 1, 2, ...)
            - total_chunks: Total number of chunks in this batch
        
        This metadata is useful for:
            - Debugging: Identify which chunk has issues
            - Tracking: Know chunk order during retrieval
            - Analytics: Understand chunk distribution
        
        Args:
            chunks: List of Document objects to enrich
        
        Returns:
            Same list of Document objects with added metadata
        
        Note:
            This modifies the metadata dict in-place for each chunk.
            Original metadata is preserved.
        """
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            # Add chunk-specific metadata
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = total_chunks
            
            # Calculate and add chunk size for analytics
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        return chunks
    
    def _validate_parameters(self) -> None:
        """
        Validate chunking parameters to prevent configuration errors.
        
        Checks:
            1. chunk_size must be positive (> 0)
            2. chunk_overlap must be non-negative (>= 0)
            3. chunk_overlap must be less than chunk_size
        
        Raises:
            ValueError: If any validation check fails
        
        Why This Matters:
            - chunk_size <= 0: No chunking would occur
            - chunk_overlap < 0: Nonsensical negative overlap
            - chunk_overlap >= chunk_size: Every chunk would be duplicated
        """
        if self.chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be positive, got {self.chunk_size}. "
                f"Check CHUNK_SIZE in your .env configuration."
            )
        
        if self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be non-negative, got {self.chunk_overlap}. "
                f"Check CHUNK_OVERLAP in your .env configuration."
            )
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size}). "
                f"Typical overlap is 10-20% of chunk_size. "
                f"Check CHUNK_SIZE and CHUNK_OVERLAP in your .env configuration."
            )
    
    def get_chunk_count_estimate(self, documents: List[Document]) -> int:
        """
        Estimate the number of chunks that would be created from documents.
        
        This is a rough estimate based on document sizes and does not account
        for separator positions. Actual chunk count may vary slightly.
        
        Args:
            documents: List of Document objects
        
        Returns:
            Estimated number of chunks (integer)
        
        Formula:
            For each document:
                chunks ≈ ceil(text_length / (chunk_size - chunk_overlap))
        
        Example:
            ```python
            chunker = TextChunker(settings)
            
            estimate = chunker.get_chunk_count_estimate(documents)
            print(f"Estimated chunks: {estimate}")
            
            actual_chunks = chunker.split_documents(documents)
            print(f"Actual chunks: {len(actual_chunks)}")
            ```
        
        Use Case:
            - Pre-flight check before chunking large document sets
            - Estimate embedding/storage costs
            - Progress bar initialization
        """
        if not documents:
            return 0
        
        total_length = sum(len(doc.page_content) for doc in documents)
        
        # Estimate chunks per document
        # Use (chunk_size - chunk_overlap) as effective chunk size
        effective_chunk_size = max(1, self.chunk_size - self.chunk_overlap)
        estimated_chunks = (total_length + effective_chunk_size - 1) // effective_chunk_size
        
        return estimated_chunks
    
    # TODO: Token-Based Splitting
    # ---------------------------
    # Currently we split by character count, but LLM context windows use tokens.
    # Token-based splitting would be more accurate for:
    #   - Controlling LLM input size precisely
    #   - Optimizing context window usage
    #   - Handling different languages (tokens ≠ characters)
    #
    # Implementation approach:
    #   1. Use tiktoken library for OpenAI models
    #   2. Use model-specific tokenizers for other providers
    #   3. Fall back to character count if tokenizer unavailable
    #
    # Example:
    #   from langchain.text_splitter import TokenTextSplitter
    #   import tiktoken
    #
    #   def split_by_tokens(
    #       self,
    #       documents: List[Document],
    #       encoding_name: str = "cl100k_base"  # GPT-3.5/4 encoding
    #   ) -> List[Document]:
    #       """Split documents by token count instead of characters."""
    #       tokenizer = tiktoken.get_encoding(encoding_name)
    #       
    #       splitter = TokenTextSplitter(
    #           chunk_size=self.chunk_size,  # Now in tokens
    #           chunk_overlap=self.chunk_overlap,
    #           encoding_name=encoding_name
    #       )
    #       
    #       return splitter.split_documents(documents)
    #
    # Dependencies: pip install tiktoken
    
    # TODO: Adaptive Chunk Sizes
    # --------------------------
    # Different document types may benefit from different chunk sizes:
    #   - Code: Smaller chunks (function-level) with syntax-aware splitting
    #   - Academic papers: Larger chunks (paragraph-level) for context
    #   - Chat logs: Very small chunks (message-level)
    #
    # Implementation approach:
    #   1. Detect document type from metadata (loader_type, file_extension)
    #   2. Apply type-specific chunking strategy
    #   3. Use appropriate separators for each type
    #
    # Example:
    #   def split_adaptive(
    #       self,
    #       documents: List[Document]
    #   ) -> List[Document]:
    #       """Split with adaptive chunk sizes based on document type."""
    #       chunks = []
    #       
    #       for doc in documents:
    #           doc_type = doc.metadata.get('loader_type', 'generic')
    #           
    #           if doc_type == 'code':
    #               # Smaller chunks for code
    #               splitter = RecursiveCharacterTextSplitter(
    #                   chunk_size=500,
    #                   chunk_overlap=50,
    #                   separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n"]
    #               )
    #           elif doc_type == 'pdf':
    #               # Larger chunks for papers/documents
    #               splitter = RecursiveCharacterTextSplitter(
    #                   chunk_size=1500,
    #                   chunk_overlap=300
    #               )
    #           else:
    #               # Default chunking
    #               splitter = self.splitter
    #           
    #           doc_chunks = splitter.split_documents([doc])
    #           chunks.extend(doc_chunks)
    #       
    #       return chunks
    
    # TODO: Overlap Optimization Experiments
    # --------------------------------------
    # Current overlap is fixed, but optimal overlap may depend on:
    #   - Document type and structure
    #   - Query patterns (keyword vs semantic)
    #   - Retrieval accuracy vs. cost tradeoffs
    #
    # Research questions to explore:
    #   1. Does higher overlap improve retrieval quality?
    #   2. What's the sweet spot for overlap % (10%, 20%, 30%)?
    #   3. Should overlap vary by chunk size?
    #   4. Can we predict optimal overlap from document analysis?
    #
    # Experiment framework:
    #   def experiment_overlaps(
    #       self,
    #       documents: List[Document],
    #       overlap_ratios: List[float] = [0.1, 0.15, 0.2, 0.25, 0.3]
    #   ) -> Dict[float, List[Document]]:
    #       """Generate chunks with different overlap ratios for A/B testing."""
    #       results = {}
    #       
    #       for ratio in overlap_ratios:
    #           overlap = int(self.chunk_size * ratio)
    #           
    #           splitter = RecursiveCharacterTextSplitter(
    #               chunk_size=self.chunk_size,
    #               chunk_overlap=overlap
    #           )
    #           
    #           chunks = splitter.split_documents(documents)
    #           results[ratio] = chunks
    #       
    #       return results
    #
    # Metrics to measure:
    #   - Retrieval accuracy (with labeled test set)
    #   - Number of chunks generated (cost)
    #   - Query latency (more chunks = slower)
    #   - Context coherence (qualitative evaluation)
    
    # TODO: Semantic Chunking
    # -----------------------
    # Instead of fixed-size chunks, split at semantic boundaries:
    #   - Topic changes (using embeddings to detect shifts)
    #   - Section headers (H1, H2, etc.)
    #   - Paragraph breaks with context awareness
    #
    # Benefits:
    #   - More coherent chunks (don't split mid-concept)
    #   - Better retrieval (chunks = semantic units)
    #   - Natural boundaries preserve meaning
    #
    # Implementation:
    #   from langchain.text_splitter import SemanticChunker
    #   
    #   def split_semantic(
    #       self,
    #       documents: List[Document],
    #       embeddings  # Embedding model for similarity
    #   ) -> List[Document]:
    #       """Split at semantic boundaries using embeddings."""
    #       splitter = SemanticChunker(
    #           embeddings=embeddings,
    #           breakpoint_threshold_type="percentile",  # or "standard_deviation"
    #           breakpoint_threshold_amount=95
    #       )
    #       
    #       return splitter.split_documents(documents)
    #
    # Tradeoffs:
    #   + Better semantic coherence
    #   + Improved retrieval quality
    #   - Slower (requires embedding computation)
    #   - Variable chunk sizes (may exceed limits)
    #
    # Dependencies: Requires embedding model (OpenAI, HuggingFace, etc.)
    
    # TODO: Chunk Quality Metrics
    # ---------------------------
    # Add methods to evaluate chunk quality:
    #   - Average chunk size and variance
    #   - Chunk size distribution histogram
    #   - Metadata completeness check
    #   - Overlap effectiveness analysis
    #
    # Example:
    #   def analyze_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
    #       """Analyze chunk quality metrics."""
    #       sizes = [len(chunk.page_content) for chunk in chunks]
    #       
    #       return {
    #           "total_chunks": len(chunks),
    #           "avg_size": sum(sizes) / len(sizes),
    #           "min_size": min(sizes),
    #           "max_size": max(sizes),
    #           "size_variance": statistics.variance(sizes),
    #           "metadata_complete": all(
    #               'chunk_index' in c.metadata for c in chunks
    #           )
    #       }
