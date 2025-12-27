"""
Abstract Base Document Loader

This module defines the interface contract that ALL document loaders must follow.
By enforcing a consistent interface, we ensure that:

1. All loaders return the same data structure (LangChain Documents)
2. New loaders can be added without modifying existing code
3. Loaders are interchangeable and testable
4. The ingestion layer remains clean and maintainable

Architecture Pattern:
--------------------
This implements the Template Method and Strategy patterns:
- BaseDocumentLoader defines the interface (Template Method)
- Concrete loaders implement their specific strategy (Strategy)
- All loaders follow the same contract

Why This Matters:
-----------------
In a RAG system, documents come from many sources (PDFs, web, YouTube, etc.).
Without a consistent interface, each loader would have different methods and
return different formats, making the system fragile and hard to maintain.

Usage Example:
--------------
    class PDFLoader(BaseDocumentLoader):
        def load_documents(self) -> List[Document]:
            # PDF-specific loading logic
            return documents
    
    loader = PDFLoader("path/to/file.pdf")
    documents = loader.load_documents()
    # documents are guaranteed to be List[Document] regardless of loader type
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document


class BaseDocumentLoader(ABC):
    """
    Abstract base class defining the interface for all document loaders.
    
    This class enforces a contract that ensures all loaders:
    - Return LangChain Document objects
    - Include standardized metadata
    - Validate their input sources
    - Provide consistent error handling
    
    Design Philosophy:
    ------------------
    - MINIMAL: Only essential methods in the interface
    - CLEAN: Clear separation of concerns
    - EXTENSIBLE: Easy to add new loader types
    - TESTABLE: Can be mocked and unit tested
    
    All concrete loaders MUST implement:
    - load_documents(): The core loading logic
    
    All concrete loaders CAN use:
    - _validate_source(): Check if source is valid
    - _add_metadata(): Standardize metadata fields
    
    Example Implementation:
    -----------------------
    class MyLoader(BaseDocumentLoader):
        def __init__(self, source_path: str):
            self.source = source_path
        
        def load_documents(self) -> List[Document]:
            # Validate input
            self._validate_source(self.source)
            
            # Load content (your logic here)
            content = self._load_content()
            
            # Create Document with metadata
            doc = Document(
                page_content=content,
                metadata=self._add_metadata({
                    "source": self.source,
                    "custom_field": "value"
                })
            )
            
            return [doc]
    """
    
    @abstractmethod
    def load_documents(self) -> List[Document]:
        """
        Load documents from the source and return LangChain Document objects.
        
        This is the ONLY required method that concrete loaders must implement.
        It defines what a "loader" fundamentally does: loads documents.
        
        Returns:
            List[Document]: List of LangChain Document objects, where each Document
                contains:
                - page_content (str): The actual text content
                - metadata (dict): Information about the document (source, date, etc.)
        
        Raises:
            ValueError: If source validation fails (use _validate_source())
            RuntimeError: If loading fails (network, parsing, etc.)
            
        Implementation Guidelines:
        -------------------------
        1. Validate your source using _validate_source()
        2. Load the content (PDF parsing, web scraping, API calls, etc.)
        3. Create Document objects with content and metadata
        4. Use _add_metadata() to standardize metadata fields
        5. Return list of Documents (even if single document)
        
        Example:
        --------
        def load_documents(self) -> List[Document]:
            # Step 1: Validate
            self._validate_source(self.source_path)
            
            # Step 2: Load content
            with open(self.source_path) as f:
                content = f.read()
            
            # Step 3: Create Document
            doc = Document(
                page_content=content,
                metadata=self._add_metadata({
                    "source": self.source_path,
                    "page_count": 1
                })
            )
            
            # Step 4: Return list
            return [doc]
        
        TODO for Future Enhancement:
        ----------------------------
        - Add async version: async def aload_documents()
        - Add streaming version: def load_documents_stream() -> Iterator[Document]
        - Add batch loading: def load_documents_batch(sources: List[str])
        """
        pass
    
    def _validate_source(self, source: Any) -> None:
        """
        Validate that the source is accessible and valid.
        
        This is an optional helper method that loaders can use to validate
        their input sources before attempting to load. It promotes fail-fast
        behavior and provides clear error messages early.
        
        Args:
            source: The source to validate (path, URL, ID, etc.)
                Type depends on loader implementation (str, Path, dict, etc.)
        
        Raises:
            ValueError: If source is None, empty, or invalid format
            FileNotFoundError: If source is a path that doesn't exist
            ConnectionError: If source is a URL that's not accessible
            
        Common Validation Patterns:
        ---------------------------
        
        For File Paths:
        >>> def _validate_source(self, source: str) -> None:
        >>>     if not source:
        >>>         raise ValueError("Source path cannot be empty")
        >>>     path = Path(source)
        >>>     if not path.exists():
        >>>         raise FileNotFoundError(f"File not found: {source}")
        >>>     if not path.is_file():
        >>>         raise ValueError(f"Not a file: {source}")
        
        For URLs:
        >>> def _validate_source(self, source: str) -> None:
        >>>     if not source:
        >>>         raise ValueError("URL cannot be empty")
        >>>     if not source.startswith(('http://', 'https://')):
        >>>         raise ValueError(f"Invalid URL: {source}")
        
        For IDs:
        >>> def _validate_source(self, source: str) -> None:
        >>>     if not source or not source.strip():
        >>>         raise ValueError("ID cannot be empty")
        
        Why This Matters:
        -----------------
        - Fail fast with clear error messages
        - Avoid wasting time on invalid inputs
        - Provide better debugging experience
        - Consistent error handling across loaders
        
        TODO for Future Enhancement:
        ----------------------------
        - Add validation for batch sources
        - Add network connectivity checks
        - Add file format validation (magic bytes)
        - Add size limits validation
        """
        # Base implementation does nothing - loaders override as needed
        pass
    
    def _add_metadata(
        self,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add standardized metadata fields to document metadata.
        
        This helper method ensures all documents have consistent metadata fields
        regardless of loader type. It adds standard fields and merges them with
        loader-specific custom metadata.
        
        Standard Fields Added:
        ----------------------
        - loader_type: Class name of the loader (e.g., "PDFLoader", "WebLoader")
        - ingestion_time: ISO timestamp when document was loaded
        - loader_version: Version of loader (if implemented in subclass)
        
        Args:
            custom_metadata: Loader-specific metadata to merge with standard fields.
                Examples:
                - PDF: {"source": "file.pdf", "page_number": 1, "author": "John"}
                - Web: {"source": "https://...", "title": "...", "publish_date": "..."}
                - YouTube: {"source": "video_id", "title": "...", "duration": 360}
        
        Returns:
            Dict[str, Any]: Combined metadata with standard + custom fields
        
        Example Usage:
        --------------
        >>> metadata = self._add_metadata({
        >>>     "source": "document.pdf",
        >>>     "page_number": 5,
        >>>     "author": "John Doe"
        >>> })
        >>> # Returns:
        >>> {
        >>>     "loader_type": "PDFLoader",
        >>>     "ingestion_time": "2025-12-21T10:30:00.123456",
        >>>     "source": "document.pdf",
        >>>     "page_number": 5,
        >>>     "author": "John Doe"
        >>> }
        
        Why This Matters:
        -----------------
        - Consistent metadata across all loaders
        - Easier to track document provenance
        - Enables filtering and debugging
        - Supports audit trails
        
        Best Practices:
        ---------------
        1. Always include "source" in custom_metadata
        2. Use standard keys when possible (author, title, date, etc.)
        3. Keep metadata lightweight (avoid large objects)
        4. Use ISO formats for dates and timestamps
        
        TODO for Future Enhancement:
        ----------------------------
        - Add schema validation for metadata
        - Add metadata compression for large values
        - Add metadata encryption for sensitive fields
        - Add metadata versioning
        """
        # Start with standard metadata
        metadata = {
            "loader_type": self.__class__.__name__,
            "ingestion_time": datetime.now().isoformat(),
        }
        
        # Merge with custom metadata (custom fields override if duplicate keys)
        if custom_metadata:
            metadata.update(custom_metadata)
        
        return metadata


# TODO: Add async loading support
#
# class AsyncBaseDocumentLoader(BaseDocumentLoader):
#     """
#     Async version of BaseDocumentLoader for concurrent loading.
#     
#     Useful for:
#     - Loading multiple documents concurrently
#     - Non-blocking I/O operations (web scraping, API calls)
#     - Large-scale ingestion pipelines
#     
#     Example:
#     --------
#     class AsyncWebLoader(AsyncBaseDocumentLoader):
#         async def load_documents(self) -> List[Document]:
#             async with aiohttp.ClientSession() as session:
#                 async with session.get(self.url) as response:
#                     content = await response.text()
#                     return [Document(page_content=content, metadata={...})]
#     
#     # Usage
#     loader = AsyncWebLoader("https://example.com")
#     documents = await loader.load_documents()
#     """
#     @abstractmethod
#     async def load_documents(self) -> List[Document]:
#         pass


# TODO: Add batch loading support
#
# class BatchDocumentLoader(BaseDocumentLoader):
#     """
#     Batch loader for processing multiple sources efficiently.
#     
#     Benefits:
#     - Amortize initialization costs
#     - Parallel processing
#     - Progress tracking
#     - Error handling for partial failures
#     
#     Example:
#     --------
#     class BatchPDFLoader(BatchDocumentLoader):
#         def load_documents_batch(
#             self,
#             sources: List[str],
#             show_progress: bool = True
#         ) -> List[Document]:
#             documents = []
#             for source in tqdm(sources, disable=not show_progress):
#                 try:
#                     docs = self._load_single(source)
#                     documents.extend(docs)
#                 except Exception as e:
#                     print(f"Failed to load {source}: {e}")
#                     continue
#             return documents
#     
#     # Usage
#     loader = BatchPDFLoader()
#     documents = loader.load_documents_batch(["file1.pdf", "file2.pdf", ...])
#     """
#     @abstractmethod
#     def load_documents_batch(
#         self,
#         sources: List[Any],
#         max_workers: int = 4
#     ) -> List[Document]:
#         pass


# TODO: Add error handling strategies
#
# Error Handling Best Practices for Loaders:
# ------------------------------------------
#
# 1. Validation Errors (ValueError):
#    - Invalid source format
#    - Missing required parameters
#    - Invalid configuration
#
# 2. File/Network Errors (FileNotFoundError, ConnectionError):
#    - Source doesn't exist
#    - Network timeout
#    - Permission denied
#
# 3. Parsing Errors (RuntimeError):
#    - Corrupted file
#    - Unsupported format
#    - Encoding issues
#
# Example Error Handling Pattern:
#
# def load_documents(self) -> List[Document]:
#     try:
#         # Validate
#         self._validate_source(self.source)
#         
#         # Load
#         content = self._load_content()
#         
#         # Parse
#         documents = self._parse_content(content)
#         
#         return documents
#         
#     except ValueError as e:
#         raise ValueError(f"Invalid source '{self.source}': {e}")
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Source not found: {self.source}")
#     except Exception as e:
#         raise RuntimeError(f"Failed to load {self.source}: {e}")
