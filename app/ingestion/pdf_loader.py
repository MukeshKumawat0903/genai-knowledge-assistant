"""
PDF Document Loader

This module provides a clean, production-ready PDF loader that extends
the BaseDocumentLoader interface.

Purpose:
    - Load single PDF files or entire directories of PDFs
    - Use LangChain's battle-tested PDF loaders (PyPDFLoader, PyPDFDirectoryLoader)
    - Add standardized metadata to each document
    - Handle common edge cases (missing files, no PDFs found)

Why This Matters:
    - PDFs are the most common document format in enterprise settings
    - Clean abstraction allows easy switching between PDF parsing libraries
    - Standardized metadata enables better retrieval and tracking
    - Directory loading enables batch processing of document collections

Usage Example:
    ```python
    from app.ingestion.pdf_loader import PDFDocumentLoader
    
    # Load a single PDF
    loader = PDFDocumentLoader(
        path="research_papers/attention_is_all_you_need.pdf",
        source_name="Transformer Paper"
    )
    docs = loader.load_documents()
    
    # Load all PDFs from a directory
    loader = PDFDocumentLoader(
        path="research_papers/",
        source_name="AI Research Collection"
    )
    docs = loader.load_documents()  # Returns all PDFs in directory
    ```

Design Pattern:
    - Extends BaseDocumentLoader (Template Method pattern)
    - Uses LangChain loaders (Adapter pattern)
    - Configuration-driven behavior (Strategy pattern)
"""

from typing import List
from pathlib import Path

# LangChain document loaders for PDF processing
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_core.documents import Document

# Import our base loader interface
from .base_loader import BaseDocumentLoader


class PDFDocumentLoader(BaseDocumentLoader):
    """
    PDF Document Loader for single files or directories.
    
    This loader extends BaseDocumentLoader to provide PDF-specific
    functionality using LangChain's PyPDFLoader and PyPDFDirectoryLoader.
    
    Features:
        - Single file loading with PyPDFLoader
        - Directory batch loading with PyPDFDirectoryLoader
        - Automatic metadata enrichment (source, file_name, loader_type)
        - Path validation with clear error messages
        - Page-level document splitting (one Document per page)
    
    Attributes:
        path (Path): File or directory path to load PDFs from
        source_name (str): Optional human-readable source name for metadata
    
    Why PyPDF?
        - PyPDFLoader is LangChain's default PDF loader
        - Good balance of speed, accuracy, and simplicity
        - Handles most common PDFs without external dependencies
        - Returns one Document per page (good for retrieval)
    """
    
    def __init__(self, path: str, source_name: str = None):
        """
        Initialize the PDF document loader.
        
        Args:
            path: Path to a PDF file or directory containing PDFs.
                  Can be absolute or relative.
                  Examples:
                    - "documents/report.pdf" (single file)
                    - "documents/research_papers/" (directory)
            
            source_name: Optional human-readable name for the source.
                        Used in metadata for better tracking.
                        If not provided, uses the file/directory name.
                        Examples:
                          - "Annual Report 2024"
                          - "AI Research Collection"
        
        Raises:
            ValueError: If path is None or empty string
        
        Example:
            >>> loader = PDFDocumentLoader("data/report.pdf", "Q4 Report")
            >>> loader = PDFDocumentLoader("data/pdfs/")  # Load all PDFs
        """
        # Validate path is provided
        if not path:
            raise ValueError("Path cannot be None or empty")
        
        # Convert to Path object for easier manipulation
        self.path = Path(path)
        
        # Use provided source_name or default to file/directory name
        self.source_name = source_name or self.path.name
    
    def load_documents(self) -> List[Document]:
        """
        Load PDF document(s) into LangChain Document objects.
        
        This method implements the required interface from BaseDocumentLoader.
        It automatically detects whether the path is a file or directory and
        uses the appropriate LangChain loader.
        
        Behavior:
            - If path is a file: Uses PyPDFLoader (returns one Document per page)
            - If path is a directory: Uses PyPDFDirectoryLoader (loads all PDFs)
            - Adds standardized metadata to each document
        
        Returns:
            List[Document]: List of LangChain Document objects.
                           Each document represents one page from a PDF.
                           
        Raises:
            FileNotFoundError: If the path does not exist
            ValueError: If directory contains no PDF files
            
        Metadata Added:
            - source: Full path to the PDF file
            - file_name: Name of the PDF file (e.g., "report.pdf")
            - loader_type: Always "pdf"
            - page: Page number within the PDF (0-indexed)
            
        Example:
            >>> loader = PDFDocumentLoader("data/report.pdf")
            >>> docs = loader.load_documents()
            >>> print(len(docs))  # Number of pages
            >>> print(docs[0].metadata)
            {'source': 'data/report.pdf', 'file_name': 'report.pdf', 
             'loader_type': 'pdf', 'page': 0}
        
        Notes:
            - One Document object is created per page
            - This is beneficial for RAG as it allows page-level retrieval
            - Text chunking should be done separately if needed
        """
        # Step 1: Validate the path exists / is usable
        self._validate_source(str(self.path))
        
        # Step 2: Determine if we're loading a file or directory
        if self.path.is_file():
            # Load a single PDF file
            documents = self._load_single_pdf()
        elif self.path.is_dir():
            # Load all PDFs from directory
            documents = self._load_directory()
        else:
            # Path exists but is neither file nor directory (e.g., symlink)
            raise ValueError(
                f"Path must be a file or directory, got: {self.path}"
            )
        
        # Step 3: Add standardized metadata to all documents
        # BaseDocumentLoader._add_metadata() RETURNS a merged metadata dict, so we must assign it.
        for doc in documents:
            pdf_metadata = dict(doc.metadata or {})
            source_value = pdf_metadata.get("source") or str(self.path)
            pdf_metadata["source"] = source_value
            pdf_metadata["file_name"] = Path(str(source_value)).name
            # Tests expect a stable, lowercase loader_type value for PDFs
            pdf_metadata["loader_type"] = "pdf"
            doc.metadata = self._add_metadata(pdf_metadata)
        
        return documents

    def _validate_source(self, source: str) -> None:
        """Validate a PDF file path or directory path.

        Raises:
            ValueError: for empty/invalid sources or non-PDF files
            FileNotFoundError: if the path does not exist
            ValueError: if a directory contains no PDFs
        """
        if not source or not str(source).strip():
            raise ValueError("Source path cannot be empty")

        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {source}")

        if path.is_file():
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"File must have .pdf extension, got: {path.suffix}")
            return

        if path.is_dir():
            has_pdf = any(path.glob("*.pdf"))
            if not has_pdf:
                raise ValueError(
                    f"No PDF files found in directory: {path}\n"
                    f"Ensure the directory contains .pdf files."
                )
            return

        raise ValueError(f"Path must be a file or directory, got: {path}")
    
    def _load_single_pdf(self) -> List[Document]:
        """
        Load a single PDF file using PyPDFLoader.
        
        This is an internal helper method for loading one PDF file.
        
        Returns:
            List[Document]: One Document per page
            
        Raises:
            ValueError: If file is not a PDF
            
        Implementation Notes:
            - Uses LangChain's PyPDFLoader
            - PyPDFLoader automatically splits by page
            - Each page becomes one Document object
            - Metadata includes: source (file path), page (page number)
        """
        # Validate file extension
        if self.path.suffix.lower() != ".pdf":
            raise ValueError(
                f"File must have .pdf extension, got: {self.path.suffix}"
            )
        
        # Use LangChain's PyPDFLoader
        # PyPDFLoader loads and splits PDF into one Document per page
        loader = PyPDFLoader(str(self.path))
        documents = loader.load()
        
        return documents
    
    def _load_directory(self) -> List[Document]:
        """
        Load all PDF files from a directory using PyPDFDirectoryLoader.
        
        This is an internal helper method for batch loading PDFs from a folder.
        
        Returns:
            List[Document]: Documents from all PDFs (one Document per page)
            
        Raises:
            ValueError: If directory contains no PDF files
            
        Implementation Notes:
            - Uses LangChain's PyPDFDirectoryLoader
            - Recursively finds all .pdf files in directory
            - Each page from each PDF becomes one Document
            - Metadata includes: source (file path), page (page number)
            
        TODO:
            - Add recursive parameter to control subdirectory traversal
            - Add glob_pattern parameter for more flexible file matching
            - Add max_files parameter to limit number of PDFs loaded
        """
        # Use LangChain's PyPDFDirectoryLoader
        # It automatically finds and loads all PDFs in the directory
        loader = PyPDFDirectoryLoader(str(self.path))
        documents = loader.load()
        
        # Check if any PDFs were found
        if not documents:
            raise ValueError(
                f"No PDF files found in directory: {self.path}\n"
                f"Ensure the directory contains .pdf files."
            )
        
        return documents
