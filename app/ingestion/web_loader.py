"""
Web Document Loader

This module provides a clean, production-ready web content loader that extends
the BaseDocumentLoader interface.

Purpose:
    - Load web pages from single URLs or URL lists
    - Use LangChain's battle-tested web loaders (WebBaseLoader)
    - Add standardized metadata to each document
    - Handle common edge cases (invalid URLs, fetch failures)

Why This Matters:
    - Web content is a critical knowledge source for RAG systems
    - Clean abstraction allows easy switching between web scraping libraries
    - Standardized metadata enables better retrieval and tracking
    - Batch URL loading enables efficient content ingestion

Usage Example:
    ```python
    from app.ingestion.web_loader import WebDocumentLoader
    
    # Load a single URL
    loader = WebDocumentLoader(
        urls="https://en.wikipedia.org/wiki/Artificial_intelligence",
        source_name="AI Wikipedia"
    )
    docs = loader.load_documents()
    
    # Load multiple URLs
    urls = [
        "https://blog.example.com/post1",
        "https://blog.example.com/post2",
        "https://blog.example.com/post3"
    ]
    loader = WebDocumentLoader(urls=urls, source_name="Tech Blog")
    docs = loader.load_documents()  # Returns one Document per URL
    ```

Design Pattern:
    - Extends BaseDocumentLoader (Template Method pattern)
    - Uses LangChain loaders (Adapter pattern)
    - Configuration-driven behavior (Strategy pattern)
"""

from typing import List, Union
from urllib.parse import urlparse

# LangChain document loaders for web content
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

# Import our base loader interface
from .base_loader import BaseDocumentLoader


class WebDocumentLoader(BaseDocumentLoader):
    """
    Web Document Loader for single URLs or URL lists.
    
    This loader extends BaseDocumentLoader to provide web-specific
    functionality using LangChain's WebBaseLoader.
    
    Features:
        - Single URL loading
        - Batch URL loading (multiple pages at once)
        - Automatic metadata enrichment (source, loader_type, ingestion_time)
        - URL validation with clear error messages
        - HTML parsing and text extraction
    
    Attributes:
        urls (List[str]): List of URLs to load content from
        source_name (str): Optional human-readable source name for metadata
    
    Why WebBaseLoader?
        - WebBaseLoader is LangChain's default web loader
        - Uses BeautifulSoup for HTML parsing
        - Handles most common web pages without JavaScript
        - Good balance of simplicity and functionality
        - Returns one Document per URL
    """
    
    def __init__(self, urls: Union[str, List[str]], source_name: str = None):
        """
        Initialize the web document loader.
        
        Args:
            urls: A single URL string or list of URL strings.
                  Must be valid HTTP/HTTPS URLs.
                  Examples:
                    - "https://example.com/article"
                    - ["https://blog.com/post1", "https://blog.com/post2"]
            
            source_name: Optional human-readable name for the source.
                        Used in metadata for better tracking.
                        If not provided, uses the domain name from first URL.
                        Examples:
                          - "Company Blog"
                          - "AI Research Articles"
        
        Raises:
            ValueError: If urls is None, empty, or contains invalid URLs
        
        Example:
            >>> loader = WebDocumentLoader("https://example.com")
            >>> loader = WebDocumentLoader(["https://a.com", "https://b.com"])
        """
        # Normalize input: convert single URL to list
        if isinstance(urls, str):
            self.urls = [urls]
        elif isinstance(urls, list):
            self.urls = urls
        else:
            raise ValueError(
                f"urls must be a string or list of strings, got: {type(urls)}"
            )
        
        # Validate URLs are provided
        if not self.urls:
            raise ValueError("URLs list cannot be empty")
        
        # Validate each URL format
        for url in self.urls:
            if not url or not isinstance(url, str):
                raise ValueError(f"Invalid URL: {url}")
            self._validate_url_format(url)
        
        # Use provided source_name or extract from first URL
        if source_name:
            self.source_name = source_name
        else:
            # Extract domain name from first URL as default
            parsed = urlparse(self.urls[0])
            self.source_name = parsed.netloc or "Web Content"
    
    def load_documents(self) -> List[Document]:
        """Load web pages and return one Document per URL with standardized metadata."""
        # Load web content using LangChain's WebBaseLoader
        documents = self._load_web_content()
        
        # Add standardized metadata to all documents
        for doc in documents:
            # Ensure source URL is in metadata
            if "source" not in doc.metadata:
                doc.metadata["source"] = doc.metadata.get("url", "unknown")
            
            # Add loader type (required by base interface)
            self._add_metadata(doc.metadata)
        
        return documents
    
    def _load_web_content(self) -> List[Document]:
        """
        Load web content using WebBaseLoader.
        
        This is an internal helper method for loading web pages.
        
        Returns:
            List[Document]: One Document per URL
            
        Raises:
            ValueError: If any URL fails to load
            
        Implementation Notes:
            - Uses LangChain's WebBaseLoader
            - WebBaseLoader uses BeautifulSoup4 for HTML parsing
            - Automatically extracts text content
            - Preserves page title in metadata
        """
        try:
            # WebBaseLoader can handle multiple URLs
            loader = WebBaseLoader(self.urls)
            documents = loader.load()
            
            return documents
            
        except Exception as e:
            # Provide clear error message with context
            raise ValueError(
                f"Failed to load web content from URLs: {self.urls}\n"
                f"Error: {str(e)}\n"
                f"Ensure URLs are accessible and contain valid HTML."
            )
    
    def _validate_url_format(self, url: str) -> None:
        """
        Validate that a URL has correct format.
        
        Args:
            url: URL string to validate
            
        Raises:
            ValueError: If URL format is invalid
            
        Implementation Notes:
            - Checks for http:// or https:// scheme
            - Validates URL has a network location (domain)
            - Does NOT check if URL is accessible (that happens during load)
        """
        try:
            parsed = urlparse(url)
            
            # Check for valid scheme
            if parsed.scheme not in ['http', 'https']:
                raise ValueError(
                    f"URL must start with http:// or https://, got: {url}"
                )
            
            # Check for valid network location (domain)
            if not parsed.netloc:
                raise ValueError(
                    f"URL must contain a valid domain, got: {url}"
                )
            
        except Exception as e:
            raise ValueError(f"Invalid URL format: {url}\nError: {str(e)}")
    
    # ========================================================================
    # TODO: Advanced Features (Future Enhancements)
    # ========================================================================
    
    # TODO: Rate Limiting and Politeness
    # -----------------------------------
    # When scraping multiple URLs, especially from the same domain,
    # it's important to implement rate limiting to avoid overwhelming
    # the server or getting blocked.
    #
    # Implementation approach:
    #   1. Add delay between requests to same domain
    #   2. Use exponential backoff for retries
    #   3. Respect robots.txt rules
    #   4. Add custom headers (User-Agent, etc.)
    #
    # Example:
    #   import time
    #   from collections import defaultdict
    #   from urllib.parse import urlparse
    #
    #   def load_with_rate_limiting(
    #       self,
    #       delay_seconds: float = 1.0
    
    # ========================================================================
    # Supported Future Enhancements (See GitHub Issues)
    # ========================================================================
    # - Rate limiting for multiple URLs
    # - JavaScript rendering (Selenium/Playwright)
    # - Website crawling with depth limits
    # - Content filtering by CSS selectors
    # - Retry logic with exponential backoff
    # - Partial success handling for multiple URLs
