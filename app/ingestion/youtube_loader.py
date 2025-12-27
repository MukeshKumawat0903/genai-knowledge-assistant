"""
YouTube Document Loader

This module provides a clean, production-ready YouTube transcript loader that extends
the BaseDocumentLoader interface.

Purpose:
    - Load YouTube video transcripts as text documents
    - Use LangChain's YoutubeLoader for reliable transcript fetching
    - Add standardized metadata to each document
    - Handle common edge cases (no transcript, invalid URL)

Why This Matters:
    - YouTube videos contain valuable knowledge (tutorials, lectures, interviews)
    - Transcripts make video content searchable in RAG systems
    - Clean abstraction allows easy video content ingestion
    - Standardized metadata enables better retrieval and tracking

Usage Example:
    ```python
    from app.ingestion.youtube_loader import YouTubeDocumentLoader
    
    # Load a single video
    loader = YouTubeDocumentLoader(
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        source_name="Product Tutorial"
    )
    docs = loader.load_documents()
    
    # Load with specific language
    loader = YouTubeDocumentLoader(
        video_url="https://youtu.be/dQw4w9WgXcQ",
        language="es",  # Spanish transcript
        source_name="Spanish Tutorial"
    )
    docs = loader.load_documents()
    ```

Design Pattern:
    - Extends BaseDocumentLoader (Template Method pattern)
    - Uses LangChain loaders (Adapter pattern)
    - Configuration-driven behavior (Strategy pattern)
"""

from typing import List, Optional
import re
from urllib.parse import urlparse, parse_qs

# LangChain document loaders for YouTube
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document

# Import our base loader interface
from .base_loader import BaseDocumentLoader


class YouTubeDocumentLoader(BaseDocumentLoader):
    """
    YouTube Document Loader for video transcripts.
    
    This loader extends BaseDocumentLoader to provide YouTube-specific
    functionality using LangChain's YoutubeLoader.
    
    Features:
        - Single video transcript loading
        - Language selection (if multiple transcripts available)
        - Automatic metadata enrichment (source, video_id, loader_type)
        - URL validation with clear error messages
        - Handles various YouTube URL formats
    
    Attributes:
        video_url (str): YouTube video URL
        language (str): Preferred transcript language (default: "en")
        source_name (str): Optional human-readable source name for metadata
        video_id (str): Extracted YouTube video ID
    
    Why YoutubeLoader?
        - YoutubeLoader is LangChain's default YouTube loader
        - Uses youtube-transcript-api for reliable transcript fetching
        - Handles both regular and auto-generated captions
        - Returns transcript as a single Document
    """
    
    def __init__(
        self,
        video_url: str,
        language: str = "en",
        source_name: str = None
    ):
        """
        Initialize the YouTube document loader.
        
        Args:
            video_url: YouTube video URL. Supports multiple formats:
                      - https://www.youtube.com/watch?v=VIDEO_ID
                      - https://youtu.be/VIDEO_ID
                      - https://www.youtube.com/embed/VIDEO_ID
                      - https://m.youtube.com/watch?v=VIDEO_ID
            
            language: Language code for transcript (default: "en").
                     Uses 2-letter ISO 639-1 codes.
                     Examples: "en" (English), "es" (Spanish), "fr" (French)
                     If specified language not available, will try to use
                     available transcript or fail with clear message.
            
            source_name: Optional human-readable name for the source.
                        Used in metadata for better tracking.
                        If not provided, uses video title (if available).
                        Examples:
                          - "Machine Learning Tutorial"
                          - "Product Demo Video"
        
        Raises:
            ValueError: If video_url is None, empty, or invalid format
        
        Example:
            >>> loader = YouTubeDocumentLoader(
            ...     video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            ...     language="en",
            ...     source_name="Tutorial"
            ... )
        """
        # Validate video URL is provided
        if not video_url:
            raise ValueError("video_url cannot be None or empty")
        
        # Validate video URL format and extract video ID
        self.video_id = self._extract_video_id(video_url)
        
        # Store parameters
        self.video_url = video_url
        self.language = language
        self.source_name = source_name
    
    def load_documents(self) -> List[Document]:
        """Load YouTube transcript and return as single Document with metadata."""
        # Load transcript using LangChain's YoutubeLoader
        documents = self._load_transcript()
        
        # Add standardized metadata
        for doc in documents:
            # Ensure video URL is in metadata
            if "source" not in doc.metadata:
                doc.metadata["source"] = self.video_url
            
            # Add video ID
            doc.metadata["video_id"] = self.video_id
            
            # Add loader type (required by base interface)
            self._add_metadata(doc.metadata)
        
        return documents
    
    def _load_transcript(self) -> List[Document]:
        """
        Load transcript using LangChain's YoutubeLoader.
        
        This is an internal helper method for fetching transcripts.
        
        Returns:
            List[Document]: List with one Document containing transcript
            
        Raises:
            ValueError: If transcript is not available
            
        Implementation Notes:
            - Uses LangChain's YoutubeLoader
            - YoutubeLoader uses youtube-transcript-api
            - Automatically fetches transcript in specified language
            - Falls back to available transcripts if language not found
            - Returns metadata including title and author
        """
        try:
            # YoutubeLoader needs the video ID, not full URL
            # But it can also accept full URL and extract ID
            loader = YoutubeLoader.from_youtube_url(
                youtube_url=self.video_url,
                language=self.language
            )
            documents = loader.load()
            
            # Verify we got content
            if not documents:
                raise ValueError(
                    f"No transcript available for video: {self.video_url}\n"
                    f"Video ID: {self.video_id}\n"
                    f"The video may not have captions enabled."
                )
            
            return documents
            
        except Exception as e:
            # Provide clear error message with context
            error_msg = str(e)
            
            # Check for common errors
            if "transcript" in error_msg.lower() or "subtitle" in error_msg.lower():
                raise ValueError(
                    f"Transcript not available for video: {self.video_url}\n"
                    f"Video ID: {self.video_id}\n"
                    f"Language: {self.language}\n"
                    f"Error: {error_msg}\n\n"
                    f"Possible reasons:\n"
                    f"  - Video has no captions/subtitles\n"
                    f"  - Requested language '{self.language}' not available\n"
                    f"  - Video is private or age-restricted"
                )
            else:
                raise ValueError(
                    f"Failed to load transcript from YouTube: {self.video_url}\n"
                    f"Video ID: {self.video_id}\n"
                    f"Error: {error_msg}"
                )
    
    def _extract_video_id(self, url: str) -> str:
        """
        Extract YouTube video ID from various URL formats.
        
        Args:
            url: YouTube video URL
            
        Returns:
            str: YouTube video ID (11 characters)
            
        Raises:
            ValueError: If URL is invalid or video ID cannot be extracted
            
        Supported Formats:
            - https://www.youtube.com/watch?v=VIDEO_ID
            - https://youtu.be/VIDEO_ID
            - https://www.youtube.com/embed/VIDEO_ID
            - https://m.youtube.com/watch?v=VIDEO_ID
            - https://www.youtube.com/v/VIDEO_ID
        
        Implementation Notes:
            - Handles most common YouTube URL formats
            - Video IDs are always 11 characters
            - Validates extracted ID format
        """
        try:
            # Parse URL
            parsed = urlparse(url)
            
            # Handle youtu.be short URLs
            if parsed.netloc in ['youtu.be', 'www.youtu.be']:
                # Video ID is in the path
                video_id = parsed.path.lstrip('/')
                # Remove any additional path components
                video_id = video_id.split('/')[0]
            
            # Handle standard youtube.com URLs
            elif 'youtube.com' in parsed.netloc:
                # Check for /watch?v= format
                if 'watch' in parsed.path:
                    query_params = parse_qs(parsed.query)
                    if 'v' in query_params:
                        video_id = query_params['v'][0]
                    else:
                        raise ValueError("No video ID found in URL")
                
                # Check for /embed/ format
                elif '/embed/' in parsed.path:
                    video_id = parsed.path.split('/embed/')[1]
                    video_id = video_id.split('/')[0]
                
                # Check for /v/ format
                elif '/v/' in parsed.path:
                    video_id = parsed.path.split('/v/')[1]
                    video_id = video_id.split('/')[0]
                
                else:
                    raise ValueError(f"Unsupported YouTube URL format: {url}")
            
            else:
                raise ValueError(f"Not a valid YouTube URL: {url}")
            
            # Remove any query parameters or fragments
            video_id = video_id.split('?')[0].split('#')[0]
            
            # Validate video ID format (should be 11 characters)
            if not video_id or len(video_id) != 11:
                raise ValueError(
                    f"Invalid video ID extracted: '{video_id}'\n"
                    f"YouTube video IDs are always 11 characters."
                )
            
            # Validate video ID contains only valid characters
            if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
                raise ValueError(
                    f"Invalid video ID format: '{video_id}'\n"
                    f"Video IDs should contain only letters, numbers, hyphens, and underscores."
                )
            
            return video_id
            
        except ValueError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to extract video ID from URL: {url}\n"
                f"Error: {str(e)}"
            )
    
    # ========================================================================
    # TODO: Advanced Features (Future Enhancements)
    # ========================================================================
    
    # TODO: Chunking Long Transcripts
    # --------------------------------
    # Long video transcripts can be too large for embedding models or LLM context windows.
    # Implementing intelligent chunking preserves context and improves retrieval.
    #
    # Implementation approach:
    #   1. Split transcript by timestamps
    #   2. Create overlapping chunks with time boundaries
    #   3. Preserve sentence boundaries
    #   4. Add timestamp metadata to each chunk
    #
    # Example:
    #   from langchain.text_splitter import RecursiveCharacterTextSplitter
    #
    #   def load_with_chunking(
    
    # ========================================================================
    # Supported Future Enhancements (See GitHub Issues)
    # ========================================================================
    # - Transcript chunking with timestamp preservation
    # - Multi-language support with language fallback
    # - Caption type selection (manual vs auto-generated)
    # - Playlist batch loading
    # - Video metadata enrichment via YouTube API
