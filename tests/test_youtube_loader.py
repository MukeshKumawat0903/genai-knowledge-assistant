"""
Test script for YouTubeDocumentLoader

Tests the YouTube document loader implementation with various scenarios.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.youtube_loader import YouTubeDocumentLoader


def test_video_id_extraction():
    """Test video ID extraction from various URL formats."""
    print("=" * 70)
    print("Testing Video ID Extraction")
    print("=" * 70)
    
    test_cases = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/v/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ]
    
    for url, expected_id in test_cases:
        try:
            loader = YouTubeDocumentLoader(video_url=url)
            extracted_id = loader.video_id
            status = "✓" if extracted_id == expected_id else "✗"
            print(f"{status} {url}")
            print(f"  Expected: {expected_id}, Got: {extracted_id}")
        except Exception as e:
            print(f"✗ {url}")
            print(f"  Error: {e}")
    
    print()


def test_invalid_urls():
    """Test error handling for invalid URLs."""
    print("=" * 70)
    print("Testing Invalid URL Handling")
    print("=" * 70)
    
    invalid_urls = [
        "",
        "not-a-url",
        "https://example.com",
        "https://www.youtube.com/watch?v=invalid",  # Too short
        "https://www.youtube.com/watch?v=toolongvideoid12345",  # Too long
    ]
    
    for url in invalid_urls:
        try:
            loader = YouTubeDocumentLoader(video_url=url)
            print(f"✗ Should have failed: {url}")
        except ValueError as e:
            print(f"✓ Correctly rejected: {url}")
            print(f"  Error: {e}")
        except Exception as e:
            print(f"? Unexpected error: {url}")
            print(f"  Error: {e}")
    
    print()


def test_single_video_loading():
    """Test loading transcript from a single video."""
    print("=" * 70)
    print("Testing Single Video Loading")
    print("=" * 70)
    
    # Use a well-known video with transcripts
    # Note: You can replace this with any video URL that has transcripts
    test_video = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        print(f"Loading: {test_video}")
        loader = YouTubeDocumentLoader(
            video_url=test_video,
            language="en",
            source_name="Test Video"
        )
        
        documents = loader.load_documents()
        
        print(f"✓ Successfully loaded {len(documents)} document(s)")
        
        if documents:
            doc = documents[0]
            print(f"\nDocument Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            
            print(f"\nContent Preview (first 200 chars):")
            print(f"  {doc.page_content[:200]}...")
    
    except Exception as e:
        print(f"✗ Failed to load video")
        print(f"  Error: {e}")
        print(f"\nNote: This video might not have transcripts available.")
        print(f"      Try with a different video URL that has captions enabled.")
    
    print()


def test_language_parameter():
    """Test loading with different language parameters."""
    print("=" * 70)
    print("Testing Language Parameter")
    print("=" * 70)
    
    test_video = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    languages = ["en", "es", "fr", "de"]
    
    for lang in languages:
        try:
            print(f"Trying language: {lang}")
            loader = YouTubeDocumentLoader(
                video_url=test_video,
                language=lang
            )
            
            documents = loader.load_documents()
            print(f"  ✓ Loaded with {lang} language")
        
        except ValueError as e:
            if "transcript" in str(e).lower() or "subtitle" in str(e).lower():
                print(f"  ✗ No transcript available for {lang}")
            else:
                print(f"  ? Error: {e}")
        except Exception as e:
            print(f"  ? Unexpected error: {e}")
    
    print()


def test_interface_compliance():
    """Test that YouTubeDocumentLoader complies with BaseDocumentLoader interface."""
    print("=" * 70)
    print("Testing Interface Compliance")
    print("=" * 70)
    
    test_video = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        loader = YouTubeDocumentLoader(video_url=test_video)
        
        # Check required method exists
        if hasattr(loader, 'load_documents'):
            print("✓ load_documents() method exists")
        else:
            print("✗ load_documents() method missing")
        
        # Check it's callable
        if callable(getattr(loader, 'load_documents', None)):
            print("✓ load_documents() is callable")
        else:
            print("✗ load_documents() is not callable")
        
        # Check return type
        try:
            documents = loader.load_documents()
            
            if isinstance(documents, list):
                print("✓ Returns list")
            else:
                print(f"✗ Returns {type(documents)}, expected list")
            
            if documents:
                try:
                    from langchain_core.documents import Document
                except ImportError:  # pragma: no cover
                    from langchain.schema import Document
                if isinstance(documents[0], Document):
                    print("✓ Returns list of Document objects")
                else:
                    print(f"✗ Returns list of {type(documents[0])}")
                
                # Check metadata
                if hasattr(documents[0], 'metadata'):
                    print("✓ Documents have metadata")
                    
                    required_keys = ["source", "video_id", "loader_type", "ingestion_time"]
                    missing_keys = [key for key in required_keys if key not in documents[0].metadata]
                    
                    if not missing_keys:
                        print("✓ All required metadata keys present")
                    else:
                        print(f"✗ Missing metadata keys: {missing_keys}")
                else:
                    print("✗ Documents missing metadata attribute")
        
        except Exception as e:
            print(f"✗ Error during interface testing: {e}")
    
    except Exception as e:
        print(f"✗ Failed to create loader: {e}")
    
    print()


def test_error_messages():
    """Test that error messages are helpful and clear."""
    print("=" * 70)
    print("Testing Error Messages")
    print("=" * 70)
    
    # Test empty URL
    try:
        loader = YouTubeDocumentLoader(video_url="")
    except ValueError as e:
        print(f"✓ Empty URL error: {e}")
    
    # Test invalid video ID
    try:
        loader = YouTubeDocumentLoader(video_url="https://www.youtube.com/watch?v=invalid")
    except ValueError as e:
        print(f"✓ Invalid video ID error: {e}")
    
    # Test video without transcript (use a private/non-existent video)
    try:
        loader = YouTubeDocumentLoader(video_url="https://www.youtube.com/watch?v=00000000000")
        documents = loader.load_documents()
    except ValueError as e:
        print(f"✓ No transcript error: {e}")
    except Exception as e:
        print(f"? Unexpected error type: {type(e).__name__}: {e}")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("YouTube Document Loader Test Suite")
    print("=" * 70 + "\n")
    
    # Run tests
    test_video_id_extraction()
    test_invalid_urls()
    test_interface_compliance()
    test_error_messages()
    
    # Interactive test
    print("=" * 70)
    print("Interactive Test")
    print("=" * 70)
    print("\nYou can now test with your own video URL.")
    print("Enter a YouTube video URL (or press Enter to skip):")
    
    user_url = input("> ").strip()
    
    if user_url:
        test_single_video_loading_custom(user_url)
    
    print("\n" + "=" * 70)
    print("Test Suite Complete")
    print("=" * 70)


def test_single_video_loading_custom(video_url: str):
    """Test loading a custom video URL."""
    try:
        print(f"\nLoading: {video_url}")
        loader = YouTubeDocumentLoader(video_url=video_url)
        
        print(f"Video ID: {loader.video_id}")
        
        documents = loader.load_documents()
        
        print(f"✓ Successfully loaded {len(documents)} document(s)")
        
        if documents:
            doc = documents[0]
            print(f"\nDocument Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            
            content_length = len(doc.page_content)
            print(f"\nContent Length: {content_length} characters")
            
            preview_length = min(500, content_length)
            print(f"\nContent Preview (first {preview_length} chars):")
            print(f"  {doc.page_content[:preview_length]}...")
    
    except Exception as e:
        print(f"✗ Failed to load video")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
