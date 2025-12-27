"""
Document Indexer - Quick Demo

This script demonstrates the DocumentIndexer for RAG systems.
It shows how to index documents from different sources into a vector store.

Usage:
    python demo_indexer.py
"""

from langchain_core.documents import Document
from app.rag.indexer import create_indexer


def demo_basic_indexing():
    """Demonstrate basic document indexing."""
    print("=" * 80)
    print("Document Indexer - Basic Demo")
    print("=" * 80)
    print()
    
    # Create sample documents
    sample_documents = [
        Document(
            page_content="Retrieval-Augmented Generation (RAG) is a technique that combines "
                        "information retrieval with text generation. It retrieves relevant "
                        "documents and uses them to generate more accurate responses.",
            metadata={"source": "rag_overview.txt", "topic": "RAG"}
        ),
        Document(
            page_content="Vector databases store high-dimensional embeddings and enable "
                        "semantic search. Common vector databases include FAISS, Chroma, "
                        "Pinecone, and Weaviate.",
            metadata={"source": "vector_db.txt", "topic": "Vector Stores"}
        ),
        Document(
            page_content="LangChain is a framework for developing applications powered by "
                        "language models. It provides tools for document loading, text "
                        "splitting, vector stores, and chains.",
            metadata={"source": "langchain.txt", "topic": "LangChain"}
        ),
        Document(
            page_content="Embeddings are numerical representations of text that capture "
                        "semantic meaning. Similar texts have similar embeddings, enabling "
                        "semantic search and retrieval.",
            metadata={"source": "embeddings.txt", "topic": "Embeddings"}
        ),
    ]
    
    print(f"📚 Sample Documents: {len(sample_documents)}")
    for i, doc in enumerate(sample_documents, 1):
        print(f"   {i}. {doc.metadata['topic']} ({len(doc.page_content)} chars)")
    print()
    
    try:
        # Create indexer with default settings
        print("🔧 Creating indexer (using settings from .env)...")
        indexer = create_indexer()
        print("✓ Indexer created")
        print()
        
        # Index the documents
        print("📊 Indexing documents...")
        print("-" * 80)
        result = indexer.index_documents(sample_documents)
        print()
        
        # Display results
        if result['success']:
            print("✓ INDEXING SUCCESSFUL!")
            print("=" * 80)
            print(f"  Input Documents:    {result['num_documents']}")
            print(f"  Output Chunks:      {result['num_chunks']}")
            print(f"  Vector Store Type:  {result['vector_store_type']}")
            print(f"  Chunks per Doc:     {result['num_chunks'] / result['num_documents']:.1f}")
            print("=" * 80)
            print()
            print(f"✓ {result['message']}")
        else:
            print("✗ INDEXING FAILED")
            print("=" * 80)
            print(f"  Error: {result['message']}")
            print("=" * 80)
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        print()
        print("Note: This demo requires:")
        print("  1. .env file with proper configuration")
        print("  2. LangChain packages installed (langchain, sentence-transformers)")
        print("  3. All components properly configured")


def demo_custom_settings():
    """Demonstrate indexing with custom settings."""
    print()
    print("=" * 80)
    print("Document Indexer - Custom Settings Demo")
    print("=" * 80)
    print()
    
    # Create sample documents
    documents = [
        Document(
            page_content="Custom chunking settings allow you to control how documents are split.",
            metadata={"source": "custom.txt"}
        ),
        Document(
            page_content="Smaller chunks provide more precise retrieval but may lose context.",
            metadata={"source": "custom.txt"}
        ),
    ]
    
    try:
        # Create indexer with custom settings
        print("🔧 Creating indexer with custom settings:")
        print("   - Chunk Size: 50 characters")
        print("   - Chunk Overlap: 10 characters")
        print("   - Vector Store: FAISS")
        print()
        
        indexer = create_indexer(
            chunk_size=50,
            chunk_overlap=10,
            vector_store_type='faiss'
        )
        
        # Index with custom settings
        result = indexer.index_documents(documents)
        
        if result['success']:
            print("✓ Custom indexing successful!")
            print(f"  Created {result['num_chunks']} small chunks")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")


def demo_error_handling():
    """Demonstrate error handling."""
    print()
    print("=" * 80)
    print("Document Indexer - Error Handling Demo")
    print("=" * 80)
    print()
    
    try:
        indexer = create_indexer()
        
        # Test 1: Empty document list
        print("Test 1: Empty document list")
        result = indexer.index_documents([])
        print(f"  Result: {result['message']}")
        print()
        
        # Test 2: Invalid document types
        print("Test 2: Invalid document types")
        result = indexer.index_documents(["not a document", 123])
        print(f"  Result: {result['message']}")
        print()
        
        # Test 3: Empty content
        print("Test 3: Documents with empty content")
        empty_docs = [
            Document(page_content="Valid", metadata={}),
            Document(page_content="", metadata={}),
        ]
        result = indexer.index_documents(empty_docs)
        print(f"  Result: {result['message']}")
        print()
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")


def demo_pipeline_flow():
    """Show the indexing pipeline flow."""
    print()
    print("=" * 80)
    print("Indexing Pipeline Flow")
    print("=" * 80)
    print()
    print("1. Raw Documents")
    print("   ↓")
    print("2. Validation")
    print("   - Check document types")
    print("   - Verify non-empty content")
    print("   ↓")
    print("3. Text Chunking")
    print("   - Split by semantic boundaries")
    print("   - Apply chunk size and overlap")
    print("   - Preserve metadata")
    print("   ↓")
    print("4. Embedding Generation")
    print("   - Convert text to vectors")
    print("   - Use configured embedding model")
    print("   ↓")
    print("5. Vector Storage")
    print("   - Store in vector database")
    print("   - Index for fast search")
    print("   ↓")
    print("6. Persistence")
    print("   - Save to disk (FAISS)")
    print("   - Auto-persist (Chroma, Pinecone)")
    print("   ↓")
    print("7. Ready for Retrieval!")
    print()


def main():
    """Run all demos."""
    print("\n\n")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                     DOCUMENT INDEXER DEMONSTRATION                         ║")
    print("║                                                                            ║")
    print("║  This demo shows how to use the DocumentIndexer to build a searchable     ║")
    print("║  vector store for RAG systems.                                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Show pipeline flow
    demo_pipeline_flow()
    
    # Run basic demo
    demo_basic_indexing()
    
    # Show custom settings
    demo_custom_settings()
    
    # Show error handling
    demo_error_handling()
    
    print()
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Configure your .env file")
    print("  2. Install required packages:")
    print("     pip install langchain sentence-transformers faiss-cpu")
    print("  3. Load your own documents (PDF, Web, YouTube, etc.)")
    print("  4. Index them: result = indexer.index_documents(documents)")
    print("  5. Use retriever to query: retriever.retrieve(query)")
    print()
    print("Quick Start Code:")
    print("-" * 80)
    print("from app.rag.indexer import create_indexer")
    print("from app.ingestion.pdf_loader import PDFDocumentLoader")
    print()
    print("# Load and index")
    print("documents = PDFDocumentLoader('./docs').load_documents()")
    print("indexer = create_indexer()")
    print("result = indexer.index_documents(documents)")
    print()
    print("# Check result")
    print("if result['success']:")
    print("    print(f\"Indexed {result['num_chunks']} chunks\")")
    print("-" * 80)
    print()


if __name__ == "__main__":
    main()
