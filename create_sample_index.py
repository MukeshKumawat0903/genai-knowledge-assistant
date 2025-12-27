"""
Create a sample vector store index for testing RAG mode.
This script creates a FAISS index with demo documents.
"""
from langchain_core.documents import Document
from app.rag.indexer import create_indexer

# Create sample documents
docs = [
    Document(
        page_content='RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate more accurate and contextual responses.',
        metadata={'source': 'sample_rag_intro.txt', 'topic': 'RAG Basics'}
    ),
    Document(
        page_content='Vector databases store embeddings - numerical representations of text that capture semantic meaning. Similar texts have similar embeddings, enabling semantic search beyond keyword matching.',
        metadata={'source': 'sample_vectors.txt', 'topic': 'Embeddings'}
    ),
    Document(
        page_content='LangChain is a framework for developing applications powered by language models. It provides tools for chains, agents, memory, and retrieval to build sophisticated LLM applications.',
        metadata={'source': 'sample_langchain.txt', 'topic': 'LangChain'}
    ),
    Document(
        page_content='The GenAI Knowledge Assistant uses FAISS for vector storage, supports multiple document sources (PDF, web, YouTube), and can operate in both RAG mode for document Q&A and Agent mode for multi-tool queries.',
        metadata={'source': 'sample_assistant.txt', 'topic': 'Assistant Features'}
    ),
    Document(
        page_content='FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It is commonly used in RAG systems for fast retrieval of relevant documents.',
        metadata={'source': 'sample_faiss.txt', 'topic': 'FAISS'}
    )
]

print('🔄 Creating vector store index with sample documents...')
print(f'   Documents to index: {len(docs)}')

try:
    # Create indexer and index the documents
    indexer = create_indexer()
    result = indexer.index_documents(docs)
    
    print(f"\n✅ Index created successfully!")
    print(f"   Documents indexed: {result.get('document_count', 0)}")
    print(f"   Chunks created: {result.get('chunk_count', 0)}")
    print(f"   Vector store type: {result.get('vector_store_type', 'unknown')}")
    print(f"   Status: {result.get('status', 'unknown')}")
    print(f"\n✓ You can now use RAG mode in the app!")
except Exception as e:
    print(f"\n❌ Error creating index: {e}")
    import traceback
    traceback.print_exc()
