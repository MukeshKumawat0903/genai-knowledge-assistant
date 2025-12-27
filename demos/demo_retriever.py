"""
RAG Retriever - Quick Demo

This script demonstrates the RAGRetriever for document retrieval in RAG systems.

Usage:
    python demo_retriever.py
"""

from app.rag.retriever import create_retriever


def demo_basic_retrieval():
    """Demonstrate basic document retrieval."""
    print("=" * 80)
    print("RAG Retriever - Basic Retrieval Demo")
    print("=" * 80)
    print()
    
    try:
        print("🔧 Creating retriever from existing vector store...")
        retriever = create_retriever()
        print("✓ Retriever created")
        print()
        
        # Test query
        query = "What is Retrieval-Augmented Generation?"
        print(f"📝 Query: {query}")
        print()
        
        print("🔍 Retrieving relevant documents...")
        docs = retriever.retrieve(query)
        print()
        
        print(f"✓ Found {len(docs)} relevant documents:")
        print("-" * 80)
        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. {doc.page_content[:150]}...")
            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print()
        print("Note: Run indexer first to create vector store:")
        print("  python demo_indexer.py")


def demo_custom_k():
    """Demonstrate retrieval with custom k parameter."""
    print()
    print("=" * 80)
    print("Custom K Parameter Demo")
    print("=" * 80)
    print()
    
    try:
        retriever = create_retriever()
        
        # Retrieve with different k values
        for k in [2, 5, 10]:
            print(f"🔍 Retrieving top-{k} documents...")
            docs = retriever.retrieve("machine learning", k=k)
            print(f"✓ Retrieved {len(docs)} documents")
        
    except RuntimeError as e:
        print(f"✗ Error: {e}")


def demo_conversational_retrieval():
    """Demonstrate history-aware retrieval."""
    print()
    print("=" * 80)
    print("Conversational Retrieval Demo")
    print("=" * 80)
    print()
    
    try:
        retriever = create_retriever()
        
        # Simulate conversation
        chat_history = [
            ("What is RAG?", "RAG combines retrieval with generation"),
            ("How does it work?", "It retrieves docs then generates answers")
        ]
        
        # Follow-up query
        print("📝 Previous questions:")
        for q, a in chat_history:
            print(f"   Q: {q}")
            print(f"   A: {a[:50]}...")
        print()
        
        follow_up = "What are the benefits?"
        print(f"📝 Follow-up: {follow_up}")
        print()
        
        print("🔍 Retrieving with chat history...")
        docs = retriever.retrieve(follow_up, chat_history=chat_history)
        print(f"✓ Retrieved {len(docs)} context-aware documents")
        
    except RuntimeError as e:
        print(f"✗ Error: {e}")


def demo_retrieval_with_scores():
    """Demonstrate retrieval with similarity scores."""
    print()
    print("=" * 80)
    print("Retrieval with Scores Demo")
    print("=" * 80)
    print()
    
    try:
        retriever = create_retriever()
        
        query = "neural networks"
        print(f"📝 Query: {query}")
        print()
        
        print("🔍 Retrieving with similarity scores...")
        docs_with_scores = retriever.retrieve_with_scores(query)
        print()
        
        print(f"✓ Retrieved {len(docs_with_scores)} documents with scores:")
        print("-" * 80)
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   Content: {doc.page_content[:100]}...")
        
    except RuntimeError as e:
        print(f"✗ Error: {e}")


def main():
    """Run all demos."""
    print("\n\n")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                     RAG RETRIEVER DEMONSTRATION                            ║")
    print("║                                                                            ║")
    print("║  This demo shows how to retrieve relevant documents from a vector store   ║")
    print("║  for Retrieval-Augmented Generation (RAG) systems.                        ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Run demos
    demo_basic_retrieval()
    demo_custom_k()
    demo_conversational_retrieval()
    demo_retrieval_with_scores()
    
    print()
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Experiment with different queries")
    print("  2. Try different search types (mmr, similarity_score_threshold)")
    print("  3. Integrate with LLM chain for complete RAG")
    print()
    print("Quick Start Code:")
    print("-" * 80)
    print("from app.rag.retriever import create_retriever")
    print()
    print("# Create retriever")
    print("retriever = create_retriever()")
    print()
    print("# Retrieve documents")
    print('docs = retriever.retrieve("Your question here")')
    print()
    print("# Use with LangChain")
    print("langchain_retriever = retriever.get_retriever()")
    print("-" * 80)
    print()


if __name__ == "__main__":
    main()
