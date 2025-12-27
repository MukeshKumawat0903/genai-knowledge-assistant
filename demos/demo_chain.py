"""
RAG Chain Demo

This demo shows how to use the RAGChain to combine document retrieval
with LLM generation for question answering.

Run with:
    python demo_chain.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from app.rag.chain import RAGChain, create_rag_chain
from app.core.llm import LLMFactory


# ============================================================================
# MOCK COMPONENTS (for demo without full setup)
# ============================================================================

class MockRetriever:
    """
    Mock retriever that simulates document retrieval.
    
    In production, this would be a real RAGRetriever instance.
    """
    
    def __init__(self):
        # Simulate a knowledge base
        self.knowledge_base = [
            Document(
                page_content="RAG (Retrieval-Augmented Generation) is an AI technique that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate accurate, grounded responses.",
                metadata={"source": "rag_overview.pdf", "page": 1}
            ),
            Document(
                page_content="RAG systems typically use vector databases to store document embeddings. When a query arrives, it's converted to an embedding and similar documents are retrieved using semantic search.",
                metadata={"source": "rag_technical.pdf", "page": 5}
            ),
            Document(
                page_content="The key benefit of RAG is grounding - answers are based on actual retrieved documents rather than just the LLM's parametric knowledge, reducing hallucinations.",
                metadata={"source": "rag_benefits.md", "page": 1}
            ),
            Document(
                page_content="RAG pipelines have three main stages: indexing (documents are chunked and embedded), retrieval (relevant chunks are fetched for a query), and generation (LLM produces an answer using the context).",
                metadata={"source": "rag_pipeline.pdf", "page": 3}
            ),
            Document(
                page_content="Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to find patterns in data.",
                metadata={"source": "ml_basics.pdf", "page": 1}
            )
        ]
    
    def retrieve(self, query: str, chat_history=None, k=3, **kwargs):
        """
        Simulate document retrieval.
        
        In reality, this would:
        1. Convert query to embedding
        2. Search vector store for similar documents
        3. Return top-k most relevant
        """
        # Simple keyword matching for demo
        query_lower = query.lower()
        
        # Score documents by keyword overlap
        scored_docs = []
        for doc in self.knowledge_base:
            score = 0
            content_lower = doc.page_content.lower()
            
            # Check for key terms
            if "rag" in query_lower and "rag" in content_lower:
                score += 2
            if "retrieval" in query_lower and "retrieval" in content_lower:
                score += 1
            if "machine learning" in query_lower and "machine learning" in content_lower:
                score += 2
            if "pipeline" in query_lower and "pipeline" in content_lower:
                score += 1
            if "benefit" in query_lower and "benefit" in content_lower:
                score += 1
            
            scored_docs.append((doc, score))
        
        # Sort by score and return top-k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in scored_docs[:k] if score > 0]
        
        # If no matches, return top 2 RAG-related docs as fallback
        if not top_docs:
            top_docs = self.knowledge_base[:2]
        
        return top_docs


class MockLLM:
    """
    Mock LLM that simulates text generation.
    
    In production, this would be a real LLM (Groq, etc.).
    """
    
    def invoke(self, prompt: str):
        """
        Simulate LLM generation.
        
        Extracts context from prompt and generates a simple response.
        """
        class Response:
            def __init__(self, content):
                self.content = content
        
        # Simple simulation: extract key info from context
        if "Context:" in prompt:
            context_part = prompt.split("Context:")[1].split("Question:")[0]
            
            # Generate a mock answer
            if "RAG" in prompt:
                answer = "Based on the provided context, RAG (Retrieval-Augmented Generation) is an AI technique that combines document retrieval with text generation. It helps produce more accurate and grounded responses by using information from a knowledge base."
            elif "pipeline" in prompt.lower():
                answer = "According to the context, RAG pipelines consist of three main stages: indexing (documents are chunked and embedded), retrieval (relevant chunks are fetched), and generation (LLM produces answers using the retrieved context)."
            else:
                answer = "Based on the context provided, the documents explain various aspects of RAG systems and their implementation."
        else:
            answer = "I don't have enough information to answer this question."
        
        return Response(answer)


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def demo_basic_query():
    """Demo 1: Basic question answering."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Question Answering")
    print("=" * 70)
    
    # Create mock components
    retriever = MockRetriever()
    llm = MockLLM()
    
    # Create chain
    chain = RAGChain(
        retriever=retriever,
        llm=llm
    )
    
    # Ask a question
    query = "What is RAG?"
    print(f"\n🔍 Query: {query}\n")
    
    result = chain.run(query, k=3)
    
    # Display answer
    print(f"💡 Answer:\n{result['answer']}\n")
    
    # Display sources
    print(f"📚 Sources ({len(result['source_documents'])}):")
    for i, doc in enumerate(result['source_documents'], 1):
        source = doc.metadata.get("source", "Unknown")
        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        print(f"  {i}. {source}")
        print(f"     {preview}\n")
    
    # Display metadata
    print(f"📊 Metadata:")
    print(f"  - Query: {result['metadata']['query']}")
    print(f"  - Sources used: {result['metadata']['num_sources']}")
    print(f"  - Has context: {result['metadata']['has_context']}")


def demo_with_chat_history():
    """Demo 2: Question with chat history."""
    print("\n" + "=" * 70)
    print("DEMO 2: Question Answering with Chat History")
    print("=" * 70)
    
    retriever = MockRetriever()
    llm = MockLLM()
    chain = RAGChain(retriever=retriever, llm=llm)
    
    # Simulate a conversation
    chat_history = [
        ("What is RAG?", "RAG is Retrieval-Augmented Generation..."),
        ("Why is it useful?", "It helps reduce hallucinations...")
    ]
    
    query = "How does the pipeline work?"
    
    print(f"\n📜 Previous conversation:")
    for q, a in chat_history:
        print(f"  Q: {q}")
        print(f"  A: {a[:50]}...\n")
    
    print(f"🔍 Current query: {query}\n")
    
    result = chain.run(query, chat_history=chat_history, k=2)
    
    print(f"💡 Answer:\n{result['answer']}\n")
    print(f"📚 Sources: {result['metadata']['num_sources']}")


def demo_no_relevant_docs():
    """Demo 3: Handling when no documents are found."""
    print("\n" + "=" * 70)
    print("DEMO 3: No Relevant Documents")
    print("=" * 70)
    
    # Create retriever that returns empty results
    class EmptyRetriever:
        def retrieve(self, query, chat_history=None, **kwargs):
            return []
    
    chain = RAGChain(
        retriever=EmptyRetriever(),
        llm=MockLLM()
    )
    
    query = "What is quantum computing?"
    print(f"\n🔍 Query: {query}\n")
    
    result = chain.run(query)
    
    print(f"💡 Answer:\n{result['answer']}\n")
    print(f"📊 Metadata:")
    print(f"  - Sources found: {result['metadata']['num_sources']}")
    print(f"  - Has context: {result['metadata']['has_context']}")


def demo_without_sources():
    """Demo 4: Chain configured to not return sources."""
    print("\n" + "=" * 70)
    print("DEMO 4: Without Source Documents")
    print("=" * 70)
    
    chain = RAGChain(
        retriever=MockRetriever(),
        llm=MockLLM(),
        return_source_documents=False
    )
    
    query = "What is RAG?"
    print(f"\n🔍 Query: {query}\n")
    
    result = chain.run(query)
    
    print(f"💡 Answer:\n{result['answer']}\n")
    print(f"📊 Result keys: {list(result.keys())}")
    print(f"   (Note: 'source_documents' is not included)")


def demo_convenience_function():
    """Demo 5: Using create_rag_chain convenience function."""
    print("\n" + "=" * 70)
    print("DEMO 5: Convenience Function")
    print("=" * 70)
    
    retriever = MockRetriever()
    llm = MockLLM()
    
    # Create chain using convenience function
    chain = create_rag_chain(
        retriever=retriever,
        llm=llm,
        return_source_documents=True
    )
    
    print("\n✅ Chain created with create_rag_chain()")
    print(f"   - Type: {type(chain).__name__}")
    print(f"   - Return sources: {chain.return_source_documents}")
    
    query = "Explain RAG benefits"
    print(f"\n🔍 Query: {query}\n")
    
    result = chain.run(query, k=2)
    print(f"💡 Answer:\n{result['answer']}\n")


def demo_context_formatting():
    """Demo 6: Show how context is formatted."""
    print("\n" + "=" * 70)
    print("DEMO 6: Context Formatting")
    print("=" * 70)
    
    chain = RAGChain(
        retriever=MockRetriever(),
        llm=MockLLM()
    )
    
    # Get some sample documents
    docs = [
        Document(
            page_content="RAG combines retrieval and generation.",
            metadata={"source": "rag_basics.pdf"}
        ),
        Document(
            page_content="It helps reduce hallucinations.",
            metadata={"source": "rag_benefits.md"}
        )
    ]
    
    context = chain._format_context(docs)
    
    print("\n📄 Formatted Context:")
    print("-" * 70)
    print(context)
    print("-" * 70)
    print("\nNote: Each document is labeled with its source for attribution")


def demo_prompt_structure():
    """Demo 7: Show the complete prompt structure."""
    print("\n" + "=" * 70)
    print("DEMO 7: Prompt Structure")
    print("=" * 70)
    
    chain = RAGChain(
        retriever=MockRetriever(),
        llm=MockLLM()
    )
    
    prompt = chain._create_prompt(
        query="What is RAG?",
        context="RAG is Retrieval-Augmented Generation. It combines retrieval with LLM generation.",
        chat_history=[("Previous question", "Previous answer")]
    )
    
    print("\n📝 Complete Prompt Structure:")
    print("-" * 70)
    print(prompt)
    print("-" * 70)
    print("\n✅ Notice:")
    print("  - Grounding instructions at the top")
    print("  - Chat history included")
    print("  - Context clearly marked")
    print("  - Question at the end")
    print("  - Reminder about using only the context")


def demo_error_handling():
    """Demo 8: Error handling."""
    print("\n" + "=" * 70)
    print("DEMO 8: Error Handling")
    print("=" * 70)
    
    chain = RAGChain(
        retriever=MockRetriever(),
        llm=MockLLM()
    )
    
    # Try empty query
    print("\n❌ Test 1: Empty query")
    try:
        chain.run("")
    except ValueError as e:
        print(f"   Caught error: {e}")
    
    # Try whitespace query
    print("\n❌ Test 2: Whitespace query")
    try:
        chain.run("   ")
    except ValueError as e:
        print(f"   Caught error: {e}")
    
    print("\n✅ Both errors handled correctly!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("RAG CHAIN DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows the RAGChain in action.")
    print("Note: Using mock components (retriever and LLM) for demonstration.")
    print("In production, you would use real RAGRetriever and LLMFactory.")
    
    try:
        # Run all demos
        demo_basic_query()
        demo_with_chat_history()
        demo_no_relevant_docs()
        demo_without_sources()
        demo_convenience_function()
        demo_context_formatting()
        demo_prompt_structure()
        demo_error_handling()
        
        print("\n" + "=" * 70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📚 Next Steps:")
        print("  1. Review the test suite: tests/test_chain.py")
        print("  2. Check documentation: docs/CHAIN_QUICK_REF.md")
        print("  3. Integrate with real RAGRetriever and LLMFactory")
        print("  4. Try with your own documents and questions")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
