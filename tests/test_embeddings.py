"""
Embedding Manager Test/Demo Script

This script demonstrates how to use the EmbeddingManager for RAG systems.
It shows lazy initialization, embedding generation, and similarity calculation.

Usage:
    python test_embeddings.py
"""

import numpy as np
from app.core.embeddings import EmbeddingManager
from app.utils.config import get_settings


def calculate_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def main():
    """Test the embedding manager."""
    
    print("=" * 70)
    print("Embedding Manager - Test & Demo")
    print("=" * 70)
    print()
    
    try:
        # Get settings
        settings = get_settings()
        
        print("📋 Configuration:")
        print(f"   • Embedding Model: {settings.embedding_model_name}")
        print()
        
        # Initialize manager (no model loaded yet)
        print("🔧 Initializing EmbeddingManager...")
        manager = EmbeddingManager()
        print("✅ Manager created (model NOT loaded yet - lazy initialization)")
        print()
        
        # Get embeddings object (model loads here on first call)
        print("🚀 Getting embeddings object (loading model - may take 10-30 seconds)...")
        embeddings = manager.get_embeddings()
        print(f"✅ Model loaded: {type(embeddings).__name__}")
        print()
        
        # Test with sample texts
        print("=" * 70)
        print("📝 Testing Semantic Similarity")
        print("=" * 70)
        print()
        
        # Sample texts with different levels of similarity
        query = "What is artificial intelligence?"
        
        texts = [
            "Artificial intelligence is the simulation of human intelligence by machines.",
            "Machine learning is a subset of AI that learns from data.",
            "The recipe for chocolate cake involves flour, eggs, and cocoa.",
        ]
        
        print(f"Query: '{query}'")
        print()
        print("Comparing against:")
        for i, text in enumerate(texts, 1):
            print(f"  {i}. {text}")
        print()
        
        # Generate embeddings
        print("🔄 Generating embeddings...")
        query_embedding = embeddings.embed_query(query)
        text_embeddings = [embeddings.embed_query(text) for text in texts]
        
        print(f"✅ Query embedding shape: {len(query_embedding)} dimensions")
        print()
        
        # Calculate similarities
        print("📊 Similarity Scores (0=unrelated, 1=identical):")
        print()
        
        similarities = []
        for i, (text, text_emb) in enumerate(zip(texts, text_embeddings), 1):
            similarity = calculate_similarity(query_embedding, text_emb)
            similarities.append((similarity, text))
            print(f"  {i}. Score: {similarity:.4f}")
            print(f"     Text: {text[:60]}...")
            print()
        
        # Show ranking
        print("🏆 Ranked by Relevance:")
        print()
        ranked = sorted(similarities, key=lambda x: x[0], reverse=True)
        for i, (score, text) in enumerate(ranked, 1):
            print(f"  {i}. [{score:.4f}] {text[:60]}...")
        print()
        
        # Demonstrate batch processing
        print("=" * 70)
        print("⚡ Batch Processing Demo")
        print("=" * 70)
        print()
        
        batch_texts = [
            "Machine learning algorithms",
            "Deep neural networks",
            "Natural language processing",
            "Computer vision systems",
            "Reinforcement learning agents",
        ]
        
        print(f"Embedding {len(batch_texts)} documents in batch...")
        batch_embeddings = embeddings.embed_documents(batch_texts)
        print(f"✅ Generated {len(batch_embeddings)} embeddings")
        print(f"   Each with {len(batch_embeddings[0])} dimensions")
        print()
        
        # Show usage in RAG context
        print("=" * 70)
        print("🔗 Usage in RAG Pipeline")
        print("=" * 70)
        print()
        
        print("from app.core.embeddings import EmbeddingManager")
        print("from langchain_community.vectorstores import FAISS")
        print()
        print("# Initialize")
        print("manager = EmbeddingManager()")
        print("embeddings = manager.get_embeddings()")
        print()
        print("# Create vector store from documents")
        print("vector_store = FAISS.from_documents(")
        print("    documents=documents,")
        print("    embedding=embeddings")
        print(")")
        print()
        print("# Similarity search")
        print("results = vector_store.similarity_search(")
        print("    query='What is RAG?',")
        print("    k=5")
        print(")")
        print()
        
        # Performance tip
        print("=" * 70)
        print("💡 Performance Tips")
        print("=" * 70)
        print()
        print("1. Lazy Initialization:")
        print("   • Model loads only when first needed")
        print("   • Subsequent calls reuse cached model")
        print("   • Saves memory if embeddings not used")
        print()
        print("2. Batch Processing:")
        print("   • Use embed_documents() for multiple texts")
        print("   • More efficient than embedding one at a time")
        print("   • Automatically processes in batches of 32")
        print()
        print("3. Model Selection:")
        print("   • all-MiniLM-L6-v2: Fast, 384 dim, 80MB")
        print("   • all-mpnet-base-v2: Better quality, 768 dim, 420MB")
        print("   • Set in .env: EMBEDDING_MODEL_NAME=...")
        print()
        
        print("=" * 70)
        print("✅ Test completed successfully!")
        print("=" * 70)
        print()
        
    except ValueError as e:
        print()
        print("❌ Configuration Error:")
        print(f"   {str(e)}")
        print()
        print("💡 Solution:")
        print("   1. Check your .env file")
        print("   2. Set EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2")
        print("   3. Run: python test_config.py")
        print()
        return 1
    
    except RuntimeError as e:
        print()
        print("❌ Model Loading Error:")
        print(f"   {str(e)}")
        print()
        return 1
    
    except Exception as e:
        print()
        print(f"❌ Unexpected Error: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("💡 Troubleshooting:")
        print("   • Ensure internet connection (first download)")
        print("   • Check available disk space (~500MB)")
        print("   • Verify Python has network access")
        print("   • Try a smaller model: all-MiniLM-L6-v2")
        print()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
