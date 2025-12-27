```
RAG RETRIEVER - QUICK REFERENCE
==================================================================================

OVERVIEW
--------
RAGRetriever fetches relevant documents from a vector store for a given query.
It provides flexible retrieval strategies, history-aware querying, and 
configurable filtering to get the most relevant context for your RAG chain.

Key features:
✅ Multiple search strategies (similarity, MMR, score threshold)
✅ Chat history integration for conversational RAG
✅ Configurable k (number of results) and scoring
✅ Score filtering for quality control
✅ Direct LangChain retriever access

==================================================================================

BASIC USAGE
-----------

1. SIMPLE RETRIEVAL
   ```python
   from app.rag.retriever import RAGRetriever
   from app.rag.indexer import create_indexer
   
   # Create and populate vector store
   indexer = create_indexer()
   vector_store = indexer.index_documents(documents, "my_index")
   
   # Create retriever
   retriever = RAGRetriever(vector_store)
   
   # Retrieve documents
   docs = retriever.retrieve("What is RAG?", k=4)
   
   for doc in docs:
       print(doc.page_content)
   ```

2. WITH CONVENIENCE FUNCTION
   ```python
   from app.rag.retriever import create_retriever
   
   # Auto-loads settings from config
   retriever = create_retriever(vector_store)
   
   # Use immediately
   docs = retriever.retrieve("What is RAG?")
   ```

3. WITH SCORES
   ```python
   # Get documents with relevance scores
   results = retriever.retrieve_with_scores("What is RAG?", k=3)
   
   for doc, score in results:
       print(f"Score: {score:.3f}")
       print(f"Content: {doc.page_content[:100]}...\n")
   ```

==================================================================================

INITIALIZATION
--------------

RAGRetriever(
    vector_store,           # VectorStore instance (required)
    k: int = 4,            # Default number of documents to retrieve
    score_threshold: float = 0.0,  # Minimum similarity score
    search_type: str = "similarity",  # Search strategy
    **kwargs                # Additional config
)

PARAMETERS:
  vector_store: FAISS/Chroma/Pinecone vector store with documents
  k: Default number of results (can override per query)
  score_threshold: Filter out docs below this score (0.0 = no filter)
  search_type: "similarity" or "mmr" (Maximal Marginal Relevance)
  **kwargs: Stored in self.config for extensions

EXAMPLES:
  # Basic
  retriever = RAGRetriever(vector_store)
  
  # With filtering
  retriever = RAGRetriever(vector_store, score_threshold=0.7)
  
  # With MMR (diverse results)
  retriever = RAGRetriever(vector_store, search_type="mmr")
  
  # All options
  retriever = RAGRetriever(
      vector_store=vector_store,
      k=5,
      score_threshold=0.6,
      search_type="similarity"
  )

==================================================================================

MAIN METHOD: retrieve()
------------------------

docs = retriever.retrieve(
    query,                  # Search query (required)
    k: int = None,         # Override default k
    chat_history = None,    # List of (question, answer) tuples
    **kwargs                # Additional retrieval parameters
)

PARAMETERS:
  query: Search query string (non-empty)
  k: Number of documents (overrides default if provided)
  chat_history: [(q1, a1), (q2, a2), ...] for context-aware retrieval
  **kwargs: score_threshold, search_type, etc.

RETURNS:
  List[Document] - Ordered by relevance (most relevant first)
  
  Document structure:
  - page_content: Text content
  - metadata: {"source": "file.pdf", "page": 1, ...}

RAISES:
  ValueError: If query is empty

==================================================================================

RETRIEVAL METHODS
-----------------

1. retrieve() - Main method
   Returns: List[Document]
   Use: When you just need the documents

2. retrieve_with_scores() - With relevance scores
   Returns: List[Tuple[Document, float]]
   Use: When you need to see/log scores

3. get_retriever() - Get LangChain retriever
   Returns: VectorStoreRetriever
   Use: For direct LangChain integration

EXAMPLES:

# Method 1: Just documents
docs = retriever.retrieve("What is RAG?")
print(f"Found {len(docs)} documents")

# Method 2: With scores
results = retriever.retrieve_with_scores("What is RAG?")
for doc, score in results:
    print(f"Relevance: {score:.2f}")

# Method 3: LangChain integration
lc_retriever = retriever.get_retriever(k=5)
docs = lc_retriever.get_relevant_documents("What is RAG?")

==================================================================================

SEARCH STRATEGIES
-----------------

SIMILARITY SEARCH (default):
  Finds most similar documents based on cosine similarity.
  
  Best for: General question answering
  Pros: Fast, straightforward
  Cons: May return redundant documents
  
  Usage:
    retriever = RAGRetriever(vector_store, search_type="similarity")
    docs = retriever.retrieve("What is RAG?")

MAXIMAL MARGINAL RELEVANCE (MMR):
  Balances relevance with diversity to avoid redundancy.
  
  Best for: When you want diverse perspectives
  Pros: Reduces redundancy, broader coverage
  Cons: Slightly slower
  
  Usage:
    retriever = RAGRetriever(vector_store, search_type="mmr")
    docs = retriever.retrieve("What is RAG?")

SCORE THRESHOLD FILTERING:
  Only returns documents above minimum similarity score.
  
  Best for: Quality control, filtering low-relevance results
  Pros: Ensures quality, reduces noise
  Cons: Might return fewer than k documents
  
  Usage:
    retriever = RAGRetriever(
        vector_store,
        score_threshold=0.7  # Only docs with score >= 0.7
    )
    docs = retriever.retrieve("What is RAG?")

==================================================================================

CHAT HISTORY INTEGRATION
-------------------------

For conversational RAG, pass chat history to retrieve():

EXAMPLE:
```python
# First question
docs1 = retriever.retrieve("What is RAG?")
answer1 = generate_answer(docs1)  # From RAGChain

# Build history
history = [("What is RAG?", answer1)]

# Follow-up (retriever uses history for context)
docs2 = retriever.retrieve(
    "How does it work?",
    chat_history=history
)
answer2 = generate_answer(docs2)

# Continue conversation
history.append(("How does it work?", answer2))
docs3 = retriever.retrieve("What are the benefits?", chat_history=history)
```

HOW IT WORKS (Current Implementation):
  - History is passed through but not actively used yet
  - Prepared for future LLM-based query rewriting
  - See TODO in retriever.py for enhancement

FUTURE ENHANCEMENT:
  - LLM will rewrite follow-up questions into standalone queries
  - Example: "How does it work?" → "How does RAG work?"
  - Improves retrieval quality for conversational queries

==================================================================================

EXAMPLES
--------

EXAMPLE 1: Basic Document Retrieval
```python
from app.rag.retriever import create_retriever
from app.rag.indexer import create_indexer

# Index some documents
indexer = create_indexer()
docs = [Document(page_content="RAG is...", metadata={"source": "a.pdf"})]
vector_store = indexer.index_documents(docs, "my_docs")

# Create retriever
retriever = create_retriever(vector_store)

# Retrieve
results = retriever.retrieve("What is RAG?", k=3)
print(f"Found {len(results)} relevant documents")
```

EXAMPLE 2: With Score Filtering
```python
# Only get high-quality matches
retriever = RAGRetriever(vector_store, score_threshold=0.75)

docs = retriever.retrieve("What is RAG?", k=5)
# May return fewer than 5 if some scores < 0.75

print(f"High-quality matches: {len(docs)}")
```

EXAMPLE 3: Checking Relevance Scores
```python
results = retriever.retrieve_with_scores("What is RAG?", k=5)

for i, (doc, score) in enumerate(results, 1):
    print(f"\nResult {i} (Score: {score:.3f}):")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Preview: {doc.page_content[:100]}...")
    
    if score < 0.5:
        print("⚠️ Low relevance score - may not be useful")
```

EXAMPLE 4: MMR for Diversity
```python
# Get diverse results (less redundancy)
retriever = RAGRetriever(vector_store, search_type="mmr")

docs = retriever.retrieve("Benefits of RAG", k=5)
# Results will be relevant but also diverse
```

EXAMPLE 5: Dynamic k Based on Query
```python
retriever = create_retriever(vector_store)

# Simple question - need fewer docs
docs1 = retriever.retrieve("What is RAG?", k=2)

# Complex question - need more context
docs2 = retriever.retrieve(
    "Compare RAG vs fine-tuning for different use cases",
    k=10
)
```

EXAMPLE 6: Integration with RAGChain
```python
from app.rag.retriever import create_retriever
from app.rag.chain import create_rag_chain

# Create retriever
retriever = create_retriever(vector_store)

# Create chain with retriever
chain = create_rag_chain(retriever)

# Chain automatically uses retriever
result = chain.run("What is RAG?")
print(result["answer"])
```

EXAMPLE 7: Conversational Retrieval
```python
retriever = create_retriever(vector_store)
history = []

# First question
docs1 = retriever.retrieve("What is machine learning?")
answer1 = "Machine learning is..."
history.append(("What is machine learning?", answer1))

# Follow-up with context
docs2 = retriever.retrieve(
    "How is it different from traditional programming?",
    chat_history=history
)
```

==================================================================================

RESPONSE STRUCTURE
------------------

RETRIEVE() RETURNS:
[
    Document(
        page_content="RAG stands for Retrieval-Augmented Generation...",
        metadata={
            "source": "rag_basics.pdf",
            "page": 1,
            "chunk_id": 0,
            ...
        }
    ),
    Document(
        page_content="RAG systems combine retrieval with LLMs...",
        metadata={
            "source": "rag_overview.md",
            "page": 3,
            "chunk_id": 5,
            ...
        }
    ),
    ...
]

RETRIEVE_WITH_SCORES() RETURNS:
[
    (Document(...), 0.892),  # Highest relevance
    (Document(...), 0.854),
    (Document(...), 0.801),
    (Document(...), 0.745),
    ...
]

GET_RETRIEVER() RETURNS:
VectorStoreRetriever object (LangChain interface)

==================================================================================

CONFIGURATION
-------------

FROM SETTINGS (via create_retriever):
  retriever = create_retriever(vector_store)
  # Reads from Settings:
  # - retriever_top_k (default k)
  # - similarity_threshold (score filtering)

EXPLICIT PARAMETERS:
  retriever = RAGRetriever(
      vector_store=vector_store,
      k=5,
      score_threshold=0.7,
      search_type="similarity"
  )

PER-QUERY OVERRIDE:
  docs = retriever.retrieve(
      "query",
      k=10,                    # Overrides default
      score_threshold=0.8      # Overrides default
  )

==================================================================================

BEST PRACTICES
--------------

✅ DO:
  - Use create_retriever() for quick setup
  - Check document count before processing (might be 0)
  - Use score_threshold to filter low-quality results
  - Log relevance scores for monitoring
  - Use MMR when diversity is important
  - Pass chat_history for conversational RAG
  - Adjust k based on query complexity

❌ DON'T:
  - Don't assume exactly k documents returned (filtering may reduce count)
  - Don't ignore relevance scores (they indicate quality)
  - Don't use very high score_threshold (might get 0 results)
  - Don't retrieve more documents than needed (impacts performance)
  - Don't forget to validate query is non-empty
  - Don't modify returned Document objects (they're from vector store)

==================================================================================

TROUBLESHOOTING
---------------

ISSUE: No documents returned
SOLUTIONS:
  - Lower score_threshold (or set to 0.0)
  - Increase k
  - Check if documents were properly indexed
  - Verify query is similar to indexed content

ISSUE: Low relevance scores
SOLUTIONS:
  - Check embedding model matches indexing model
  - Verify documents contain relevant information
  - Try rephrasing query
  - Consider re-indexing with better chunking

ISSUE: Too many irrelevant results
SOLUTIONS:
  - Increase score_threshold (e.g., 0.7)
  - Use smaller k
  - Improve document quality
  - Use more specific queries

ISSUE: Redundant results
SOLUTIONS:
  - Switch to MMR search: search_type="mmr"
  - Adjust chunk_overlap in indexing
  - Post-process to remove duplicates

==================================================================================

PERFORMANCE TIPS
----------------

SPEED:
  - FAISS is fastest (optimized for similarity search)
  - Chroma is good balance (persistence + speed)
  - Pinecone/Qdrant for cloud scale

QUALITY:
  - Use score_threshold to ensure quality
  - Monitor relevance scores over time
  - A/B test different k values
  - Experiment with similarity vs MMR

COST (Token Usage):
  - Retrieve only needed documents (smaller k)
  - Use score_threshold to exclude low-quality docs
  - Consider caching frequent queries

==================================================================================

INTEGRATION POINTS
------------------

WITH VECTOR STORE:
  retriever = RAGRetriever(vector_store)
  # Works with: FAISS, Chroma, Pinecone, Qdrant
  # Requires: vector_store.similarity_search_with_score()

WITH RAGChain:
  chain = RAGChain(retriever=retriever, llm=llm)
  # Chain calls: retriever.retrieve(query, chat_history, **kwargs)

WITH LangChain:
  lc_retriever = retriever.get_retriever(k=5)
  # Use in LangChain chains, agents, etc.

==================================================================================

ADVANCED USAGE
--------------

CUSTOM SEARCH PARAMETERS:
```python
# Pass custom kwargs to vector store
docs = retriever.retrieve(
    "query",
    k=5,
    fetch_k=20,         # MMR: number to fetch before filtering
    lambda_mult=0.5,    # MMR: diversity parameter
    score_threshold=0.7
)
```

SUBCLASSING:
```python
class CustomRetriever(RAGRetriever):
    """Add custom retrieval logic."""
    
    def retrieve(self, query, **kwargs):
        # Pre-process query
        query = self._preprocess_query(query)
        
        # Call parent
        docs = super().retrieve(query, **kwargs)
        
        # Post-process results
        return self._postprocess_docs(docs)
```

METADATA FILTERING (Future):
```python
# Filter by metadata (when implemented)
docs = retriever.retrieve(
    "What is RAG?",
    metadata_filter={"source": "*.pdf", "date": "2024"}
)
```

==================================================================================

ERROR HANDLING
--------------

VALIDATION ERRORS:
```python
try:
    docs = retriever.retrieve("")
except ValueError as e:
    print(f"Invalid query: {e}")
```

NO RESULTS:
```python
docs = retriever.retrieve("query", k=5)
if not docs:
    print("No relevant documents found")
    # Handle gracefully (don't assume docs exist)
```

VECTOR STORE ERRORS:
```python
try:
    docs = retriever.retrieve("query")
except Exception as e:
    print(f"Retrieval failed: {e}")
    # Vector store might be corrupted or inaccessible
```

==================================================================================

TESTING
-------

See tests/test_retriever.py for comprehensive test suite:
  ✅ 20+ test cases
  ✅ All retrieval methods tested
  ✅ Edge cases (empty query, no results)
  ✅ Score filtering validation
  ✅ Chat history integration
  ✅ Mock-based (fast)

Run tests:
  pytest tests/test_retriever.py -v

==================================================================================

DEMO
----

Run the demo to see retriever in action:
  python demo_retriever.py

Demos include:
  1. Basic retrieval with mock vector store
  2. Retrieval with scores
  3. Score threshold filtering
  4. MMR search for diversity

==================================================================================

TODO FEATURES
-------------

🔜 LLM-BASED HISTORY-AWARE RETRIEVAL:
   Use LLM to rewrite follow-up questions into standalone queries
   
   Example:
     History: [("What is RAG?", "RAG is...")]
     Query: "How does it work?"
     Rewritten: "How does RAG work?"  # Better for retrieval

🔜 HYBRID SEARCH:
   Combine semantic search with keyword search
   
   Benefits:
   - Semantic: Understands meaning
   - Keyword: Exact term matching
   - Hybrid: Best of both

🔜 METADATA FILTERING:
   Filter by source, date, author, etc.
   
   Example:
     docs = retriever.retrieve(
         "query",
         metadata_filter={"source": "technical_docs/*.pdf"}
     )

🔜 RE-RANKING:
   Use cross-encoder to re-rank initial results
   
   Process:
   1. Fast retrieval (get top 20)
   2. Slow re-ranking (re-rank to top 5)
   3. Better quality final results

🔜 QUERY EXPANSION:
   Generate multiple query variations for broader coverage

==================================================================================

RESOURCES
---------

Code:
  - Implementation: app/rag/retriever.py
  - Tests: tests/test_retriever.py
  - Demo: demo_retriever.py

Documentation:
  - This file: docs/RETRIEVER_QUICK_REF.md
  - Implementation details: docs/reference/RETRIEVER_IMPLEMENTATION_SUMMARY.md
  - Indexer docs: docs/INDEXER_QUICK_REF.md
  - Chain docs: docs/CHAIN_QUICK_REF.md

Related Components:
  - DocumentIndexer: app/rag/indexer.py
  - RAGChain: app/rag/chain.py
  - VectorStoreManager: app/vectorstore/manager.py
  - Settings: app/core/config.py

==================================================================================
```