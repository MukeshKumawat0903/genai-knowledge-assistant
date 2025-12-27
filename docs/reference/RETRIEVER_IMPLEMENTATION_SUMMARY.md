```
RAG RETRIEVER - IMPLEMENTATION SUMMARY
==================================================================================

DOCUMENT PURPOSE
----------------
This document provides comprehensive technical details about the RAGRetriever
implementation, including architecture, design decisions, workflow internals,
and extension strategies for developers.

==================================================================================

ARCHITECTURE OVERVIEW
---------------------

COMPONENT ROLE:
  RAGRetriever is the RETRIEVAL component of the RAG pipeline.
  
  RAG Pipeline Stages:
  1. INDEXING (DocumentIndexer) - Write documents to vector store
  2. RETRIEVAL (RAGRetriever) - Fetch relevant documents  ← THIS
  3. GENERATION (RAGChain) - Generate answer from context
  
  Retriever bridges: Vector Store ↔ RAG Chain

DEPENDENCIES:
  ┌──────────────┐
  │ RAGRetriever │
  └──────┬───────┘
         │
         └────────────┐
                      │
                ┌─────▼──────┐
                │VectorStore │
                └────────────┘
                (FAISS/Chroma/etc.)

KEY INTERFACES:
  - VectorStore: similarity_search(), similarity_search_with_score()
  - Returns: List[Document] or List[Tuple[Document, float]]
  - Settings: retriever_top_k, similarity_threshold

==================================================================================

CLASS DESIGN
------------

class RAGRetriever:
    """
    Wrapper around vector store for RAG-specific retrieval.
    
    Responsibilities:
    - Fetch relevant documents from vector store
    - Apply score filtering and ranking
    - Support multiple search strategies
    - Integrate chat history (for future enhancement)
    - Provide flexible k and threshold control
    
    Design Patterns:
    - Wrapper: Wraps VectorStore with RAG-friendly interface
    - Adapter: Adapts vector store API to RAG needs
    - Strategy: Supports different search strategies
    """
    
    ATTRIBUTES:
      vector_store: VectorStore - Underlying storage
      k: int - Default number of results
      score_threshold: float - Minimum similarity score
      search_type: str - Search strategy (similarity/mmr)
      config: dict - Additional configuration

    METHODS:
      __init__: Store vector store and config
      retrieve: Main retrieval method
      retrieve_with_scores: Get results with scores
      get_retriever: Return LangChain retriever
      _validate_query: Input validation

==================================================================================

INITIALIZATION
--------------

def __init__(
    self,
    vector_store,
    k: int = 4,
    score_threshold: float = 0.0,
    search_type: str = "similarity",
    **kwargs
):
    """
    Initialize retriever with vector store and parameters.
    
    DESIGN DECISION: Configuration via Constructor
    - Accept vector store reference (not path or config)
    - Benefits: flexible, works with any vector store implementation
    - Follows Dependency Injection pattern
    
    PARAMETERS:
      vector_store: Pre-initialized VectorStore instance
        - Must support similarity_search() and similarity_search_with_score()
        - Works with FAISS, Chroma, Pinecone, Qdrant, etc.
      
      k: Default number of documents to retrieve
        - Typical values: 3-10
        - Can override per query
      
      score_threshold: Minimum similarity score (0.0-1.0)
        - 0.0 = no filtering (return all k results)
        - 0.7 = only high-quality matches
        - May return fewer than k documents
      
      search_type: Search strategy
        - "similarity": Standard cosine similarity
        - "mmr": Maximal Marginal Relevance (diverse results)
      
      **kwargs: Stored for extensions
    
    VALIDATION:
      - No validation at init (fail fast on first use)
      - Vector store validated by duck typing
    """
    self.vector_store = vector_store
    self.k = k
    self.score_threshold = score_threshold
    self.search_type = search_type
    self.config = kwargs

DESIGN RATIONALE:
  ✅ Accepts vector store instance (flexible)
  ✅ Sensible defaults (k=4, no score filtering)
  ✅ Configurable search strategy
  ✅ **kwargs for extensibility
  ✅ Simple initialization (no complex logic)

==================================================================================

MAIN WORKFLOW: retrieve()
--------------------------

def retrieve(
    self,
    query: str,
    k: Optional[int] = None,
    chat_history: Optional[List[tuple]] = None,
    **kwargs
) -> List[Document]:
    """
    Retrieve relevant documents for a query.
    
    WORKFLOW STEPS:
    1. Validate query (raise ValueError if empty)
    2. Determine k (use parameter or default)
    3. Process chat history (future: query rewriting)
    4. Execute vector search
    5. Apply score filtering (if threshold > 0)
    6. Return documents
    
    CONTROL FLOW:
      query → validate → process_history → search → filter → return
    """

STEP-BY-STEP IMPLEMENTATION:

# STEP 1: Validate Query
self._validate_query(query)

RATIONALE:
  - Fail fast on invalid input
  - Clear error messages
  - Prevent wasted vector search calls

# STEP 2: Determine k
actual_k = k if k is not None else self.k

RATIONALE:
  - Allow per-query override
  - Fall back to default
  - Flexibility without complexity

# STEP 3: Process Chat History (Future)
if chat_history:
    # TODO: Use LLM to rewrite query based on history
    # query = self._rewrite_query_with_history(query, chat_history)
    pass

RATIONALE:
  - Prepared for future enhancement
  - Chat history passed through for now
  - When implemented, will improve conversational retrieval

# STEP 4: Execute Search
if self.score_threshold > 0:
    # Search with scores for filtering
    results = self.vector_store.similarity_search_with_score(
        query,
        k=actual_k
    )
    # Filter by score threshold
    filtered = [(doc, score) for doc, score in results if score >= self.score_threshold]
    documents = [doc for doc, score in filtered]
else:
    # No filtering, direct search
    documents = self.vector_store.similarity_search(
        query,
        k=actual_k
    )

RATIONALE:
  - Two paths: with/without score filtering
  - Optimization: skip scoring if not needed
  - Filtering happens in-memory (fast)

# STEP 5: Return Documents
return documents

RATIONALE:
  - Simple return (list of Documents)
  - Consistent with LangChain conventions
  - Easy to use in downstream components

==================================================================================

RETRIEVAL METHODS
-----------------

METHOD 1: retrieve()

Purpose: Main retrieval method, returns documents only
Signature: retrieve(query, k, chat_history, **kwargs) -> List[Document]
Use Cases: Standard RAG workflows, when scores not needed

Example:
  docs = retriever.retrieve("What is RAG?", k=5)
  for doc in docs:
      print(doc.page_content)

METHOD 2: retrieve_with_scores()

Purpose: Get documents with relevance scores
Signature: retrieve_with_scores(query, k, **kwargs) -> List[Tuple[Document, float]]
Use Cases: Monitoring, debugging, quality assessment

Example:
  results = retriever.retrieve_with_scores("What is RAG?", k=5)
  for doc, score in results:
      print(f"Score: {score:.3f} - {doc.page_content[:50]}...")

Implementation:
  def retrieve_with_scores(self, query: str, k: Optional[int] = None, **kwargs):
      self._validate_query(query)
      actual_k = k if k is not None else self.k
      return self.vector_store.similarity_search_with_score(query, k=actual_k)

RATIONALE:
  - Direct pass-through to vector store
  - No score filtering (user wants all scores)
  - Simple, minimal processing

METHOD 3: get_retriever()

Purpose: Return LangChain-compatible retriever object
Signature: get_retriever(k, **kwargs) -> VectorStoreRetriever
Use Cases: LangChain integration, chains, agents

Example:
  lc_retriever = retriever.get_retriever(k=5)
  docs = lc_retriever.get_relevant_documents("What is RAG?")

Implementation:
  def get_retriever(self, k: Optional[int] = None, **kwargs):
      actual_k = k if k is not None else self.k
      return self.vector_store.as_retriever(
          search_type=self.search_type,
          search_kwargs={"k": actual_k, **kwargs}
      )

RATIONALE:
  - Exposes underlying LangChain retriever
  - Enables advanced LangChain features
  - Maintains configuration (k, search_type)

==================================================================================

QUERY VALIDATION
----------------

def _validate_query(self, query: str) -> None:
    """
    Validate query before retrieval.
    
    CHECKS:
    1. Query is not None
    2. Query is not empty string
    3. Query is not just whitespace
    
    RAISES:
      ValueError with descriptive message
    """
    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty. Please provide a valid search query."
        )

DESIGN RATIONALE:
  ✅ Fail fast on invalid input
  ✅ Clear error message for users
  ✅ Prevents wasted vector search
  ✅ Private method (internal validation)

ALTERNATIVE APPROACHES CONSIDERED:

1. Silent failure (return empty list):
   Rejected: Hides bugs, confusing behavior

2. Warning instead of error:
   Rejected: Empty query is programmer error, not expected condition

3. Auto-fix (use default query):
   Rejected: "Explicit is better than implicit"

==================================================================================

SEARCH STRATEGIES
-----------------

SIMILARITY SEARCH (default):

Algorithm: Cosine similarity between query and document embeddings
Process:
  1. Convert query to embedding
  2. Compute cosine similarity with all document embeddings
  3. Sort by similarity (descending)
  4. Return top k

Pros:
  - Fast (optimized vector operations)
  - Intuitive (most similar first)
  - Works well for most use cases

Cons:
  - May return very similar documents (redundancy)
  - No diversity optimization

Use When:
  - Standard question answering
  - Need fastest retrieval
  - Redundancy is acceptable

Code:
  docs = self.vector_store.similarity_search(query, k=k)

MAXIMAL MARGINAL RELEVANCE (MMR):

Algorithm: Balances relevance and diversity
Process:
  1. Fetch more candidates (e.g., 20)
  2. Pick most relevant document
  3. For each remaining, score = relevance - λ * max_similarity_to_selected
  4. Pick highest score, repeat until k documents

Parameters:
  - fetch_k: Number of initial candidates (default: 20)
  - lambda_mult: Diversity vs relevance (0 = max diversity, 1 = max relevance)

Pros:
  - Reduces redundancy
  - Broader information coverage
  - Better for complex queries

Cons:
  - Slower (more documents fetched)
  - More complex algorithm
  - Requires tuning lambda_mult

Use When:
  - Need diverse perspectives
  - Avoiding redundant information
  - Answering complex questions

Code:
  retriever = RAGRetriever(vector_store, search_type="mmr")
  docs = retriever.retrieve(query, fetch_k=20, lambda_mult=0.5)

SCORE THRESHOLD FILTERING:

Algorithm: Filter out documents below minimum score
Process:
  1. Execute similarity search with scores
  2. Filter results: keep only score >= threshold
  3. Return filtered documents

Effect:
  - May return fewer than k documents
  - Guarantees minimum quality
  - Empty result if no docs meet threshold

Pros:
  - Quality control
  - Filters noise
  - Adaptable to query difficulty

Cons:
  - Unpredictable result count
  - May return no results
  - Requires threshold tuning

Use When:
  - Quality more important than quantity
  - Downstream process needs high confidence
  - Prefer no answer over wrong answer

Code:
  retriever = RAGRetriever(vector_store, score_threshold=0.7)
  docs = retriever.retrieve(query, k=5)  # May return < 5

==================================================================================

SCORE FILTERING IMPLEMENTATION
-------------------------------

CONDITIONAL FILTERING:

if self.score_threshold > 0:
    # Get scores for filtering
    results = self.vector_store.similarity_search_with_score(query, k=k)
    
    # Filter by threshold
    filtered = [
        (doc, score) 
        for doc, score in results 
        if score >= self.score_threshold
    ]
    
    # Extract documents
    documents = [doc for doc, score in filtered]
else:
    # No filtering needed
    documents = self.vector_store.similarity_search(query, k=k)

return documents

DESIGN RATIONALE:

Two-Path Approach:
  - WITH filtering: Get scores, filter, extract docs
  - WITHOUT filtering: Direct search (faster)

Why Not Always Get Scores?
  - similarity_search_with_score() is slightly slower
  - Unnecessary if threshold = 0
  - Optimization for common case

Filter In-Memory:
  - Vector store returns k results
  - Filtering happens in Python (fast)
  - Alternative: pass threshold to vector store (not all support)

May Return Fewer Than k:
  - Expected behavior when filtering
  - Caller should check len(documents)
  - Better than returning low-quality results

==================================================================================

CHAT HISTORY INTEGRATION
-------------------------

CURRENT IMPLEMENTATION:

def retrieve(self, query: str, chat_history: Optional[List[tuple]] = None, **kwargs):
    # History is accepted but not yet used
    if chat_history:
        # TODO: Implement history-aware retrieval
        pass
    
    # Continue with standard retrieval
    documents = self.vector_store.similarity_search(query, k=k)
    return documents

WHY NOT IMPLEMENTED YET?
  - Requires LLM for query rewriting
  - Adds complexity and latency
  - Most users can integrate RAGChain directly (which handles history)
  - Prepared for future enhancement

FUTURE IMPLEMENTATION PLAN:

def retrieve(self, query: str, chat_history: Optional[List[tuple]] = None, **kwargs):
    self._validate_query(query)
    
    # Rewrite query if history provided
    if chat_history and self.use_history_rewriting:
        query = self._rewrite_query_with_history(query, chat_history)
    
    # Continue with (possibly rewritten) query
    documents = self.vector_store.similarity_search(query, k=k)
    return documents

def _rewrite_query_with_history(self, query: str, history: List[tuple]) -> str:
    """
    Use LLM to rewrite follow-up query into standalone query.
    
    Example:
      History: [("What is RAG?", "RAG is Retrieval-Augmented Generation...")]
      Query: "How does it work?"
      Rewritten: "How does RAG work?"
    """
    # Format history
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
    
    # Create prompt
    prompt = f"""Given the conversation history and a follow-up question, rewrite the follow-up question to be a standalone question.

Conversation History:
{history_text}

Follow-up Question: {query}

Standalone Question:"""
    
    # Call LLM
    llm = LLMFactory.create()
    response = llm.invoke(prompt)
    
    return response.content.strip()

BENEFITS OF HISTORY-AWARE RETRIEVAL:
  ✅ Better retrieval for follow-up questions
  ✅ Resolves pronouns ("it", "they", "that")
  ✅ Adds missing context
  ✅ Improves conversational RAG

COSTS:
  ❌ Additional LLM call (adds latency)
  ❌ Token usage
  ❌ Potential for rewriting errors
  ❌ Complexity

WHEN TO ENABLE:
  - Multi-turn conversations
  - Users ask follow-up questions
  - Retrieval quality issues with pronouns
  - Latency is acceptable

==================================================================================

DESIGN PATTERNS
---------------

1. WRAPPER PATTERN
   
   Problem: Vector store API is low-level and complex
   Solution: RAGRetriever wraps it with simpler, RAG-focused interface
   
   Benefits:
   - Cleaner API for RAG use cases
   - Hides vector store complexity
   - Adds RAG-specific features (score filtering, history)
   
   Example:
     # Without wrapper (complex)
     results = vector_store.similarity_search_with_score(query, k=5)
     filtered = [(d, s) for d, s in results if s >= 0.7]
     docs = [d for d, s in filtered]
     
     # With wrapper (simple)
     docs = retriever.retrieve(query, k=5)

2. ADAPTER PATTERN
   
   Problem: Need to work with different vector store implementations
   Solution: Adapt various vector stores to common interface
   
   Benefits:
   - Works with FAISS, Chroma, Pinecone, etc.
   - Consistent API regardless of backend
   - Easy to switch vector stores
   
   Example:
     # Same code works with any vector store
     retriever = RAGRetriever(faiss_store)
     retriever = RAGRetriever(chroma_store)
     retriever = RAGRetriever(pinecone_store)

3. STRATEGY PATTERN
   
   Problem: Need different search strategies
   Solution: search_type parameter selects strategy
   
   Benefits:
   - Flexibility (similarity vs MMR)
   - Runtime selection
   - Easy to add new strategies
   
   Example:
     # Strategy 1: Similarity
     retriever = RAGRetriever(vs, search_type="similarity")
     
     # Strategy 2: MMR
     retriever = RAGRetriever(vs, search_type="mmr")

==================================================================================

INTEGRATION POINTS
------------------

WITH VECTOR STORE:

  Requirements:
  - similarity_search(query, k) -> List[Document]
  - similarity_search_with_score(query, k) -> List[Tuple[Document, float]]
  - as_retriever(**kwargs) -> VectorStoreRetriever (for LangChain)
  
  Supported Vector Stores:
  - FAISS (fast, local)
  - Chroma (persistent, local)
  - Pinecone (cloud, scalable)
  - Qdrant (cloud, feature-rich)

WITH RAGChain:

  Chain calls: retriever.retrieve(query, chat_history, **kwargs)
  Returns: List[Document]
  
  Chain uses:
  - Documents for context
  - Metadata for source attribution
  - Automatic integration via constructor injection

WITH LANGCHAIN:

  Get retriever: lc_retriever = retriever.get_retriever(k=5)
  Use in: RetrievalQA, ConversationalRetrievalChain, etc.
  
  Benefits:
  - Access to LangChain ecosystem
  - Advanced chains and agents
  - Streaming, callbacks, etc.

==================================================================================

CONVENIENCE FUNCTION
--------------------

def create_retriever(
    vector_store,
    k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    search_type: str = "similarity",
    **kwargs
) -> RAGRetriever:
    """
    Convenience function with Settings integration.
    
    KEY FEATURES:
    - Reads defaults from Settings (retriever_top_k, similarity_threshold)
    - Allows parameter override
    - Simpler for common case
    
    Usage:
      # Use Settings defaults
      retriever = create_retriever(vector_store)
      
      # Override settings
      retriever = create_retriever(
          vector_store,
          k=10,
          score_threshold=0.8
      )
    """
    settings = get_settings()
    
    return RAGRetriever(
        vector_store=vector_store,
        k=k if k is not None else settings.retriever_top_k,
        score_threshold=score_threshold if score_threshold is not None else settings.similarity_threshold,
        search_type=search_type,
        **kwargs
    )

DESIGN RATIONALE:
  - Reduce boilerplate (no need to import Settings)
  - Sensible defaults from configuration
  - Still allows full customization
  - Consistent with create_indexer(), create_rag_chain()

==================================================================================

TESTABILITY
-----------

UNIT TESTING APPROACH:

Mock Vector Store:
  mock_vs = Mock()
  mock_vs.similarity_search.return_value = [Document(...)]
  retriever = RAGRetriever(mock_vs)

Test Isolation:
  - Each method tested independently
  - Mock vector store for predictable results
  - Verify method calls and return values

Example Test:
  def test_retrieve_basic():
      mock_vs = Mock()
      mock_vs.similarity_search.return_value = [
          Document(page_content="test")
      ]
      retriever = RAGRetriever(mock_vs, k=3)
      
      docs = retriever.retrieve("query")
      
      mock_vs.similarity_search.assert_called_once_with("query", k=3)
      assert len(docs) == 1

Coverage Areas:
  ✅ Initialization
  ✅ Basic retrieval
  ✅ Score filtering
  ✅ Query validation
  ✅ k override
  ✅ Multiple search strategies
  ✅ LangChain retriever creation
  ✅ Edge cases (empty results, invalid query)

See tests/test_retriever.py for complete suite (20+ tests)

==================================================================================

PERFORMANCE CONSIDERATIONS
--------------------------

BOTTLENECKS:

1. Vector Search
   - FAISS: ~50ms for 100k documents
   - Chroma: ~100ms for 100k documents
   - Scales logarithmically with dataset size

2. Score Filtering
   - In-memory filtering: <1ms
   - Negligible overhead

3. Query Rewriting (Future)
   - LLM call: 200-500ms
   - Will be major bottleneck when implemented

OPTIMIZATION STRATEGIES:

1. Use Appropriate k
   - Smaller k = faster (less to fetch and rank)
   - Typical: k=3-5 for most queries
   - Increase only when needed

2. Choose Right Vector Store
   - FAISS: Fastest for read-heavy workloads
   - Chroma: Good balance with persistence
   - Pinecone: Best for distributed/cloud

3. Optimize Embeddings
   - Smaller embedding dimension = faster search
   - text-embedding-3-small (OpenAI) balances quality/speed

4. Caching (Application Level)
   - Cache frequent queries
   - Invalidate when documents change
   - Significant speedup for repeated queries

MEMORY USAGE:

- Small per-retriever (~1KB overhead)
- Documents kept in memory during processing
- Vector store keeps embeddings in memory (FAISS) or DB (Chroma)
- No persistent memory in retriever itself

==================================================================================

FUTURE ENHANCEMENTS
-------------------

ENHANCEMENT 1: LLM-Based History-Aware Retrieval

Current: History passed but not used
Future: LLM rewrites follow-up questions

Implementation:
  if chat_history and self.use_history_rewriting:
      query = self._rewrite_query_with_history(query, chat_history)

Benefits:
  - Better conversational retrieval
  - Resolves pronouns and context
  - Standalone queries improve search quality

ENHANCEMENT 2: Hybrid Search

Current: Pure semantic search
Future: Combine semantic + keyword search

Implementation:
  semantic_results = vector_store.similarity_search(query, k=k*2)
  keyword_results = self._keyword_search(query, k=k*2)
  combined = self._merge_and_rerank(semantic_results, keyword_results, k=k)

Benefits:
  - Semantic: Understands meaning
  - Keyword: Exact term matching
  - Hybrid: Best of both worlds

ENHANCEMENT 3: Metadata Filtering

Current: No filtering by metadata
Future: Filter by source, date, author, etc.

Implementation:
  docs = retriever.retrieve(
      "query",
      metadata_filter={"source": "*.pdf", "date_range": "2024"}
  )

Benefits:
  - Targeted retrieval
  - Scope control
  - Better organization

ENHANCEMENT 4: Re-Ranking

Current: Single-stage retrieval
Future: Two-stage retrieval + re-ranking

Implementation:
  # Stage 1: Fast retrieval (get 20)
  candidates = vector_store.similarity_search(query, k=20)
  
  # Stage 2: Slow re-ranking (keep 5)
  reranked = self._cross_encoder_rerank(query, candidates, k=5)

Benefits:
  - Better quality final results
  - Fast initial filtering
  - Precise final ranking

ENHANCEMENT 5: Query Expansion

Current: Single query
Future: Multiple query variations

Implementation:
  queries = self._expand_query("What is RAG?")
  # ["What is RAG?", "Explain RAG", "RAG definition", ...]
  
  all_results = []
  for q in queries:
      all_results.extend(retriever.retrieve(q, k=2))
  
  deduplicated = self._deduplicate(all_results)

Benefits:
  - Broader coverage
  - Multiple phrasings
  - More robust retrieval

==================================================================================

SECURITY CONSIDERATIONS
-----------------------

QUERY INJECTION:

Risk: Malicious queries could exploit vector search
Mitigation:
  - Query is just text (no code execution)
  - Validation prevents empty queries
  - Vector search is read-only

INFORMATION DISCLOSURE:

Risk: Retrieve unauthorized documents
Mitigation:
  - Vector store should implement access control
  - Retriever respects vector store permissions
  - No bypass mechanisms

RESOURCE EXHAUSTION:

Risk: Very large k could exhaust memory
Mitigation:
  - Reasonable k limits (typically < 100)
  - Vector store has internal limits
  - Application-level rate limiting

==================================================================================

DEPLOYMENT CONSIDERATIONS
--------------------------

PRODUCTION CHECKLIST:

✅ Vector Store Setup
  - Choose appropriate store for scale
  - Configure persistence (Chroma) or backups (FAISS)
  - Test disaster recovery

✅ Configuration
  - Set appropriate k in Settings
  - Tune score_threshold for quality
  - Choose search_type based on use case

✅ Monitoring
  - Log retrieval times
  - Track result counts
  - Monitor relevance scores

✅ Error Handling
  - Wrap retrieve() in try-except
  - Handle empty results gracefully
  - Provide user-friendly messages

EXAMPLE PRODUCTION WRAPPER:

class ProductionRetriever:
    def __init__(self, retriever, logger, cache):
        self.retriever = retriever
        self.logger = logger
        self.cache = cache
    
    async def retrieve(self, query, **kwargs):
        # Check cache
        cache_key = f"{query}:{kwargs}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # Retrieve
            start = time.time()
            docs = self.retriever.retrieve(query, **kwargs)
            duration = time.time() - start
            
            # Log metrics
            self.logger.info(
                f"Retrieved {len(docs)} docs in {duration:.3f}s"
            )
            
            # Cache
            await self.cache.set(cache_key, docs, ttl=3600)
            
            return docs
        
        except Exception as e:
            self.logger.error(f"Retrieval failed: {query}, Error: {e}")
            raise

==================================================================================

REFERENCES
----------

Code:
  - app/rag/retriever.py (implementation)
  - tests/test_retriever.py (test suite)
  - demo_retriever.py (examples)

Related Components:
  - app/rag/indexer.py (document indexing)
  - app/rag/chain.py (answer generation)
  - app/vectorstore/manager.py (vector store management)
  - app/core/config.py (configuration)

Documentation:
  - docs/RETRIEVER_QUICK_REF.md (usage guide)
  - docs/INDEXER_QUICK_REF.md (indexer docs)
  - docs/CHAIN_QUICK_REF.md (chain docs)

External Resources:
  - FAISS: https://github.com/facebookresearch/faiss
  - Chroma: https://www.trychroma.com/
  - LangChain: https://python.langchain.com/
  - MMR Paper: https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf

==================================================================================
END OF DOCUMENT
==================================================================================
```