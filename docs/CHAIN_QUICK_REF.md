```
RAG CHAIN - QUICK REFERENCE
==================================================================================

OVERVIEW
--------
RAGChain combines document retrieval with LLM generation to answer questions 
with source grounding. It orchestrates the complete RAG workflow from query to 
answer with cited sources.

Key features:
✅ Source-grounded responses (reduces hallucinations)
✅ Chat history integration
✅ Automatic context formatting
✅ Flexible configuration
✅ Comprehensive error handling

==================================================================================

BASIC USAGE
-----------

1. SIMPLE QUESTION ANSWERING
   ```python
   from app.rag.chain import RAGChain
   from app.rag.retriever import create_retriever
   from app.core.llm import LLMFactory
   
   # Create retriever and LLM
   retriever = create_retriever(vector_store)
   llm = LLMFactory.create()
   
   # Create chain
   chain = RAGChain(retriever=retriever, llm=llm)
   
   # Ask a question
   result = chain.run("What is RAG?")
   print(result["answer"])
   ```

2. WITH CHAT HISTORY
   ```python
   # Maintain conversation context
   chat_history = [
       ("What is machine learning?", "ML is..."),
       ("How does it work?", "It uses...")
   ]
   
   result = chain.run(
       "What about deep learning?",
       chat_history=chat_history
   )
   ```

3. CONVENIENCE FUNCTION
   ```python
   from app.rag.chain import create_rag_chain
   
   # Auto-creates LLM if not provided
   chain = create_rag_chain(retriever)
   
   # Or with custom LLM
   chain = create_rag_chain(retriever, llm=custom_llm)
   ```

==================================================================================

INITIALIZATION
--------------

RAGChain(
    retriever,              # RAGRetriever instance
    llm,                    # LangChain-compatible LLM
    prompt_manager=None,    # Optional custom prompts
    memory=None,            # Optional conversation memory
    return_source_documents=True,  # Include sources in response
    **kwargs                # Additional config
)

PARAMETERS:
  retriever: RAGRetriever instance for document fetching
  llm: LLM instance from LLMFactory.create()
  prompt_manager: Custom prompt templates (future enhancement)
  memory: Conversation memory (future enhancement)
  return_source_documents: If False, excludes sources from result
  **kwargs: Stored in self.config for custom extensions

==================================================================================

MAIN METHOD: run()
------------------

result = chain.run(
    query,                  # User's question (required)
    chat_history=None,      # List of (question, answer) tuples
    **kwargs                # Passed to retriever (k, score_threshold, etc.)
)

PARAMETERS:
  query: User's question (string, non-empty)
  chat_history: [(q1, a1), (q2, a2), ...] for conversation context
  **kwargs: Additional parameters passed to retriever.retrieve()
    - k: Number of documents to retrieve (default from Settings)
    - score_threshold: Minimum relevance score
    - search_type: "similarity" or "mmr"

RETURNS:
  Dictionary with:
  {
      "answer": "Generated answer string",
      "source_documents": [Document objects],  # If return_source_documents=True
      "metadata": {
          "query": "Original query",
          "num_sources": 3,
          "has_context": True
      }
  }

RAISES:
  ValueError: If query is empty or whitespace only

==================================================================================

RESPONSE STRUCTURE
------------------

SUCCESSFUL RESPONSE (with documents):
{
    "answer": "Based on the context, RAG is...",
    "source_documents": [
        Document(
            page_content="...",
            metadata={"source": "file.pdf", "page": 1}
        ),
        ...
    ],
    "metadata": {
        "query": "What is RAG?",
        "num_sources": 3,
        "has_context": True
    }
}

NO DOCUMENTS FOUND:
{
    "answer": "I don't have enough information to answer...",
    "source_documents": [],
    "metadata": {
        "query": "What is XYZ?",
        "num_sources": 0,
        "has_context": False
    }
}

==================================================================================

GROUNDING ENFORCEMENT
---------------------
The chain enforces source grounding through its prompt:

INSTRUCTIONS TO LLM:
  - "Use ONLY the information from the context"
  - "If the answer cannot be found, say 'I don't know'"
  - "Do not make up or infer information"
  - Explicit reminder at the end

PROMPT STRUCTURE:
  1. Grounding instructions
  2. Previous conversation (if provided)
  3. Context from retrieved documents
  4. User's question
  5. Reminder to use only context

CONTEXT FORMATTING:
  Documents are formatted as:
  
  [Source 1: document.pdf]
  Document content...
  
  [Source 2: article.md]
  More content...

==================================================================================

EXAMPLES
--------

EXAMPLE 1: Basic Usage
```python
from app.rag.chain import RAGChain, create_rag_chain
from app.rag.retriever import create_retriever
from app.rag.indexer import create_indexer
from app.core.llm import LLMFactory

# 1. Index documents
indexer = create_indexer()
vector_store = indexer.index_documents(documents, index_name="my_index")

# 2. Create retriever
retriever = create_retriever(vector_store)

# 3. Create chain
chain = create_rag_chain(retriever)

# 4. Ask questions
result = chain.run("What is the main topic of these documents?")
print(f"Answer: {result['answer']}\n")
print(f"Sources: {len(result['source_documents'])}")
```

EXAMPLE 2: Conversation
```python
# Maintain conversation history
history = []

# First question
result1 = chain.run("What is RAG?")
history.append(("What is RAG?", result1["answer"]))

# Follow-up (uses history)
result2 = chain.run("How does it work?", chat_history=history)
history.append(("How does it work?", result2["answer"]))

# Another follow-up
result3 = chain.run("What are the benefits?", chat_history=history)
```

EXAMPLE 3: Custom Retrieval Parameters
```python
# Control retrieval behavior
result = chain.run(
    "What is RAG?",
    k=5,                    # Retrieve 5 documents
    score_threshold=0.7,    # Min similarity score
    search_type="mmr"       # Use MMR for diversity
)
```

EXAMPLE 4: Without Source Documents
```python
# Create chain that doesn't return sources
chain = RAGChain(
    retriever=retriever,
    llm=llm,
    return_source_documents=False
)

result = chain.run("What is RAG?")
# result only has 'answer' and 'metadata', no 'source_documents'
```

EXAMPLE 5: Accessing Source Details
```python
result = chain.run("What is RAG?")

print(f"Answer: {result['answer']}\n")
print("Sources:")
for i, doc in enumerate(result['source_documents'], 1):
    source = doc.metadata.get('source', 'Unknown')
    page = doc.metadata.get('page', 'N/A')
    print(f"{i}. {source} (Page {page})")
    print(f"   Preview: {doc.page_content[:100]}...\n")
```

==================================================================================

INTEGRATION POINTS
------------------

RETRIEVER INTEGRATION:
  Chain calls: retriever.retrieve(query, chat_history, **kwargs)
  Expected return: List[Document]
  
  RAGRetriever provides:
  - retrieve(query, k, chat_history, score_threshold)
  - Multiple search strategies (similarity, MMR)
  - History-aware retrieval

LLM INTEGRATION:
  Chain calls: llm.invoke(prompt)
  Expected return: AIMessage with .content attribute
  
  LLMFactory provides:
  - Groq (fast cloud inference)
  - Easy to extend for OpenAI, Anthropic, etc.

==================================================================================

INTERNAL METHODS
----------------

_format_context(documents: List[Document]) -> str
  Formats retrieved documents into single context string
  
  Input: [Document(page_content="...", metadata={"source": "file.pdf"})]
  Output: "[Source 1: file.pdf]\n<content>\n\n[Source 2: ...]"

_create_prompt(query, context, chat_history=None) -> str
  Creates grounded prompt with instructions, history, context, and question
  
  Returns complete prompt string ready for LLM invocation

==================================================================================

ERROR HANDLING
--------------

COMMON ERRORS:

1. Empty Query
   ```python
   chain.run("")  # Raises ValueError: "Query cannot be empty"
   ```

2. No Relevant Documents
   ```python
   result = chain.run("Unrelated query")
   # Returns: "I don't have enough information to answer..."
   # LLM is NOT called, saves tokens
   ```

3. Retriever Failure
   ```python
   # Exception propagates from retriever
   try:
       result = chain.run("Query")
   except Exception as e:
       print(f"Retrieval failed: {e}")
   ```

4. LLM Failure
   ```python
   # Exception propagates from LLM
   try:
       result = chain.run("Query")
   except Exception as e:
       print(f"Generation failed: {e}")
   ```

==================================================================================

BEST PRACTICES
--------------

✅ DO:
  - Maintain chat history for multi-turn conversations
  - Check metadata['has_context'] before trusting answer
  - Display source documents to users for transparency
  - Use return_source_documents=False for token-saving applications
  - Validate queries before calling run()
  - Handle empty document cases gracefully

❌ DON'T:
  - Don't pass extremely long chat histories (summarize old turns)
  - Don't ignore metadata['has_context']=False responses
  - Don't modify returned Document objects
  - Don't assume answer is correct without checking sources
  - Don't call run() in tight loops without rate limiting

==================================================================================

DESIGN PATTERNS
---------------

FACADE PATTERN:
  RAGChain provides simple interface to complex workflow:
  - Retrieval (vector search, scoring, filtering)
  - Context formatting (document concatenation, source attribution)
  - Prompt engineering (grounding instructions, history integration)
  - LLM invocation (API calls, response parsing)
  
  All hidden behind single run() method

DEPENDENCY INJECTION:
  Chain accepts dependencies rather than creating them:
  - Retriever injected (not created internally)
  - LLM injected (not hard-coded)
  - Prompt manager optional (extensibility)
  
  Benefits: testability, flexibility, loose coupling

TEMPLATE METHOD:
  run() orchestrates workflow in fixed steps:
  1. Validate query
  2. Retrieve documents
  3. Format context
  4. Create prompt
  5. Generate answer
  6. Return response
  
  Subclasses can override individual steps

==================================================================================

ADVANCED USAGE
--------------

CUSTOM PROMPT TEMPLATES (future):
```python
from app.core.prompts import PromptManager

prompt_mgr = PromptManager()
chain = RAGChain(
    retriever=retriever,
    llm=llm,
    prompt_manager=prompt_mgr
)
# Will use custom prompts when implemented
```

CONVERSATION MEMORY (future):
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
chain = RAGChain(
    retriever=retriever,
    llm=llm,
    memory=memory
)
# Automatic history management
```

CUSTOM EXTENSIONS:
```python
class MyCustomChain(RAGChain):
    """Override _create_prompt for custom behavior."""
    
    def _create_prompt(self, query, context, chat_history=None):
        # Custom prompt logic
        return custom_prompt
```

==================================================================================

TESTING
-------

See tests/test_chain.py for comprehensive test suite:
  ✅ 30+ test cases
  ✅ Unit tests for all methods
  ✅ Integration tests for full workflow
  ✅ Edge case handling (empty query, no documents, etc.)
  ✅ Mock-based (fast, no external dependencies)

Run tests:
  pytest tests/test_chain.py -v

==================================================================================

DEMO
----

Run the demo to see chain in action:
  python demo_chain.py

Demos include:
  1. Basic question answering
  2. Chat history integration
  3. No relevant documents
  4. Without source documents
  5. Convenience function
  6. Context formatting
  7. Prompt structure
  8. Error handling

==================================================================================

TODO FEATURES
-------------

🔜 STREAMING RESPONSES:
   Stream answer tokens in real-time for better UX
   
   class StreamingRAGChain(RAGChain):
       async def run_stream(self, query):
           for chunk in self.llm.stream(prompt):
               yield chunk

🔜 INLINE CITATIONS:
   Add [1], [2] style citations to answer
   
   Example: "RAG combines retrieval [1] with generation [2]..."

🔜 CONFIDENCE SCORING:
   Estimate answer confidence based on:
   - Relevance scores of retrieved documents
   - Semantic similarity between query and answer
   - LLM uncertainty signals

🔜 CONVERSATIONAL MEMORY:
   Automatic history management with ConversationBufferMemory

🔜 MULTI-HOP RAG:
   Iterative retrieval for complex questions requiring multiple sources

🔜 ANSWER VERIFICATION:
   Second LLM call to verify faithfulness, relevance, completeness

==================================================================================

TROUBLESHOOTING
---------------

ISSUE: "Query cannot be empty"
SOLUTION: Validate query before calling run()
  if query and query.strip():
      result = chain.run(query)

ISSUE: "No relevant documents found" answers
SOLUTION: 
  - Check metadata['has_context'] in result
  - Lower score_threshold in retrieval
  - Increase k to retrieve more documents
  - Verify documents were properly indexed

ISSUE: Answers ignore sources
SOLUTION:
  - Grounding prompt should be working
  - Check if LLM is too powerful (might override instructions)
  - Try stronger grounding language in custom prompt

ISSUE: Slow response times
SOLUTION:
  - Reduce k (fewer documents to retrieve)
  - Use a faster model on Groq (smaller model, lower max tokens)
  - Cache frequently asked questions
  - Use return_source_documents=False to save tokens

==================================================================================

RESOURCES
---------

Code:
  - Implementation: app/rag/chain.py
  - Tests: tests/test_chain.py
  - Demo: demo_chain.py

Documentation:
  - This file: docs/CHAIN_QUICK_REF.md
  - Implementation details: docs/reference/CHAIN_IMPLEMENTATION_SUMMARY.md
  - Indexer docs: docs/INDEXER_QUICK_REF.md
  - Retriever docs: docs/RETRIEVER_QUICK_REF.md

Related Components:
  - RAGRetriever: app/rag/retriever.py
  - DocumentIndexer: app/rag/indexer.py
  - LLMFactory: app/core/llm.py
  - Settings: app/core/config.py

==================================================================================
```
