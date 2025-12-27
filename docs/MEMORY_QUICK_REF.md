# Conversational Memory Manager - Quick Reference

## Overview

The `ConversationMemoryManager` provides session-based conversational memory for multi-turn RAG applications. It maintains independent chat history for each user/session, enabling context-aware question answering.

## Quick Start

```python
from app.core.memory import create_memory_manager

# Create memory manager
memory_mgr = create_memory_manager()

# Add messages for a session
session_id = "user123"
memory_mgr.add_user_message(session_id, "What is RAG?")
memory_mgr.add_ai_message(session_id, "RAG stands for Retrieval Augmented Generation...")

# Get chat history
history = memory_mgr.get_chat_history(session_id)
# [("What is RAG?", "RAG stands for...")]
```

## Core Features

### ✨ Key Capabilities

- **Session-Based Isolation**: Each user/chat has independent memory
- **LangChain Integration**: Uses `ChatMessageHistory` for compatibility
- **Automatic Management**: Sessions auto-create, LRU eviction for limits
- **Simple API**: Intuitive methods for common operations
- **Production-Ready**: Handles edge cases, Unicode, special characters

### 🎯 Design Principles

- **Stateless Interface**: `session_id` passed explicitly to all methods
- **Stateful Storage**: In-memory dict of `ChatMessageHistory` objects
- **Beginner-Friendly**: Clear naming, comprehensive docstrings
- **Interview-Ready**: Clean code, design patterns, extensibility

## API Reference

### Initialization

```python
# Default (max 1000 sessions)
memory_mgr = ConversationMemoryManager()

# Custom limit
memory_mgr = ConversationMemoryManager(max_sessions=500)

# Factory function
memory_mgr = create_memory_manager(max_sessions=500)
```

### Adding Messages

```python
# Add user message
memory_mgr.add_user_message(session_id, "Your question here")

# Add AI response
memory_mgr.add_ai_message(session_id, "AI response here")

# Direct access to ChatMessageHistory
memory = memory_mgr.get_memory(session_id)
memory.add_user_message("Direct message")
```

### Retrieving History

```python
# Get as (user, ai) tuples
history = memory_mgr.get_chat_history(session_id)
# [("Question 1", "Answer 1"), ("Question 2", "Answer 2")]

# Get as LangChain message objects
messages = memory_mgr.get_chat_history(session_id, as_messages=True)
# [HumanMessage, AIMessage, HumanMessage, AIMessage]

# Get recent messages
recent = memory_mgr.get_recent_messages(session_id, n=5)
# Last 5 messages

# Direct memory access
memory = memory_mgr.get_memory(session_id)
all_messages = memory.messages
```

### Session Management

```python
# Check if session exists
exists = memory_mgr.session_exists(session_id)

# List all active sessions
sessions = memory_mgr.list_sessions()
# ["user1", "user2", "user3"]

# Get session count
count = memory_mgr.get_session_count()

# Clear session memory (keeps session)
memory_mgr.clear_memory(session_id)

# Delete session completely
deleted = memory_mgr.delete_session(session_id)
```

## Integration with RAG Chain

### Pattern 1: Manual Integration

```python
from app.core.memory import create_memory_manager
from app.core.chain import create_rag_chain

memory_mgr = create_memory_manager()
rag_chain = create_rag_chain()

def process_query(session_id: str, query: str) -> str:
    # 1. Add user query to memory
    memory_mgr.add_user_message(session_id, query)
    
    # 2. Get chat history for context
    chat_history = memory_mgr.get_chat_history(session_id)
    
    # 3. Invoke RAG chain with history
    response = rag_chain.invoke({
        "question": query,
        "chat_history": chat_history
    })
    
    # 4. Add AI response to memory
    memory_mgr.add_ai_message(session_id, response["answer"])
    
    return response["answer"]
```

### Pattern 2: LangChain RunnableWithMessageHistory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create chain with built-in memory support
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: memory_mgr.get_memory(session_id),
    input_messages_key="question",
    history_messages_key="chat_history"
)

# Use it
response = chain_with_history.invoke(
    {"question": "What is RAG?"},
    config={"configurable": {"session_id": "user123"}}
)
```

## Examples

### Example 1: Basic Conversation

```python
memory_mgr = create_memory_manager()
session = "alice"

# Turn 1
memory_mgr.add_user_message(session, "What is machine learning?")
memory_mgr.add_ai_message(session, "Machine learning is a subset of AI...")

# Turn 2
memory_mgr.add_user_message(session, "Can you give an example?")
memory_mgr.add_ai_message(session, "Sure! Email spam detection is a common example...")

# View conversation
for user_msg, ai_msg in memory_mgr.get_chat_history(session):
    print(f"User: {user_msg}")
    print(f"AI: {ai_msg}")
    print()
```

### Example 2: Multiple Users

```python
memory_mgr = create_memory_manager()

# User 1's conversation
memory_mgr.add_user_message("alice", "Tell me about Python")
memory_mgr.add_ai_message("alice", "Python is a high-level programming language...")

# User 2's conversation
memory_mgr.add_user_message("bob", "What is JavaScript?")
memory_mgr.add_ai_message("bob", "JavaScript is a scripting language...")

# Sessions are isolated
print(f"Active sessions: {memory_mgr.list_sessions()}")
# ['alice', 'bob']
```

### Example 3: Session Cleanup

```python
memory_mgr = create_memory_manager()

# After user logs out
memory_mgr.delete_session("user123")

# To reset conversation (new topic)
memory_mgr.clear_memory("user456")

# Check before using
if not memory_mgr.session_exists("user789"):
    print("New user - starting fresh conversation")
```

## LRU Eviction

The memory manager automatically evicts least-recently-used sessions when `max_sessions` is exceeded:

```python
# Create with limit of 3 sessions
memory_mgr = ConversationMemoryManager(max_sessions=3)

# Create 4 sessions
for i in range(4):
    memory_mgr.add_user_message(f"user_{i}", "Hello")

# user_0 was evicted (oldest, least recently used)
print(memory_mgr.list_sessions())
# ['user_1', 'user_2', 'user_3']
```

## Future Enhancements (TODO)

The implementation includes TODO comments for future features:

1. **PersistentMemoryManager**: Redis, PostgreSQL, MongoDB backends
2. **WindowedMemoryManager**: Keep only last N messages per session
3. **SummarizingMemoryManager**: LLM-based conversation summarization
4. **TTLMemoryManager**: Auto-expire inactive sessions
5. **ThreadSafeMemoryManager**: Locking for concurrent access
6. **AnalyticsMemoryManager**: Session statistics and monitoring
7. **ExportableMemoryManager**: JSON/CSV export/import

## Best Practices

### ✅ Do

- Pass `session_id` consistently (user_id, chat_id, or composite)
- Clear memory when starting new conversation topics
- Delete sessions on user logout
- Use `get_memory()` for direct LangChain integration
- Handle multiple users with separate session IDs

### ❌ Don't

- Store sensitive data without encryption (add encryption layer)
- Use very long session IDs (impacts dict performance)
- Ignore LRU eviction (set appropriate `max_sessions`)
- Share session IDs between users (security risk)

## Testing

Run comprehensive tests:

```bash
pytest tests/test_memory.py -v
```

Test coverage includes:
- ✅ Initialization and configuration
- ✅ Message operations (add, retrieve)
- ✅ Session management (create, clear, delete)
- ✅ LRU eviction
- ✅ Multiple session isolation
- ✅ Edge cases (empty messages, Unicode, special chars)
- ✅ Integration patterns with RAG chains

## Demo

Run interactive demo:

```bash
python demo_memory.py
```

Demos include:
1. Basic usage
2. Multiple sessions
3. Session management
4. LRU eviction
5. RAG integration patterns

## Performance Characteristics

- **Time Complexity**:
  - Add message: O(1)
  - Get memory: O(1) average
  - List sessions: O(n) where n = session count
  - LRU eviction: O(n) for finding oldest

- **Space Complexity**: O(n × m) where:
  - n = number of sessions
  - m = average messages per session

- **Typical Usage**:
  - Small apps: 10-100 concurrent sessions
  - Medium apps: 100-1000 concurrent sessions
  - Large apps: Consider Redis backend (see TODO)

## Architecture

```
ConversationMemoryManager
├── sessions: Dict[str, ChatMessageHistory]
├── max_sessions: int
└── _access_order: List[str]

Methods:
├── get_memory(session_id) → ChatMessageHistory
├── add_user_message(session_id, message)
├── add_ai_message(session_id, message)
├── get_chat_history(session_id) → List[Tuple] | List[Message]
├── get_recent_messages(session_id, n) → List[Message]
├── clear_memory(session_id)
├── delete_session(session_id) → bool
├── session_exists(session_id) → bool
├── get_session_count() → int
└── list_sessions() → List[str]
```

## Related Files

- **Implementation**: `app/core/memory.py`
- **Tests**: `tests/test_memory.py`
- **Demo**: `demo_memory.py`
- **RAG Integration**: `app/core/chain.py`

## Support

For questions or issues:
1. Check demo: `python demo_memory.py`
2. Run tests: `pytest tests/test_memory.py -v`
3. Review docstrings in `app/core/memory.py`
4. Check TODO comments for future enhancements

---

**Status**: ✅ Production-ready for in-memory multi-user conversational RAG

**Version**: 1.0.0

**Last Updated**: 2024
