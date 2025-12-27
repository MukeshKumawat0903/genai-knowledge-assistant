"""
Conversation Memory Manager Demo

This demo script shows how to use the ConversationMemoryManager
for managing multi-turn conversations in a RAG application.

Features demonstrated:
1. Creating a memory manager
2. Adding user and AI messages
3. Retrieving chat history
4. Managing multiple sessions
5. Clearing session memory
6. Session management (list, delete, check)
"""

from app.core.memory import ConversationMemoryManager, create_memory_manager


def demo_basic_usage():
    """
    Demo: Basic memory operations
    
    Shows how to:
    - Create a memory manager
    - Add messages to a session
    - Retrieve chat history
    """
    print("=" * 60)
    print("DEMO 1: Basic Usage")
    print("=" * 60)
    
    # Create memory manager
    memory_mgr = create_memory_manager()
    
    # Add conversation turns for session "user123"
    session_id = "user123"
    
    print(f"\n📝 Adding messages to session: {session_id}")
    memory_mgr.add_user_message(session_id, "What is RAG?")
    memory_mgr.add_ai_message(
        session_id, 
        "RAG stands for Retrieval Augmented Generation. It's a technique that enhances LLM responses by retrieving relevant context from a knowledge base."
    )
    
    memory_mgr.add_user_message(session_id, "How does it work?")
    memory_mgr.add_ai_message(
        session_id,
        "RAG works by: 1) Indexing documents into a vector database, 2) Retrieving relevant chunks based on the query, 3) Augmenting the LLM prompt with retrieved context."
    )
    
    # Get chat history as tuples
    print(f"\n💬 Chat History for {session_id}:")
    history = memory_mgr.get_chat_history(session_id)
    for i, (user_msg, ai_msg) in enumerate(history, 1):
        print(f"\nTurn {i}:")
        print(f"  User: {user_msg}")
        print(f"  AI: {ai_msg}")
    
    # Get recent messages
    print(f"\n🔍 Last 2 messages:")
    recent = memory_mgr.get_recent_messages(session_id, n=2)
    for msg in recent:
        role = "User" if msg.__class__.__name__ == "HumanMessage" else "AI"
        print(f"  {role}: {msg.content[:50]}...")
    
    print("\n✅ Basic usage demo complete!")


def demo_multiple_sessions():
    """
    Demo: Managing multiple concurrent sessions
    
    Shows how to:
    - Manage multiple independent sessions
    - Session isolation (no cross-talk)
    - List all active sessions
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Multiple Sessions")
    print("=" * 60)
    
    memory_mgr = create_memory_manager()
    
    # Create conversations for different users
    sessions = ["alice", "bob", "charlie"]
    
    print("\n📝 Creating conversations for multiple users:")
    for session in sessions:
        memory_mgr.add_user_message(session, f"Hello from {session}!")
        memory_mgr.add_ai_message(session, f"Hi {session}! How can I help you today?")
    
    # Show all active sessions
    print(f"\n📊 Active sessions: {memory_mgr.list_sessions()}")
    print(f"Total sessions: {memory_mgr.get_session_count()}")
    
    # Show each session's history
    print("\n💬 Individual session histories:")
    for session in sessions:
        history = memory_mgr.get_chat_history(session)
        print(f"\n  {session}:")
        for user_msg, ai_msg in history:
            print(f"    User: {user_msg}")
            print(f"    AI: {ai_msg}")
    
    print("\n✅ Multiple sessions demo complete!")


def demo_session_management():
    """
    Demo: Session management operations
    
    Shows how to:
    - Check if session exists
    - Clear session memory
    - Delete sessions completely
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Session Management")
    print("=" * 60)
    
    memory_mgr = create_memory_manager()
    
    # Create a session
    session_id = "test_user"
    memory_mgr.add_user_message(session_id, "First message")
    memory_mgr.add_ai_message(session_id, "First response")
    
    print(f"\n📝 Created session: {session_id}")
    print(f"Session exists? {memory_mgr.session_exists(session_id)}")
    print(f"Message count: {len(memory_mgr.get_memory(session_id).messages)}")
    
    # Clear memory (keeps session)
    print(f"\n🧹 Clearing session memory...")
    memory_mgr.clear_memory(session_id)
    print(f"Session exists? {memory_mgr.session_exists(session_id)}")
    print(f"Message count: {len(memory_mgr.get_memory(session_id).messages)}")
    
    # Add new messages after clearing
    memory_mgr.add_user_message(session_id, "Starting fresh!")
    memory_mgr.add_ai_message(session_id, "Yes, let's start over!")
    print(f"Message count after adding new: {len(memory_mgr.get_memory(session_id).messages)}")
    
    # Delete session completely
    print(f"\n🗑️ Deleting session...")
    deleted = memory_mgr.delete_session(session_id)
    print(f"Session deleted? {deleted}")
    print(f"Session exists? {memory_mgr.session_exists(session_id)}")
    
    print("\n✅ Session management demo complete!")


def demo_lru_eviction():
    """
    Demo: LRU (Least Recently Used) session eviction
    
    Shows how the memory manager automatically evicts old sessions
    when max_sessions limit is reached.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: LRU Eviction")
    print("=" * 60)
    
    # Create manager with small session limit
    memory_mgr = ConversationMemoryManager(max_sessions=3)
    
    print("\n📝 Creating 5 sessions (max_sessions=3):")
    
    # Create more sessions than allowed
    for i in range(5):
        session_id = f"user_{i}"
        memory_mgr.add_user_message(session_id, f"Message from user {i}")
        memory_mgr.add_ai_message(session_id, f"Response to user {i}")
        
        active = memory_mgr.list_sessions()
        print(f"\n  After creating {session_id}:")
        print(f"    Active sessions: {active}")
        print(f"    Count: {memory_mgr.get_session_count()}")
    
    print("\n📊 Final active sessions:")
    print(f"  {memory_mgr.list_sessions()}")
    print(f"  Note: Only the 3 most recent sessions are kept!")
    
    print("\n✅ LRU eviction demo complete!")


def demo_integration_with_rag():
    """
    Demo: Integration with RAG chain
    
    Shows how to use ConversationMemoryManager with a RAG chain
    for context-aware question answering.
    """
    print("\n" + "=" * 60)
    print("DEMO 5: RAG Integration Pattern")
    print("=" * 60)
    
    memory_mgr = create_memory_manager()
    session_id = "user_demo"
    
    print("\n📝 Simulating multi-turn RAG conversation:")
    
    # Simulate a multi-turn RAG conversation
    conversations = [
        {
            "query": "What is machine learning?",
            "retrieved_context": "Machine learning is a subset of AI that enables systems to learn from data...",
            "response": "Machine learning is a subset of artificial intelligence that allows computers to learn and improve from experience without being explicitly programmed."
        },
        {
            "query": "Can you give me an example?",
            "retrieved_context": "Examples include: image recognition, spam detection, recommendation systems...",
            "response": "Sure! A common example is email spam detection. The system learns from millions of emails labeled as spam or not spam, and then can automatically identify spam in new emails."
        },
        {
            "query": "How is it different from traditional programming?",
            "retrieved_context": "Traditional programming uses explicit rules, ML learns patterns from data...",
            "response": "In traditional programming, you write explicit rules. In machine learning, the system discovers patterns and rules from data automatically. It's learning by example rather than by instruction."
        }
    ]
    
    for i, turn in enumerate(conversations, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User Query: {turn['query']}")
        
        # Add user message to memory
        memory_mgr.add_user_message(session_id, turn['query'])
        
        # Get chat history for context (this would be used by RAG chain)
        history = memory_mgr.get_chat_history(session_id, as_messages=True)
        print(f"Context length: {len(history)} messages")
        
        # Simulate RAG response (in real app, this would call RAG chain)
        print(f"Retrieved Context: {turn['retrieved_context'][:60]}...")
        print(f"AI Response: {turn['response']}")
        
        # Add AI response to memory
        memory_mgr.add_ai_message(session_id, turn['response'])
    
    print(f"\n📊 Full conversation summary:")
    print(f"Total turns: {len(memory_mgr.get_chat_history(session_id))}")
    print(f"Total messages: {len(memory_mgr.get_memory(session_id).messages)}")
    
    print("\n💡 Integration pattern:")
    print("  1. User sends query")
    print("  2. Add query to memory: memory_mgr.add_user_message(session_id, query)")
    print("  3. Get history: memory_mgr.get_memory(session_id)")
    print("  4. Pass to RAG chain with history context")
    print("  5. Add response to memory: memory_mgr.add_ai_message(session_id, response)")
    print("  6. Return response to user")
    
    print("\n✅ RAG integration demo complete!")


def main():
    """Run all demos."""
    print("\n" + "🚀" * 30)
    print("CONVERSATION MEMORY MANAGER - COMPREHENSIVE DEMO")
    print("🚀" * 30)
    
    try:
        # Run all demos
        demo_basic_usage()
        demo_multiple_sessions()
        demo_session_management()
        demo_lru_eviction()
        demo_integration_with_rag()
        
        print("\n" + "=" * 60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📚 Key Takeaways:")
        print("  ✓ Session-based memory isolation")
        print("  ✓ Simple, intuitive API")
        print("  ✓ Automatic session management")
        print("  ✓ LRU eviction for resource control")
        print("  ✓ Easy integration with RAG chains")
        print("  ✓ Production-ready for multi-user systems")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
