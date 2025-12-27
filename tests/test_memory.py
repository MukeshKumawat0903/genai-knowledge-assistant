"""
Tests for Conversation Memory Manager

This test suite validates the ConversationMemoryManager's functionality
including session management, message operations, and integration patterns.
"""

import pytest
from app.core.memory import ConversationMemoryManager, create_memory_manager
try:
    from langchain_community.chat_message_histories import ChatMessageHistory
except ImportError:  # pragma: no cover
    from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


class TestConversationMemoryManager:
    """Test suite for ConversationMemoryManager class."""
    
    def test_initialization(self):
        """Test memory manager initialization."""
        memory_mgr = ConversationMemoryManager()
        
        assert memory_mgr.sessions == {}
        assert memory_mgr.max_sessions == 1000
        assert memory_mgr._access_order == []
    
    def test_initialization_custom_max_sessions(self):
        """Test initialization with custom max_sessions."""
        memory_mgr = ConversationMemoryManager(max_sessions=50)
        
        assert memory_mgr.max_sessions == 50
    
    def test_get_memory_creates_new_session(self):
        """Test that get_memory creates a new session if it doesn't exist."""
        memory_mgr = ConversationMemoryManager()
        
        session_id = "test_session"
        memory = memory_mgr.get_memory(session_id)
        
        assert isinstance(memory, ChatMessageHistory)
        assert session_id in memory_mgr.sessions
        assert session_id in memory_mgr._access_order
    
    def test_get_memory_returns_existing_session(self):
        """Test that get_memory returns existing session."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_session"
        
        # Get memory twice
        memory1 = memory_mgr.get_memory(session_id)
        memory2 = memory_mgr.get_memory(session_id)
        
        # Should be the same instance
        assert memory1 is memory2
        assert len(memory_mgr.sessions) == 1
    
    def test_add_user_message(self):
        """Test adding user messages."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_user"
        
        memory_mgr.add_user_message(session_id, "Hello!")
        
        memory = memory_mgr.get_memory(session_id)
        assert len(memory.messages) == 1
        assert isinstance(memory.messages[0], HumanMessage)
        assert memory.messages[0].content == "Hello!"
    
    def test_add_ai_message(self):
        """Test adding AI messages."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_user"
        
        memory_mgr.add_ai_message(session_id, "Hi there!")
        
        memory = memory_mgr.get_memory(session_id)
        assert len(memory.messages) == 1
        assert isinstance(memory.messages[0], AIMessage)
        assert memory.messages[0].content == "Hi there!"
    
    def test_conversation_flow(self):
        """Test a complete conversation flow."""
        memory_mgr = ConversationMemoryManager()
        session_id = "conversation_test"
        
        # Add multiple turns
        memory_mgr.add_user_message(session_id, "What is AI?")
        memory_mgr.add_ai_message(session_id, "AI stands for Artificial Intelligence.")
        memory_mgr.add_user_message(session_id, "Tell me more")
        memory_mgr.add_ai_message(session_id, "AI is the simulation of human intelligence...")
        
        memory = memory_mgr.get_memory(session_id)
        assert len(memory.messages) == 4
        
        # Verify order and types
        assert isinstance(memory.messages[0], HumanMessage)
        assert isinstance(memory.messages[1], AIMessage)
        assert isinstance(memory.messages[2], HumanMessage)
        assert isinstance(memory.messages[3], AIMessage)
    
    def test_clear_memory(self):
        """Test clearing session memory."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_clear"
        
        # Add messages
        memory_mgr.add_user_message(session_id, "Message 1")
        memory_mgr.add_ai_message(session_id, "Response 1")
        
        # Clear
        memory_mgr.clear_memory(session_id)
        
        # Session should still exist but be empty
        assert session_id in memory_mgr.sessions
        assert len(memory_mgr.get_memory(session_id).messages) == 0
    
    def test_clear_nonexistent_session(self):
        """Test clearing a session that doesn't exist (should not raise error)."""
        memory_mgr = ConversationMemoryManager()
        
        # Should not raise an error
        memory_mgr.clear_memory("nonexistent_session")
    
    def test_delete_session(self):
        """Test completely deleting a session."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_delete"
        
        # Create session
        memory_mgr.add_user_message(session_id, "Test")
        assert session_id in memory_mgr.sessions
        
        # Delete
        result = memory_mgr.delete_session(session_id)
        
        assert result is True
        assert session_id not in memory_mgr.sessions
        assert session_id not in memory_mgr._access_order
    
    def test_delete_nonexistent_session(self):
        """Test deleting a session that doesn't exist."""
        memory_mgr = ConversationMemoryManager()
        
        result = memory_mgr.delete_session("nonexistent_session")
        
        assert result is False
    
    def test_get_chat_history_tuples(self):
        """Test getting chat history as tuples."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_history"
        
        # Add conversation
        memory_mgr.add_user_message(session_id, "Question 1")
        memory_mgr.add_ai_message(session_id, "Answer 1")
        memory_mgr.add_user_message(session_id, "Question 2")
        memory_mgr.add_ai_message(session_id, "Answer 2")
        
        # Get history as tuples (default)
        history = memory_mgr.get_chat_history(session_id)
        
        assert len(history) == 2
        assert history[0] == ("Question 1", "Answer 1")
        assert history[1] == ("Question 2", "Answer 2")
    
    def test_get_chat_history_messages(self):
        """Test getting chat history as message objects."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_messages"
        
        # Add messages
        memory_mgr.add_user_message(session_id, "Hello")
        memory_mgr.add_ai_message(session_id, "Hi")
        
        # Get as messages
        history = memory_mgr.get_chat_history(session_id, as_messages=True)
        
        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)
        assert history[0].content == "Hello"
        assert history[1].content == "Hi"
    
    def test_get_recent_messages(self):
        """Test getting recent messages."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_recent"
        
        # Add 6 messages
        for i in range(3):
            memory_mgr.add_user_message(session_id, f"User {i}")
            memory_mgr.add_ai_message(session_id, f"AI {i}")
        
        # Get last 4 messages
        recent = memory_mgr.get_recent_messages(session_id, n=4)
        
        assert len(recent) == 4
        assert recent[-1].content == "AI 2"  # Most recent
        assert recent[0].content == "User 1"  # 4th from end
    
    def test_get_recent_messages_exceeds_total(self):
        """Test getting more recent messages than exist."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_exceed"
        
        # Add only 2 messages
        memory_mgr.add_user_message(session_id, "User")
        memory_mgr.add_ai_message(session_id, "AI")
        
        # Request 10 messages
        recent = memory_mgr.get_recent_messages(session_id, n=10)
        
        # Should return all available messages
        assert len(recent) == 2
    
    def test_session_exists(self):
        """Test checking if session exists."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_exists"
        
        assert not memory_mgr.session_exists(session_id)
        
        memory_mgr.add_user_message(session_id, "Test")
        
        assert memory_mgr.session_exists(session_id)
    
    def test_get_session_count(self):
        """Test getting total session count."""
        memory_mgr = ConversationMemoryManager()
        
        assert memory_mgr.get_session_count() == 0
        
        memory_mgr.add_user_message("user1", "Test")
        assert memory_mgr.get_session_count() == 1
        
        memory_mgr.add_user_message("user2", "Test")
        assert memory_mgr.get_session_count() == 2
        
        memory_mgr.delete_session("user1")
        assert memory_mgr.get_session_count() == 1
    
    def test_list_sessions(self):
        """Test listing all session IDs."""
        memory_mgr = ConversationMemoryManager()
        
        assert memory_mgr.list_sessions() == []
        
        sessions = ["alice", "bob", "charlie"]
        for session in sessions:
            memory_mgr.add_user_message(session, "Test")
        
        active = memory_mgr.list_sessions()
        assert set(active) == set(sessions)
    
    def test_lru_eviction(self):
        """Test LRU eviction when max_sessions is exceeded."""
        memory_mgr = ConversationMemoryManager(max_sessions=3)
        
        # Create 3 sessions (at limit)
        for i in range(3):
            memory_mgr.add_user_message(f"user_{i}", f"Message {i}")
        
        assert memory_mgr.get_session_count() == 3
        assert "user_0" in memory_mgr.sessions
        
        # Create 4th session (should evict user_0)
        memory_mgr.add_user_message("user_3", "Message 3")
        
        assert memory_mgr.get_session_count() == 3
        assert "user_0" not in memory_mgr.sessions
        assert "user_3" in memory_mgr.sessions
    
    def test_lru_access_order_update(self):
        """Test that accessing a session updates LRU order."""
        memory_mgr = ConversationMemoryManager(max_sessions=3)
        
        # Create 3 sessions
        memory_mgr.add_user_message("user_0", "Test")
        memory_mgr.add_user_message("user_1", "Test")
        memory_mgr.add_user_message("user_2", "Test")
        
        # Access user_0 (should move to end of access order)
        memory_mgr.get_memory("user_0")
        
        # Create new session (should evict user_1, not user_0)
        memory_mgr.add_user_message("user_3", "Test")
        
        assert "user_0" in memory_mgr.sessions
        assert "user_1" not in memory_mgr.sessions
        assert "user_2" in memory_mgr.sessions
        assert "user_3" in memory_mgr.sessions
    
    def test_multiple_sessions_isolation(self):
        """Test that sessions are isolated from each other."""
        memory_mgr = ConversationMemoryManager()
        
        # Add different messages to different sessions
        memory_mgr.add_user_message("alice", "Alice's message")
        memory_mgr.add_user_message("bob", "Bob's message")
        
        alice_history = memory_mgr.get_chat_history("alice", as_messages=True)
        bob_history = memory_mgr.get_chat_history("bob", as_messages=True)
        
        assert len(alice_history) == 1
        assert len(bob_history) == 1
        assert alice_history[0].content == "Alice's message"
        assert bob_history[0].content == "Bob's message"


class TestCreateMemoryManager:
    """Test suite for create_memory_manager factory function."""
    
    def test_create_default(self):
        """Test creating memory manager with default settings."""
        memory_mgr = create_memory_manager()
        
        assert isinstance(memory_mgr, ConversationMemoryManager)
        assert memory_mgr.max_sessions == 1000
    
    def test_create_custom_max_sessions(self):
        """Test creating memory manager with custom max_sessions."""
        memory_mgr = create_memory_manager(max_sessions=500)
        
        assert isinstance(memory_mgr, ConversationMemoryManager)
        assert memory_mgr.max_sessions == 500


class TestEdgeCases:
    """Test suite for edge cases and error handling."""
    
    def test_empty_message(self):
        """Test adding empty messages."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_empty"
        
        memory_mgr.add_user_message(session_id, "")
        memory_mgr.add_ai_message(session_id, "")
        
        memory = memory_mgr.get_memory(session_id)
        assert len(memory.messages) == 2
        assert memory.messages[0].content == ""
        assert memory.messages[1].content == ""
    
    def test_very_long_message(self):
        """Test handling very long messages."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_long"
        
        long_message = "A" * 10000
        memory_mgr.add_user_message(session_id, long_message)
        
        memory = memory_mgr.get_memory(session_id)
        assert memory.messages[0].content == long_message
    
    def test_special_characters_in_session_id(self):
        """Test session IDs with special characters."""
        memory_mgr = ConversationMemoryManager()
        
        session_ids = [
            "user@example.com",
            "user-123",
            "user_test_2024",
            "session:abc:123"
        ]
        
        for session_id in session_ids:
            memory_mgr.add_user_message(session_id, "Test")
            assert memory_mgr.session_exists(session_id)
    
    def test_unicode_messages(self):
        """Test handling Unicode characters in messages."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_unicode"
        
        unicode_messages = [
            "Hello 世界",
            "مرحبا",
            "Привет",
            "🚀 🎉 ✨"
        ]
        
        for msg in unicode_messages:
            memory_mgr.add_user_message(session_id, msg)
        
        memory = memory_mgr.get_memory(session_id)
        for i, msg in enumerate(unicode_messages):
            assert memory.messages[i].content == msg
    
    def test_get_recent_messages_zero(self):
        """Test getting 0 recent messages."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_zero"
        
        memory_mgr.add_user_message(session_id, "Test")
        recent = memory_mgr.get_recent_messages(session_id, n=0)
        
        assert len(recent) == 0
    
    def test_multiple_clears(self):
        """Test clearing the same session multiple times."""
        memory_mgr = ConversationMemoryManager()
        session_id = "test_multi_clear"
        
        memory_mgr.add_user_message(session_id, "Test")
        memory_mgr.clear_memory(session_id)
        memory_mgr.clear_memory(session_id)
        memory_mgr.clear_memory(session_id)
        
        # Should not raise error
        assert len(memory_mgr.get_memory(session_id).messages) == 0


class TestIntegrationPatterns:
    """Test patterns for integrating with RAG chains."""
    
    def test_rag_conversation_pattern(self):
        """Test typical RAG conversation pattern."""
        memory_mgr = ConversationMemoryManager()
        session_id = "rag_user"
        
        # Simulate RAG conversation flow
        # Turn 1
        memory_mgr.add_user_message(session_id, "What is machine learning?")
        # RAG would retrieve context and generate response here
        memory_mgr.add_ai_message(session_id, "Machine learning is...")
        
        # Turn 2 - with context from previous turn
        memory_mgr.add_user_message(session_id, "Can you explain more?")
        history = memory_mgr.get_memory(session_id)
        
        # Verify context is available for RAG chain
        assert len(history.messages) == 3
        assert history.messages[0].content == "What is machine learning?"
        
        memory_mgr.add_ai_message(session_id, "Sure, machine learning involves...")
        
        # Final verification
        assert len(memory_mgr.get_memory(session_id).messages) == 4
    
    def test_langchain_runnable_integration(self):
        """Test integration pattern with LangChain RunnableWithMessageHistory."""
        memory_mgr = ConversationMemoryManager()
        
        # This is how you'd pass memory to RunnableWithMessageHistory
        session_id = "langchain_user"
        
        # Setup conversation
        memory_mgr.add_user_message(session_id, "Hello")
        memory_mgr.add_ai_message(session_id, "Hi!")
        
        # Get memory callable (for RunnableWithMessageHistory)
        get_session_history = lambda sid: memory_mgr.get_memory(sid)
        
        # Verify it works
        history = get_session_history(session_id)
        assert len(history.messages) == 2
        
        # Simulate adding more messages through the chain
        get_session_history(session_id).add_user_message("How are you?")
        assert len(memory_mgr.get_memory(session_id).messages) == 3


# Pytest fixtures
@pytest.fixture
def memory_manager():
    """Fixture providing a fresh memory manager for each test."""
    return ConversationMemoryManager()


@pytest.fixture
def populated_memory_manager():
    """Fixture providing a memory manager with pre-populated sessions."""
    mgr = ConversationMemoryManager()
    
    # Add some test data
    mgr.add_user_message("user1", "Hello from user1")
    mgr.add_ai_message("user1", "Hi user1!")
    
    mgr.add_user_message("user2", "Hello from user2")
    mgr.add_ai_message("user2", "Hi user2!")
    
    return mgr


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
