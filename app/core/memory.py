"""
Conversation Memory Manager

This module provides conversational memory management for multi-turn RAG
interactions. It maintains chat history per session, enabling context-aware
question answering across multiple user turns.

Key Components:
- ConversationMemoryManager: Main class for managing chat sessions
- Session-based memory storage (in-memory by default)
- LangChain ChatMessageHistory integration
- Simple public API for adding/retrieving messages

Design:
- Stateless interface (session_id passed explicitly)
- Stateful in-memory storage (dict of ChatMessageHistory)
- Compatible with LangChain RAG chains
- Easy to extend with persistent storage

TODO Features:
- Persistent storage (Redis, PostgreSQL, MongoDB)
- Memory windowing (keep last N messages)
- Automatic summarization (compress old history)
- Multi-user concurrency (thread-safe operations)
- Memory expiration (TTL for inactive sessions)
"""

from typing import Dict, List, Tuple, Optional
try:
    from langchain_community.chat_message_histories import ChatMessageHistory
except ImportError:  # pragma: no cover
    from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class ConversationMemoryManager:
    """
    Session-based conversation memory manager for RAG applications.
    
    This class manages chat history across multiple sessions, enabling
    multi-turn conversational RAG. Each session maintains its own
    independent ChatMessageHistory.
    
    Features:
    - Session-based memory isolation
    - LangChain ChatMessageHistory integration
    - Automatic session creation
    - Memory cleanup per session
    - Simple, stateless API
    
    Attributes:
        sessions: Dictionary mapping session_id to ChatMessageHistory
        max_sessions: Maximum number of concurrent sessions (LRU eviction)
    
    Example:
        >>> memory_mgr = ConversationMemoryManager()
        >>> memory_mgr.add_user_message("user123", "What is RAG?")
        >>> memory_mgr.add_ai_message("user123", "RAG stands for...")
        >>> history = memory_mgr.get_chat_history("user123")
        >>> print(history)  # [("What is RAG?", "RAG stands for...")]
    """
    
    def __init__(self, max_sessions: int = 1000):
        """
        Initialize the memory manager.
        
        Args:
            max_sessions: Maximum number of concurrent sessions to maintain.
                         When exceeded, oldest sessions are evicted (LRU).
        """
        self.sessions: Dict[str, ChatMessageHistory] = {}
        self.max_sessions = max_sessions
        self._access_order: List[str] = []  # Track access for LRU
    
    def get_memory(self, session_id: str) -> ChatMessageHistory:
        """
        Get or create ChatMessageHistory for a session.
        
        This is the primary method for retrieving session memory. If the
        session doesn't exist, it's automatically created. The returned
        ChatMessageHistory can be used directly with LangChain chains.
        
        Args:
            session_id: Unique identifier for the conversation session
                       (e.g., user_id, chat_id, or composite key)
        
        Returns:
            ChatMessageHistory instance for the session
        
        Example:
            >>> memory = memory_mgr.get_memory("user123")
            >>> memory.add_user_message("Hello!")
            >>> memory.add_ai_message("Hi there!")
        """
        if session_id not in self.sessions:
            # Create new session if doesn't exist
            self._create_session(session_id)
        
        # Update access order for LRU
        self._update_access(session_id)
        
        return self.sessions[session_id]
    
    def add_user_message(self, session_id: str, message: str) -> None:
        """
        Add a user message to the session history.
        
        Convenience method for adding user messages. Automatically creates
        session if it doesn't exist.
        
        Args:
            session_id: Session identifier
            message: User message content
        
        Example:
            >>> memory_mgr.add_user_message("user123", "What is machine learning?")
        """
        memory = self.get_memory(session_id)
        memory.add_user_message(message)
    
    def add_ai_message(self, session_id: str, message: str) -> None:
        """
        Add an AI response to the session history.
        
        Convenience method for adding AI messages. Should be called after
        generating a response to maintain conversation context.
        
        Args:
            session_id: Session identifier
            message: AI response content
        
        Example:
            >>> memory_mgr.add_ai_message("user123", "ML is a subset of AI...")
        """
        memory = self.get_memory(session_id)
        memory.add_ai_message(message)
    
    def clear_memory(self, session_id: str) -> None:
        """
        Clear all messages for a specific session.
        
        Removes all chat history for the session but keeps the session
        active. Use this to reset a conversation while maintaining the
        session.
        
        Args:
            session_id: Session identifier to clear
        
        Example:
            >>> memory_mgr.clear_memory("user123")  # Fresh start for user123
        """
        if session_id in self.sessions:
            self.sessions[session_id].clear()
    
    def delete_session(self, session_id: str) -> bool:
        """
        Completely remove a session from memory.
        
        Unlike clear_memory(), this removes the session entirely. Use this
        when a user logs out or a chat is permanently closed.
        
        Args:
            session_id: Session identifier to delete
        
        Returns:
            True if session was deleted, False if it didn't exist
        
        Example:
            >>> memory_mgr.delete_session("user123")
            True
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            if session_id in self._access_order:
                self._access_order.remove(session_id)
            return True
        return False
    
    def get_chat_history(
        self, 
        session_id: str, 
        as_messages: bool = False
    ) -> List[Tuple[str, str]] | List[BaseMessage]:
        """
        Retrieve chat history for a session in a convenient format.
        
        This method formats the chat history for easy use in RAG chains
        or for displaying to users.
        
        Args:
            session_id: Session identifier
            as_messages: If True, return LangChain message objects.
                        If False, return list of (human, ai) tuples.
        
        Returns:
            List of message tuples [(user_msg, ai_msg), ...] or
            List of BaseMessage objects [HumanMessage, AIMessage, ...]
        
        Example:
            >>> history = memory_mgr.get_chat_history("user123")
            >>> for user_msg, ai_msg in history:
            ...     print(f"User: {user_msg}")
            ...     print(f"AI: {ai_msg}")
        """
        memory = self.get_memory(session_id)
        messages = memory.messages
        
        if as_messages:
            return messages
        
        # Convert to (human, ai) tuple pairs
        history = []
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                human_msg = messages[i].content if isinstance(messages[i], HumanMessage) else ""
                ai_msg = messages[i + 1].content if isinstance(messages[i + 1], AIMessage) else ""
                history.append((human_msg, ai_msg))
        
        return history
    
    def get_recent_messages(
        self, 
        session_id: str, 
        n: int = 5
    ) -> List[BaseMessage]:
        """
        Get the N most recent messages from a session.
        
        Useful for implementing memory windowing or showing recent context.
        
        Args:
            session_id: Session identifier
            n: Number of recent messages to retrieve
        
        Returns:
            List of the N most recent BaseMessage objects
        
        Example:
            >>> recent = memory_mgr.get_recent_messages("user123", n=3)
            >>> for msg in recent:
            ...     print(f"{msg.__class__.__name__}: {msg.content}")
        """
        memory = self.get_memory(session_id)
        if n <= 0:
            return []
        return memory.messages[-n:] if len(memory.messages) >= n else memory.messages
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists in memory.
        
        Args:
            session_id: Session identifier to check
        
        Returns:
            True if session exists, False otherwise
        """
        return session_id in self.sessions
    
    def get_session_count(self) -> int:
        """
        Get the total number of active sessions.
        
        Returns:
            Number of sessions currently in memory
        """
        return len(self.sessions)
    
    def list_sessions(self) -> List[str]:
        """
        Get list of all active session IDs.
        
        Returns:
            List of session identifiers
        
        Example:
            >>> sessions = memory_mgr.list_sessions()
            >>> print(f"Active sessions: {sessions}")
        """
        return list(self.sessions.keys())
    
    def _create_session(self, session_id: str) -> None:
        """
        Internal method to create a new session.
        
        Implements LRU eviction when max_sessions is exceeded.
        
        Args:
            session_id: New session identifier
        """
        # Check if we need to evict oldest session (LRU)
        if len(self.sessions) >= self.max_sessions and session_id not in self.sessions:
            if self._access_order:
                oldest_session = self._access_order.pop(0)
                del self.sessions[oldest_session]
        
        # Create new session
        self.sessions[session_id] = ChatMessageHistory()
        self._access_order.append(session_id)
    
    def _update_access(self, session_id: str) -> None:
        """
        Update access order for LRU tracking.
        
        Args:
            session_id: Session that was accessed
        """
        if session_id in self._access_order:
            self._access_order.remove(session_id)
        self._access_order.append(session_id)


# Convenience factory function
def create_memory_manager(max_sessions: int = 1000) -> ConversationMemoryManager:
    """
    Factory function to create a ConversationMemoryManager instance.
    
    Args:
        max_sessions: Maximum number of concurrent sessions
    
    Returns:
        Configured ConversationMemoryManager instance
    
    Example:
        >>> memory_mgr = create_memory_manager(max_sessions=500)
        >>> memory_mgr.add_user_message("user123", "Hello!")
    """
    return ConversationMemoryManager(max_sessions=max_sessions)
#    - Example:
#      persistent_mgr = PersistentMemoryManager(backend="redis", url="redis://localhost")
#
# 2. WindowedMemoryManager(ConversationMemoryManager):
#    - Keep only last N messages per session
#    - Sliding window approach
#    - Configurable window size
#    - Example:
#      windowed_mgr = WindowedMemoryManager(window_size=10)
#
# 3. SummarizingMemoryManager(ConversationMemoryManager):
#    - Automatically summarize old messages
#    - Keep summary + recent messages
#    - Use LLM for intelligent summarization
#    - Example:
#      summary_mgr = SummarizingMemoryManager(llm=chat_model, summarize_after=20)
#
# 4. TTLMemoryManager(ConversationMemoryManager):
#    - Expire sessions after inactivity
#    - Configurable TTL per session or global
#    - Background cleanup task
#    - Example:
#      ttl_mgr = TTLMemoryManager(ttl_seconds=3600)
#
# 5. ThreadSafeMemoryManager(ConversationMemoryManager):
#    - Add locking for concurrent access
#    - Use threading.Lock or asyncio.Lock
#    - Safe for multi-threaded applications
#    - Example:
#      thread_safe_mgr = ThreadSafeMemoryManager()
#
# 6. AnalyticsMemoryManager(ConversationMemoryManager):
#    - Track session statistics (message count, duration, etc.)
#    - Export analytics data
#    - Monitor memory usage
#    - Example:
#      analytics_mgr = AnalyticsMemoryManager()
#      stats = analytics_mgr.get_session_stats("user123")
#
# 7. ExportableMemoryManager(ConversationMemoryManager):
#    - Export session to JSON, CSV, or other formats
#    - Import session from file
#    - Backup/restore functionality
#    - Example:
#      exportable_mgr = ExportableMemoryManager()
#      exportable_mgr.export_session("user123", "backup.json")
#
# 8. Integration with LangChain RunnableWithMessageHistory:
#    - Direct integration with LangChain's built-in memory support
#    - Example:
#      chain_with_history = RunnableWithMessageHistory(
#          chain,
#          lambda session_id: memory_mgr.get_memory(session_id),
#          input_messages_key="question",
#          history_messages_key="chat_history"
#      )

