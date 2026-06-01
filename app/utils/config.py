"""
Configuration Manager

This module handles environment variables and application configuration.
It loads settings from .env files and provides a centralized config interface.

Features:
- Environment variable loading with python-dotenv
- Configuration validation
- Safe defaults
- Type conversion and type safety
- Singleton pattern for global access

Usage:
    from app.utils.config import get_settings
    
    settings = get_settings()
    print(settings.llm_provider)
    print(settings.llm_model_name)
"""

import os
from typing import Optional, Literal, Dict, List
from pathlib import Path
from dotenv import load_dotenv


class Settings:
    """
    Centralized configuration class for the GenAI Knowledge Assistant.
    
    Loads all configuration from environment variables with sensible defaults.
    All settings are immutable after initialization to prevent accidental changes.
    
    Environment Variables:
        See .env.example for all available configuration options.
    """
    
    def __init__(self):
        """
        Initialize settings by loading environment variables.
        
        Automatically loads .env file if present, then reads environment variables
        with fallback to safe defaults where appropriate.
        """
        # Load .env file if it exists (does nothing if file not found)
        load_dotenv()
        
        # ========================================================================
        # Application Settings
        # ========================================================================
        
        self.app_name: str = os.getenv("APP_NAME", "GenAI Knowledge Assistant")
        """Application name for logging and display"""
        
        self.environment: Literal["dev", "prod"] = os.getenv("ENVIRONMENT", "dev")  # type: ignore
        """Application environment: 'dev' for development, 'prod' for production"""
        
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        """Enable debug mode with verbose logging and error traces"""
        
        # ========================================================================
        # LLM Configuration
        # ========================================================================
        # Environment variables: LLM_PROVIDER, LLM_MODEL_NAME, LLM_TEMPERATURE, etc.
        
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "groq").lower()
        """LLM provider: 'groq', 'openai', 'anthropic', or 'google'."""

        # Available models per provider (config-driven, no hardcoding in UI)
        self.available_models: Dict[str, List[str]] = {
            "groq": [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            "openai": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ],
            "anthropic": [
                "claude-sonnet-4-6",
                "claude-opus-4-8",
                "claude-haiku-4-5-20251001",
            ],
            "google": [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-pro",
            ],
        }

        self.llm_model_name: str = os.getenv(
            "LLM_MODEL_NAME",
            self._get_default_model_name()
        )
        """
        Specific model to use (e.g., 'gpt-3.5-turbo', 'claude-3-sonnet', 'llama3-8b')
        Defaults based on selected provider for convenience.
        """
        
        self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        """
        Sampling temperature for LLM (0.0 to 1.0)
        Lower = more focused/deterministic, Higher = more creative/random
        """
        
        self.llm_max_tokens: Optional[int] = self._get_optional_int("LLM_MAX_TOKENS")
        """
        Maximum tokens to generate in LLM response
        None means use model's default. Useful for cost control.
        """
        
        # ========================================================================
        # API Keys
        # ========================================================================
        self.groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
        """Groq API key for fast inference"""

        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        """OpenAI API key for GPT-4o and related models"""

        self.anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        """Anthropic API key for Claude models"""

        self.google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        """Google API key for Gemini models"""
        
        # ========================================================================
        # Embedding Configuration
        # ========================================================================
        # Environment variables: EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME
        
        self.embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
        """
        Embedding provider to use. Options: 'openai', 'huggingface', 'sentence-transformers'
        Used for converting text to vector representations for semantic search.
        """
        
        self.embedding_model_name: str = os.getenv(
            "EMBEDDING_MODEL_NAME",
            "all-MiniLM-L6-v2"  # HuggingFace sentence-transformer model
        )
        """
        Embedding model to use (e.g., 'text-embedding-ada-002', 'all-MiniLM-L6-v2')
        Must be compatible with the selected embedding provider.
        """
        
        # ========================================================================
        # Vector Store Configuration
        # ========================================================================
        # Environment variables: VECTOR_STORE_TYPE, VECTOR_STORE_PATH, etc.
        
        self.vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "faiss").lower()
        """
        Vector database type. Options: 'faiss', 'chroma', 'pinecone', 'qdrant'
        FAISS is default (local, fast). Pinecone/Qdrant for cloud deployments.
        """
        
        self.vector_store_path: Path = Path(
            os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
        )
        """
        Local path for vector store persistence (FAISS, Chroma)
        Ignored for cloud-based stores like Pinecone.
        """
        
        self.collection_name: str = os.getenv("COLLECTION_NAME", "documents")
        """
        Collection/index name in vector store
        Used to organize different document sets.
        """
        
        # ========================================================================
        # RAG (Retrieval-Augmented Generation) Configuration
        # ========================================================================
        # Environment variables: CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_TOP_K
        
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
        """
        Size of text chunks for document splitting (in characters)
        Larger chunks = more context per chunk, but less precise retrieval.
        Typical range: 500-2000
        """
        
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        """
        Overlap between consecutive chunks (in characters)
        Prevents information loss at chunk boundaries.
        Typically 10-20% of chunk_size.
        """
        
        self.retriever_top_k: int = int(os.getenv("RETRIEVER_TOP_K", "4"))
        """
        Number of most relevant chunks to retrieve for each query
        More chunks = more context but higher token cost.
        Typical range: 3-10
        """
        
        self.similarity_threshold: float = float(
            os.getenv("SIMILARITY_THRESHOLD", "0.0")
        )
        """
        Minimum similarity score for retrieved chunks (0.0 to 1.0)
        0.0 = no filtering, higher values = stricter relevance requirement.
        """
        
        # ========================================================================
        # Agent Configuration
        # ========================================================================
        # Environment variables: AGENT_MAX_ITERATIONS, ENABLE_AGENT_TOOLS
        
        self.agent_max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
        """
        Maximum reasoning iterations for ReAct agents
        Prevents infinite loops while allowing multi-step reasoning.
        """
        
        self.enable_agent_tools: bool = os.getenv(
            "ENABLE_AGENT_TOOLS", "true"
        ).lower() == "true"
        """
        Enable agent tools (web search, calculator, etc.)
        Disable for pure RAG mode without external tools.
        """
        
        # ========================================================================
        # Data Paths
        # ========================================================================
        
        self.data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
        """Root directory for all data storage"""
        
        self.raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", "./data/raw"))
        """Directory for raw, unprocessed data files"""
        
        self.processed_data_dir: Path = Path(
            os.getenv("PROCESSED_DATA_DIR", "./data/processed")
        )
        """Directory for processed and indexed data"""
        
        # ========================================================================
        # Validation
        # ========================================================================
        # Validate configuration after loading all values
        self._validate()
    
    def _get_default_model_name(self) -> str:
        """Get default model name for the configured LLM provider."""
        defaults = {
            "groq": "llama-3.3-70b-versatile",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-6",
            "google": "gemini-1.5-flash",
        }
        return defaults.get(self.llm_provider, "llama-3.3-70b-versatile")
    
    def _get_optional_int(self, key: str) -> Optional[int]:
        """
        Get optional integer from environment.
        
        Args:
            key: Environment variable name
            
        Returns:
            Integer value or None if not set
        """
        value = os.getenv(key)
        return int(value) if value else None
    
    def _validate(self) -> None:
        """Validate configuration values."""
        # Provider
        valid_llm_providers = ["groq", "openai", "anthropic", "google"]
        if self.llm_provider not in valid_llm_providers:
            raise ValueError(
                f"Invalid LLM provider: '{self.llm_provider}'. "
                f"Must be one of: {', '.join(valid_llm_providers)}"
            )

        # Require the API key only for the configured provider
        _provider_key_map = {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
        }
        _env_var_map = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        if not _provider_key_map[self.llm_provider]:
            raise ValueError(
                f"Missing API key for '{self.llm_provider}'. "
                f"Please set {_env_var_map[self.llm_provider]} in your .env file."
            )

        # Vector store
        valid_vector_stores = ["faiss", "chroma", "pinecone"]
        if self.vector_store_type not in valid_vector_stores:
            raise ValueError(
                f"Invalid vector store type: '{self.vector_store_type}'. "
                f"Must be one of: {', '.join(valid_vector_stores)}"
            )

        # Numeric ranges
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError(
                f"LLM temperature must be between 0.0 and 2.0, got {self.llm_temperature}"
            )

        if self.chunk_size < 100:
            raise ValueError(f"Chunk size must be at least 100, got {self.chunk_size}")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"Chunk overlap ({self.chunk_overlap}) must be less than "
                f"chunk size ({self.chunk_size})"
            )

        if self.retriever_top_k < 1:
            raise ValueError(f"Retriever top_k must be at least 1, got {self.retriever_top_k}")
    
    def get_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        """Get API key for the specified provider (defaults to current provider)."""
        provider = (provider or self.llm_provider).lower()
        key_map = {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
        }
        return key_map.get(provider)
    
    def __repr__(self) -> str:
        """String representation of settings (without exposing API keys)."""
        return (
            f"Settings("
            f"llm_provider='{self.llm_provider}', "
            f"llm_model='{self.llm_model_name}', "
            f"vector_store='{self.vector_store_type}', "
            f"chunk_size={self.chunk_size}, "
            f"environment='{self.environment}'"
            f")"
        )


# ============================================================================
# Singleton Pattern - Global Settings Instance
# ============================================================================

_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global Settings instance (singleton pattern).
    
    This function ensures only one Settings object is created and reused
    throughout the application, preventing redundant .env file loading.
    
    Returns:
        Settings: The global settings instance
        
    Example:
        >>> from app.utils.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.llm_provider)
        'openai'
        >>> print(settings.chunk_size)
        1000
    """
    global _settings_instance
    
    if _settings_instance is None:
        _settings_instance = Settings()
    
    return _settings_instance


def reload_settings() -> Settings:
    """
    Force reload of settings from environment.
    
    Useful when environment variables change during runtime (rare in production,
    but helpful during development or testing).
    
    Returns:
        Settings: New settings instance
        
    Example:
        >>> settings = reload_settings()  # Re-reads .env file
    """
    global _settings_instance
    _settings_instance = Settings()
    return _settings_instance
