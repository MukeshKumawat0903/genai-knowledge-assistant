"""
LLM Factory and Abstraction Layer

This module provides a clean, extensible abstraction for working with different
LLM providers. It prevents vendor lock-in by providing a unified interface that
makes it easy to switch between cloud and local LLMs.

Key Design Principles:
- Single Responsibility: Each class has one clear purpose
- Open/Closed: Easy to add new providers without modifying existing code
- Dependency Inversion: Depend on abstractions (BaseLLM), not concrete classes
- Factory Pattern: Centralized LLM instantiation logic

Supported Providers:
- Groq (Cloud): Fast inference with Llama, Mixtral, Gemma models

Usage Example:
    from app.core.llm import LLMFactory
    
    # Factory automatically reads from Settings
    llm = LLMFactory.create()
    
    # Use the LLM in your application
    response = llm.invoke("What is RAG?")
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_groq import ChatGroq

from app.utils.config import get_settings


class BaseLLM(ABC):
    """
    Abstract base class defining the interface for all LLM implementations.
    
    This abstraction ensures that all LLM providers expose a consistent interface,
    making it trivial to swap providers without changing downstream code.
    
    Responsibilities:
    - Define the contract that all LLM implementations must follow
    - Return a LangChain-compatible chat model
    - Encapsulate provider-specific initialization logic
    """
    
    @abstractmethod
    def get_llm(self) -> Any:
        """
        Create and return a LangChain-compatible chat model instance.
        
        Returns:
            A LangChain chat model (e.g., ChatGroq)
            that can be used throughout the application for text generation.
            
        Raises:
            ValueError: If required configuration is missing or invalid
            RuntimeError: If the provider SDK fails to initialize
        """
        pass


class GroqLLM(BaseLLM):
    """
    Groq LLM implementation for fast cloud-based inference.
    
    Groq provides ultra-fast inference for open-source models like:
    - llama-3.1-70b-versatile
    - llama-3.3-70b-versatile
    - mixtral-8x7b-32768
    - gemma2-9b-it
    
    This implementation:
    - Reads all configuration from the centralized Settings
    - Validates that the Groq API key is set
    - Returns a LangChain ChatGroq instance ready for use
    
    Note: Groq offers generous free tier and excellent performance.
    """
    
    def get_llm(self) -> ChatGroq:
        """
        Create and return a ChatGroq instance configured from Settings.
        
        Returns:
            ChatGroq: Configured Groq chat model instance
            
        Raises:
            ValueError: If GROQ_API_KEY is not set in environment
        """
        settings = get_settings()
        
        # Validate API key
        if not settings.groq_api_key:
            raise ValueError(
                "Groq API key not found. Please set GROQ_API_KEY in your .env file.\n"
                "Get your free API key at: https://console.groq.com/keys"
            )
        
        # Create and return ChatGroq instance with settings from config
        return ChatGroq(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            groq_api_key=settings.groq_api_key,
        )


class LLMFactory:
    """
    Factory class for creating LLM instances based on configuration.
    
    Centralizes LLM instantiation logic, making it easy to:
    - Switch between providers by changing one config value
    - Add new providers without modifying consumer code
    - Apply consistent error handling across all providers
    
    Usage:
        llm = LLMFactory.create()
        response = llm.invoke("Explain RAG in one sentence")
    """
    
    @staticmethod
    def create() -> Any:
        """
        Create and return an LLM instance based on Settings configuration.
        
        This method:
        1. Reads the llm_provider from Settings
        2. Instantiates the corresponding LLM class
        3. Calls get_llm() to return the LangChain-compatible model
        
        Returns:
            A LangChain chat model (ChatGroq) ready for use
            
        Raises:
            ValueError: If the configured provider is not supported
            
        Example:
            llm = LLMFactory.create()
            response = llm.invoke("What is machine learning?")
            print(response.content)
        """
        settings = get_settings()
        provider = settings.llm_provider.lower()
        
        # Map provider string to corresponding class
        if provider == "groq":
            return GroqLLM().get_llm()
        
        # TODO: Uncomment when implementing additional providers
        # elif provider == "openai":
        #     return OpenAILLM().get_llm()
        #
        # elif provider == "anthropic":
        #     return AnthropicLLM().get_llm()
        #
        # elif provider == "google":
        #     return GoogleLLM().get_llm()
        
        else:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Supported providers: groq\n"
                f"Update LLM_PROVIDER in your .env file."
            )
