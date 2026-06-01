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
- OpenAI: GPT-4o, GPT-4-turbo, GPT-3.5-turbo
- Anthropic: Claude Sonnet 4.6, Claude Opus 4.8, Claude Haiku
- Google: Gemini 1.5 Pro/Flash, Gemini Pro

Usage Example:
    from app.core.llm import LLMFactory

    llm = LLMFactory.create()
    response = llm.invoke("What is RAG?")

    # Create with explicit overrides (e.g. from UI session state)
    llm = LLMFactory.create_from(provider="openai", model="gpt-4o", temperature=0.3)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_groq import ChatGroq

from app.utils.config import get_settings


class BaseLLM(ABC):
    """Abstract base class defining the interface for all LLM implementations."""

    @abstractmethod
    def get_llm(self) -> Any:
        """Create and return a LangChain-compatible chat model instance."""
        pass


class GroqLLM(BaseLLM):
    """Groq LLM — fast inference for Llama, Mixtral, Gemma models."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def get_llm(self) -> ChatGroq:
        settings = get_settings()
        if not settings.groq_api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY in your .env file.\n"
                "Get a free key at: https://console.groq.com/keys"
            )
        return ChatGroq(
            model=self._model or settings.llm_model_name,
            temperature=self._temperature if self._temperature is not None else settings.llm_temperature,
            max_tokens=self._max_tokens or settings.llm_max_tokens,
            groq_api_key=settings.groq_api_key,
        )


class OpenAILLM(BaseLLM):
    """OpenAI LLM — GPT-4o, GPT-4-turbo, and related models."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def get_llm(self) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is not installed. Run: pip install langchain-openai"
            )
        settings = get_settings()
        if not settings.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in your .env file."
            )
        return ChatOpenAI(
            model=self._model or settings.llm_model_name,
            temperature=self._temperature if self._temperature is not None else settings.llm_temperature,
            max_tokens=self._max_tokens or settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )


class AnthropicLLM(BaseLLM):
    """Anthropic LLM — Claude Sonnet, Opus, and Haiku models."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def get_llm(self) -> Any:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is not installed. Run: pip install langchain-anthropic"
            )
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY in your .env file."
            )
        return ChatAnthropic(
            model=self._model or settings.llm_model_name,
            temperature=self._temperature if self._temperature is not None else settings.llm_temperature,
            max_tokens=self._max_tokens or settings.llm_max_tokens or 4096,
            anthropic_api_key=settings.anthropic_api_key,
        )


class GoogleLLM(BaseLLM):
    """Google LLM — Gemini 1.5 Pro/Flash with 1M-token context window."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def get_llm(self) -> Any:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            )
        settings = get_settings()
        if not settings.google_api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY in your .env file."
            )
        return ChatGoogleGenerativeAI(
            model=self._model or settings.llm_model_name,
            temperature=self._temperature if self._temperature is not None else settings.llm_temperature,
            max_output_tokens=self._max_tokens or settings.llm_max_tokens,
            google_api_key=settings.google_api_key,
        )


# Provider → class mapping used by LLMFactory
_PROVIDER_MAP = {
    "groq": GroqLLM,
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "google": GoogleLLM,
}


class LLMFactory:
    """
    Factory class for creating LLM instances.

    Usage:
        # Use .env defaults
        llm = LLMFactory.create()

        # Override from UI session state
        llm = LLMFactory.create_from(provider="anthropic", model="claude-sonnet-4-6", temperature=0.5)
    """

    @staticmethod
    def create() -> Any:
        """Create an LLM using Settings defaults (provider + model from .env)."""
        settings = get_settings()
        provider = settings.llm_provider.lower()
        return LLMFactory._build(provider)

    @staticmethod
    def create_from(
        provider: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Create an LLM with explicit overrides — used by the UI runtime controls."""
        return LLMFactory._build(provider, model=model, temperature=temperature, max_tokens=max_tokens)

    @staticmethod
    def _build(
        provider: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        provider = provider.lower()
        if provider not in _PROVIDER_MAP:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Supported: {', '.join(_PROVIDER_MAP.keys())}"
            )
        cls = _PROVIDER_MAP[provider]
        return cls(model=model, temperature=temperature, max_tokens=max_tokens).get_llm()
