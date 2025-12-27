"""
Tests for RAG Chain

This test suite validates the complete RAG chain workflow including:
- Retrieval and generation integration
- Source grounding enforcement
- Chat history integration
- Edge case handling
- Error conditions
"""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document

from app.rag.chain import RAGChain, create_rag_chain


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_retriever():
    """Create a mock RAGRetriever."""
    retriever = Mock()
    retriever.retrieve = Mock()
    return retriever


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = Mock()
    # Mock the invoke method to return AIMessage-like object
    response = Mock()
    response.content = "This is a generated answer."
    llm.invoke = Mock(return_value=response)
    return llm


@pytest.fixture
def sample_documents():
    """Create sample retrieved documents."""
    return [
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation.",
            metadata={"source": "rag_basics.pdf", "page": 1}
        ),
        Document(
            page_content="RAG combines retrieval with language models.",
            metadata={"source": "rag_overview.md", "page": 2}
        )
    ]


@pytest.fixture
def chain(mock_retriever, mock_llm):
    """Create a RAG chain with mocked dependencies."""
    return RAGChain(
        retriever=mock_retriever,
        llm=mock_llm
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_chain_initialization(mock_retriever, mock_llm):
    """Test chain initializes with correct attributes."""
    chain = RAGChain(
        retriever=mock_retriever,
        llm=mock_llm,
        return_source_documents=False
    )
    
    assert chain.retriever is mock_retriever
    assert chain.llm is mock_llm
    assert chain.prompt_manager is None
    assert chain.memory is None
    assert chain.return_source_documents is False


def test_chain_initialization_with_optionals(mock_retriever, mock_llm):
    """Test chain initialization with optional parameters."""
    prompt_manager = Mock()
    memory = Mock()
    
    chain = RAGChain(
        retriever=mock_retriever,
        llm=mock_llm,
        prompt_manager=prompt_manager,
        memory=memory,
        custom_param="value"
    )
    
    assert chain.prompt_manager is prompt_manager
    assert chain.memory is memory
    assert chain.config["custom_param"] == "value"


# ============================================================================
# RUN METHOD TESTS
# ============================================================================

def test_run_basic_query(chain, mock_retriever, mock_llm, sample_documents):
    """Test basic query execution."""
    # Setup mocks
    mock_retriever.retrieve.return_value = sample_documents
    
    # Run chain
    result = chain.run("What is RAG?")
    
    # Verify retriever was called
    mock_retriever.retrieve.assert_called_once_with(
        query="What is RAG?",
        chat_history=None
    )
    
    # Verify LLM was called
    assert mock_llm.invoke.called
    
    # Verify result structure
    assert "answer" in result
    assert "source_documents" in result
    assert "metadata" in result
    assert result["answer"] == "This is a generated answer."
    assert len(result["source_documents"]) == 2
    assert result["metadata"]["query"] == "What is RAG?"
    assert result["metadata"]["num_sources"] == 2
    assert result["metadata"]["has_context"] is True


def test_run_with_chat_history(chain, mock_retriever, mock_llm, sample_documents):
    """Test query execution with chat history."""
    mock_retriever.retrieve.return_value = sample_documents
    
    chat_history = [
        ("What is ML?", "Machine Learning is..."),
        ("Tell me more", "It involves...")
    ]
    
    result = chain.run("What about RAG?", chat_history=chat_history)
    
    # Verify chat history passed to retriever
    mock_retriever.retrieve.assert_called_once_with(
        query="What about RAG?",
        chat_history=chat_history
    )
    
    # Verify prompt includes history
    call_args = mock_llm.invoke.call_args[0][0]
    assert "Previous conversation" in call_args
    assert "What is ML?" in call_args


def test_run_with_kwargs(chain, mock_retriever, mock_llm, sample_documents):
    """Test passing additional kwargs to retriever."""
    mock_retriever.retrieve.return_value = sample_documents
    
    result = chain.run("What is RAG?", k=5, score_threshold=0.7)
    
    # Verify kwargs passed to retriever
    mock_retriever.retrieve.assert_called_once_with(
        query="What is RAG?",
        chat_history=None,
        k=5,
        score_threshold=0.7
    )


def test_run_without_source_documents(mock_retriever, mock_llm, sample_documents):
    """Test chain configured to not return source documents."""
    chain = RAGChain(
        retriever=mock_retriever,
        llm=mock_llm,
        return_source_documents=False
    )
    
    mock_retriever.retrieve.return_value = sample_documents
    
    result = chain.run("What is RAG?")
    
    assert "answer" in result
    assert "metadata" in result
    assert "source_documents" not in result


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_run_empty_query(chain):
    """Test handling of empty query."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        chain.run("")


def test_run_whitespace_query(chain):
    """Test handling of whitespace-only query."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        chain.run("   ")


def test_run_no_documents_found(chain, mock_retriever, mock_llm):
    """Test handling when no documents are retrieved."""
    mock_retriever.retrieve.return_value = []
    
    result = chain.run("What is XYZ?")
    
    # Verify LLM was NOT called
    assert not mock_llm.invoke.called
    
    # Verify appropriate response
    assert "answer" in result
    assert "I don't have enough information" in result["answer"]
    assert result["source_documents"] == []
    assert result["metadata"]["num_sources"] == 0
    assert result["metadata"]["has_context"] is False


def test_run_query_stripping(chain, mock_retriever, mock_llm, sample_documents):
    """Test query is stripped of whitespace."""
    mock_retriever.retrieve.return_value = sample_documents
    
    result = chain.run("  What is RAG?  ")
    
    # Verify stripped query
    mock_retriever.retrieve.assert_called_once_with(
        query="What is RAG?",
        chat_history=None
    )


# ============================================================================
# CONTEXT FORMATTING TESTS
# ============================================================================

def test_format_context_basic(chain, sample_documents):
    """Test basic context formatting."""
    context = chain._format_context(sample_documents)
    
    assert "[Source 1: rag_basics.pdf]" in context
    assert "[Source 2: rag_overview.md]" in context
    assert "RAG stands for" in context
    assert "RAG combines" in context
    assert "\n\n" in context  # Documents separated by double newline


def test_format_context_empty(chain):
    """Test formatting empty document list."""
    context = chain._format_context([])
    assert context == ""


def test_format_context_unknown_source(chain):
    """Test formatting documents without source metadata."""
    docs = [
        Document(
            page_content="Test content",
            metadata={}
        )
    ]
    
    context = chain._format_context(docs)
    assert "[Source 1: Unknown]" in context
    assert "Test content" in context


def test_format_context_ordering(chain):
    """Test documents are numbered in order."""
    docs = [
        Document(page_content="First", metadata={"source": "a.pdf"}),
        Document(page_content="Second", metadata={"source": "b.pdf"}),
        Document(page_content="Third", metadata={"source": "c.pdf"})
    ]
    
    context = chain._format_context(docs)
    
    # Verify ordering
    idx_1 = context.index("[Source 1")
    idx_2 = context.index("[Source 2")
    idx_3 = context.index("[Source 3")
    assert idx_1 < idx_2 < idx_3


# ============================================================================
# PROMPT CREATION TESTS
# ============================================================================

def test_create_prompt_basic(chain):
    """Test basic prompt creation."""
    prompt = chain._create_prompt(
        query="What is RAG?",
        context="RAG is Retrieval-Augmented Generation."
    )
    
    # Verify grounding instructions
    assert "Use ONLY the information" in prompt
    assert "Do not make up" in prompt
    
    # Verify context and question included
    assert "Context:" in prompt
    assert "RAG is Retrieval-Augmented Generation." in prompt
    assert "Question: What is RAG?" in prompt
    
    # Verify reminder
    assert "Remember: Use only the context above" in prompt


def test_create_prompt_with_history(chain):
    """Test prompt creation with chat history."""
    chat_history = [
        ("What is ML?", "Machine Learning is..."),
        ("How does it work?", "It uses algorithms...")
    ]
    
    prompt = chain._create_prompt(
        query="What about RAG?",
        context="RAG combines retrieval...",
        chat_history=chat_history
    )
    
    # Verify history included
    assert "Previous conversation:" in prompt
    assert "Human: What is ML?" in prompt
    assert "Assistant: Machine Learning is..." in prompt
    assert "Human: How does it work?" in prompt


def test_create_prompt_no_history(chain):
    """Test prompt creation without history."""
    prompt = chain._create_prompt(
        query="What is RAG?",
        context="RAG is..."
    )
    
    assert "Previous conversation:" not in prompt


def test_create_prompt_structure(chain):
    """Test prompt has correct structure."""
    prompt = chain._create_prompt(
        query="Test query",
        context="Test context",
        chat_history=[("Q1", "A1")]
    )
    
    # Verify order: instructions -> history -> context -> question -> reminder
    parts = prompt.split("\n")
    
    # Find key sections
    has_instructions = any("Use ONLY" in part for part in parts)
    has_history = any("Previous conversation" in part for part in parts)
    has_context = any("Context:" in part for part in parts)
    has_question = any("Question:" in part for part in parts)
    has_reminder = any("Remember:" in part for part in parts)
    
    assert all([has_instructions, has_history, has_context, has_question, has_reminder])


# ============================================================================
# LLM RESPONSE HANDLING TESTS
# ============================================================================

def test_run_extracts_content_attribute(chain, mock_retriever, sample_documents):
    """Test extraction of answer from AIMessage-like response."""
    mock_retriever.retrieve.return_value = sample_documents
    
    # Mock LLM response with content attribute
    llm = Mock()
    response = Mock()
    response.content = "Answer from content attribute"
    llm.invoke = Mock(return_value=response)
    chain.llm = llm
    
    result = chain.run("Test query")
    assert result["answer"] == "Answer from content attribute"


def test_run_fallback_to_str(chain, mock_retriever, sample_documents):
    """Test fallback to str() if no content attribute."""
    mock_retriever.retrieve.return_value = sample_documents
    
    # Mock LLM response without content attribute
    llm = Mock()
    response = "String response"
    llm.invoke = Mock(return_value=response)
    chain.llm = llm
    
    result = chain.run("Test query")
    assert result["answer"] == "String response"


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

def test_create_rag_chain_minimal(mock_retriever, mock_llm, monkeypatch):
    """Test create_rag_chain with minimal parameters."""
    # Mock LLMFactory
    mock_factory = Mock()
    mock_factory.create = Mock(return_value=mock_llm)
    
    from app.rag import chain as chain_module
    monkeypatch.setattr(chain_module, 'LLMFactory', mock_factory)
    
    # Create chain without LLM (should auto-create)
    result = create_rag_chain(retriever=mock_retriever)
    
    assert isinstance(result, RAGChain)
    assert result.retriever is mock_retriever
    assert result.llm is mock_llm
    assert result.return_source_documents is True


def test_create_rag_chain_with_llm(mock_retriever, mock_llm):
    """Test create_rag_chain with provided LLM."""
    result = create_rag_chain(
        retriever=mock_retriever,
        llm=mock_llm,
        return_source_documents=False
    )
    
    assert isinstance(result, RAGChain)
    assert result.retriever is mock_retriever
    assert result.llm is mock_llm
    assert result.return_source_documents is False


def test_create_rag_chain_with_kwargs(mock_retriever, mock_llm):
    """Test create_rag_chain passes kwargs correctly."""
    prompt_manager = Mock()
    
    result = create_rag_chain(
        retriever=mock_retriever,
        llm=mock_llm,
        prompt_manager=prompt_manager,
        custom_param="value"
    )
    
    assert result.prompt_manager is prompt_manager
    assert result.config["custom_param"] == "value"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_workflow(chain, mock_retriever, mock_llm, sample_documents):
    """Test complete end-to-end workflow."""
    # Setup
    mock_retriever.retrieve.return_value = sample_documents
    mock_llm.invoke.return_value.content = "RAG is Retrieval-Augmented Generation."
    
    # Execute
    result = chain.run(
        "What is RAG?",
        chat_history=[("Previous Q", "Previous A")],
        k=4,
        score_threshold=0.7
    )
    
    # Verify complete flow
    # 1. Retriever called with query, history, and kwargs
    mock_retriever.retrieve.assert_called_once()
    call_kwargs = mock_retriever.retrieve.call_args[1]
    assert call_kwargs["query"] == "What is RAG?"
    assert call_kwargs["k"] == 4
    assert call_kwargs["score_threshold"] == 0.7
    
    # 2. LLM called with formatted prompt
    assert mock_llm.invoke.called
    prompt = mock_llm.invoke.call_args[0][0]
    assert "Previous conversation" in prompt
    assert "Context:" in prompt
    assert "What is RAG?" in prompt
    
    # 3. Result properly formatted
    assert result["answer"] == "RAG is Retrieval-Augmented Generation."
    assert len(result["source_documents"]) == 2
    assert result["metadata"]["num_sources"] == 2


def test_grounding_enforcement(chain, mock_retriever, mock_llm, sample_documents):
    """Test that grounding instructions are properly enforced."""
    mock_retriever.retrieve.return_value = sample_documents
    
    chain.run("What is RAG?")
    
    # Extract prompt sent to LLM
    prompt = mock_llm.invoke.call_args[0][0]
    
    # Verify strong grounding language
    assert "Use ONLY the information" in prompt
    assert "i don't know" in prompt.lower()
    assert "Do not make up" in prompt
    assert "not explicitly stated" in prompt


def test_metadata_correctness(chain, mock_retriever, mock_llm):
    """Test metadata is correctly populated in results."""
    docs = [
        Document(page_content="Doc 1", metadata={"source": "a.pdf"}),
        Document(page_content="Doc 2", metadata={"source": "b.pdf"}),
        Document(page_content="Doc 3", metadata={"source": "c.pdf"})
    ]
    mock_retriever.retrieve.return_value = docs
    
    result = chain.run("Test query")
    
    metadata = result["metadata"]
    assert metadata["query"] == "Test query"
    assert metadata["num_sources"] == 3
    assert metadata["has_context"] is True


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_retriever_exception_propagates(chain, mock_retriever):
    """Test that retriever exceptions are propagated."""
    mock_retriever.retrieve.side_effect = Exception("Retrieval failed")
    
    with pytest.raises(Exception, match="Retrieval failed"):
        chain.run("Test query")


def test_llm_exception_propagates(chain, mock_retriever, mock_llm, sample_documents):
    """Test that LLM exceptions are propagated."""
    mock_retriever.retrieve.return_value = sample_documents
    mock_llm.invoke.side_effect = Exception("LLM failed")
    
    with pytest.raises(Exception, match="LLM failed"):
        chain.run("Test query")
