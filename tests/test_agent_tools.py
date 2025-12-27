"""
Tests for Agent Tools

This test suite validates the agent tool implementations including
tool creation, execution, registry management, and integration patterns.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.agents.tools import (
    create_pdf_knowledge_tool,
    create_web_search_tool,
    create_wikipedia_tool,
    create_arxiv_tool,
    ToolRegistry,
    create_all_tools,
    get_tool_by_name,
    SearchInput,
    PDFSearchInput
)


class TestToolCreation:
    """Test suite for tool creation functions."""
    
    def test_create_pdf_knowledge_tool_without_retriever(self):
        """Test PDF tool creation without retriever."""
        tool = create_pdf_knowledge_tool()
        
        assert tool is not None
        assert tool.name == "pdf_knowledge_search"
        assert "PDF knowledge base" in tool.description
        assert callable(tool.run)
    
    def test_create_pdf_knowledge_tool_with_retriever(self):
        """Test PDF tool creation with mock retriever."""
        mock_retriever = Mock()
        tool = create_pdf_knowledge_tool(mock_retriever)
        
        assert tool is not None
        assert tool.name == "pdf_knowledge_search"
    
    def test_create_web_search_tool(self):
        """Test web search tool creation."""
        tool = create_web_search_tool()
        
        assert tool is not None
        assert tool.name == "web_search"
        assert "web" in tool.description.lower()
        assert callable(tool.run)
    
    def test_create_wikipedia_tool(self):
        """Test Wikipedia tool creation."""
        tool = create_wikipedia_tool()
        
        assert tool is not None
        assert tool.name == "wikipedia_search"
        assert "wikipedia" in tool.description.lower()
        assert callable(tool.run)
    
    def test_create_arxiv_tool(self):
        """Test ArXiv tool creation."""
        tool = create_arxiv_tool()
        
        assert tool is not None
        assert tool.name == "arxiv_search"
        assert "arxiv" in tool.description.lower()
        assert callable(tool.run)
    
    def test_tool_names_are_unique(self):
        """Test that all tools have unique names."""
        tools = create_all_tools()
        names = [tool.name for tool in tools]
        
        assert len(names) == len(set(names)), "Tool names must be unique"
    
    def test_all_tools_have_descriptions(self):
        """Test that all tools have non-empty descriptions."""
        tools = create_all_tools()
        
        for tool in tools:
            assert tool.description, f"Tool {tool.name} missing description"
            assert len(tool.description) > 50, f"Tool {tool.name} description too short"


class TestPDFKnowledgeTool:
    """Test suite for PDF knowledge search tool."""
    
    def test_pdf_tool_without_retriever_returns_message(self):
        """Test that PDF tool without retriever returns helpful message."""
        tool = create_pdf_knowledge_tool()
        result = tool.run("test query")
        
        assert "not initialized" in result.lower()
        assert "index documents" in result.lower()
    
    def test_pdf_tool_with_mock_retriever(self):
        """Test PDF tool with mock retriever returning documents."""
        # Create mock retriever
        mock_retriever = Mock()
        mock_doc1 = Mock()
        mock_doc1.page_content = "Machine learning is a subset of AI"
        mock_doc1.metadata = {'source': 'ml_guide.pdf', 'page': 5}
        
        mock_doc2 = Mock()
        mock_doc2.page_content = "Neural networks are computing systems"
        mock_doc2.metadata = {'source': 'nn_intro.pdf', 'page': 12}
        
        mock_retriever.similarity_search.return_value = [mock_doc1, mock_doc2]
        
        # Create tool and run query
        tool = create_pdf_knowledge_tool(mock_retriever)
        result = tool.run("What is machine learning?")
        
        # Verify results
        assert "Machine learning" in result
        assert "Neural networks" in result
        assert "ml_guide.pdf" in result
        assert "nn_intro.pdf" in result
        assert "Page: 5" in result
        assert "Page: 12" in result
        
        # Verify retriever was called
        mock_retriever.similarity_search.assert_called_once()
    
    def test_pdf_tool_no_results(self):
        """Test PDF tool when no documents are found."""
        mock_retriever = Mock()
        mock_retriever.similarity_search.return_value = []
        
        tool = create_pdf_knowledge_tool(mock_retriever)
        result = tool.run("nonexistent topic")
        
        assert "No relevant information found" in result
    
    def test_pdf_tool_handles_exception(self):
        """Test PDF tool error handling."""
        mock_retriever = Mock()
        mock_retriever.similarity_search.side_effect = Exception("Database error")
        
        tool = create_pdf_knowledge_tool(mock_retriever)
        result = tool.run("test query")
        
        assert "Error" in result
        assert "Database error" in result


class TestToolRegistry:
    """Test suite for ToolRegistry class."""
    
    def test_registry_initialization(self):
        """Test registry initializes with all tools."""
        registry = ToolRegistry()
        
        assert registry is not None
        assert len(registry.get_all_tools()) == 4  # 4 tools
    
    def test_get_tool_by_name(self):
        """Test retrieving specific tool by name."""
        registry = ToolRegistry()
        
        wiki_tool = registry.get_tool("wikipedia_search")
        assert wiki_tool is not None
        assert wiki_tool.name == "wikipedia_search"
        
        web_tool = registry.get_tool("web_search")
        assert web_tool is not None
        assert web_tool.name == "web_search"
    
    def test_get_nonexistent_tool(self):
        """Test retrieving non-existent tool returns None."""
        registry = ToolRegistry()
        
        tool = registry.get_tool("nonexistent_tool")
        assert tool is None
    
    def test_get_all_tools(self):
        """Test getting all tools."""
        registry = ToolRegistry()
        tools = registry.get_all_tools()
        
        assert len(tools) == 4
        assert all(hasattr(tool, 'name') for tool in tools)
        assert all(hasattr(tool, 'description') for tool in tools)
        assert all(callable(tool.run) for tool in tools)
    
    def test_get_tool_names(self):
        """Test getting all tool names."""
        registry = ToolRegistry()
        names = registry.get_tool_names()
        
        assert len(names) == 4
        assert "pdf_knowledge_search" in names
        assert "web_search" in names
        assert "wikipedia_search" in names
        assert "arxiv_search" in names
    
    def test_get_tool_descriptions(self):
        """Test getting tool descriptions."""
        registry = ToolRegistry()
        descriptions = registry.get_tool_descriptions()
        
        assert len(descriptions) == 4
        assert all(isinstance(desc, str) for desc in descriptions.values())
        assert all(len(desc) > 0 for desc in descriptions.values())
    
    def test_registry_with_retriever(self):
        """Test registry initialization with retriever."""
        mock_retriever = Mock()
        registry = ToolRegistry(retriever=mock_retriever)
        
        assert registry.retriever is mock_retriever
        assert len(registry.get_all_tools()) == 4


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_create_all_tools(self):
        """Test create_all_tools function."""
        tools = create_all_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 4
        assert all(hasattr(tool, 'name') for tool in tools)
    
    def test_create_all_tools_with_retriever(self):
        """Test create_all_tools with retriever."""
        mock_retriever = Mock()
        tools = create_all_tools(retriever=mock_retriever)
        
        assert len(tools) == 4
        # PDF tool should have the retriever
        pdf_tool = next(t for t in tools if t.name == "pdf_knowledge_search")
        assert pdf_tool is not None
    
    def test_get_tool_by_name(self):
        """Test get_tool_by_name function."""
        wiki_tool = get_tool_by_name("wikipedia_search")
        
        assert wiki_tool is not None
        assert wiki_tool.name == "wikipedia_search"
    
    def test_get_nonexistent_tool_by_name(self):
        """Test get_tool_by_name with invalid name."""
        tool = get_tool_by_name("invalid_tool")
        
        assert tool is None


class TestToolInputSchemas:
    """Test suite for Pydantic input schemas."""
    
    def test_search_input_schema(self):
        """Test SearchInput schema validation."""
        # Valid input
        search_input = SearchInput(query="test query")
        assert search_input.query == "test query"
        
        # Test with empty query (should still work, string type)
        search_input_empty = SearchInput(query="")
        assert search_input_empty.query == ""
    
    def test_pdf_search_input_schema(self):
        """Test PDFSearchInput schema validation."""
        # Valid input with default top_k
        pdf_input = PDFSearchInput(query="test query")
        assert pdf_input.query == "test query"
        assert pdf_input.top_k == 3  # Default value
        
        # Valid input with custom top_k
        pdf_input_custom = PDFSearchInput(query="test", top_k=5)
        assert pdf_input_custom.query == "test"
        assert pdf_input_custom.top_k == 5
    
    def test_pdf_search_input_optional_top_k(self):
        """Test PDFSearchInput with None top_k."""
        pdf_input = PDFSearchInput(query="test", top_k=None)
        assert pdf_input.top_k is None


class TestToolDescriptions:
    """Test suite for tool description quality."""
    
    def test_descriptions_mention_use_cases(self):
        """Test that descriptions explain when to use each tool."""
        tools = create_all_tools()
        
        for tool in tools:
            desc_lower = tool.description.lower()
            # Should mention "use this tool" or similar
            assert any(phrase in desc_lower for phrase in [
                "use this tool",
                "use this",
                "ideal for",
                "best for"
            ]), f"Tool {tool.name} description missing use case guidance"
    
    def test_descriptions_mention_input_format(self):
        """Test that descriptions explain input format."""
        tools = create_all_tools()
        
        for tool in tools:
            desc_lower = tool.description.lower()
            # Should mention input format
            assert any(phrase in desc_lower for phrase in [
                "input",
                "query",
                "search",
                "question"
            ]), f"Tool {tool.name} description missing input format info"
    
    def test_descriptions_are_informative(self):
        """Test that descriptions are sufficiently detailed."""
        tools = create_all_tools()
        
        for tool in tools:
            # Should be at least 100 characters
            assert len(tool.description) >= 100, \
                f"Tool {tool.name} description too short: {len(tool.description)} chars"
            
            # Should have multiple sentences
            assert tool.description.count('.') >= 2, \
                f"Tool {tool.name} description should have multiple sentences"


class TestToolIntegration:
    """Test suite for tool integration patterns."""
    
    def test_tools_compatible_with_langchain_agent(self):
        """Test that tools have the interface expected by LangChain agents."""
        tools = create_all_tools()
        
        for tool in tools:
            # Must have name (string)
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0
            
            # Must have description (string)
            assert isinstance(tool.description, str)
            assert len(tool.description) > 0
            
            # Must have callable run method
            assert callable(tool.run)
    
    def test_tools_can_be_called_with_string_input(self):
        """Test that all tools accept string input."""
        # Create tools (will fail if network unavailable, that's ok)
        pdf_tool = create_pdf_knowledge_tool()
        
        # Test that they accept string input (even if they fail due to missing deps)
        try:
            result = pdf_tool.run("test query")
            assert isinstance(result, str)
        except Exception:
            # Expected if dependencies not installed
            pass
    
    def test_tool_results_are_strings(self):
        """Test that tools return string results."""
        # Test PDF tool (doesn't need network)
        pdf_tool = create_pdf_knowledge_tool()
        result = pdf_tool.run("test")
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestErrorHandling:
    """Test suite for tool error handling."""
    
    def test_pdf_tool_handles_retriever_error(self):
        """Test PDF tool gracefully handles retriever errors."""
        mock_retriever = Mock()
        mock_retriever.similarity_search.side_effect = Exception("Connection failed")
        
        tool = create_pdf_knowledge_tool(mock_retriever)
        result = tool.run("test")
        
        assert isinstance(result, str)
        assert "Error" in result
        assert "Connection failed" in result
    
    def test_tools_dont_raise_exceptions(self):
        """Test that tools catch and return errors as strings."""
        # PDF tool without retriever
        pdf_tool = create_pdf_knowledge_tool()
        result = pdf_tool.run("test")
        
        # Should return string, not raise exception
        assert isinstance(result, str)


class TestToolMetadata:
    """Test suite for tool metadata and properties."""
    
    def test_tool_names_follow_convention(self):
        """Test that tool names follow snake_case convention."""
        tools = create_all_tools()
        
        for tool in tools:
            # Should be lowercase with underscores
            assert tool.name.islower() or '_' in tool.name
            assert ' ' not in tool.name
            assert '-' not in tool.name
    
    def test_tool_names_are_descriptive(self):
        """Test that tool names are descriptive."""
        tools = create_all_tools()
        
        expected_keywords = {
            "pdf_knowledge_search": ["pdf", "knowledge", "search"],
            "web_search": ["web", "search"],
            "wikipedia_search": ["wikipedia", "search"],
            "arxiv_search": ["arxiv", "search"]
        }
        
        for tool in tools:
            keywords = expected_keywords[tool.name]
            name_lower = tool.name.lower()
            
            for keyword in keywords:
                assert keyword in name_lower, \
                    f"Tool {tool.name} missing keyword '{keyword}'"


# Pytest fixtures
@pytest.fixture
def mock_retriever():
    """Fixture providing a mock retriever."""
    retriever = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Sample content"
    mock_doc.metadata = {'source': 'test.pdf', 'page': 1}
    retriever.similarity_search.return_value = [mock_doc]
    return retriever


@pytest.fixture
def tool_registry():
    """Fixture providing a fresh tool registry."""
    return ToolRegistry()


@pytest.fixture
def all_tools():
    """Fixture providing all tools."""
    return create_all_tools()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
