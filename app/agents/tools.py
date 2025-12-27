"""Agent Tools for GenAI Knowledge Assistant - Wrappers for PDF, web, Wikipedia, and ArXiv search."""

from typing import Optional, List, Dict, Any

try:
    # LangChain 1.0+ uses langchain_core
    from langchain_core.tools import BaseTool, tool, Tool
    USING_DECORATOR = True
except ImportError:
    # LangChain 0.x legacy - tool decorator not available
    from langchain.tools import Tool, StructuredTool, BaseTool
    USING_DECORATOR = False

from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from pydantic import BaseModel, Field


# =============================================================================
# Tool Input Schemas (for StructuredTool validation)
# =============================================================================

class SearchInput(BaseModel):
    """Input schema for search tools."""
    query: str = Field(
        description="The search query or question to find information about"
    )


class PDFSearchInput(BaseModel):
    """Input schema for PDF knowledge search."""
    query: str = Field(
        description="The question or topic to search in the PDF knowledge base"
    )
    top_k: Optional[int] = Field(
        default=3,
        description="Number of relevant document chunks to retrieve (default: 3)"
    )


# =============================================================================
# Tool 1: PDF Knowledge Search
# =============================================================================

def create_pdf_knowledge_tool(retriever=None) -> BaseTool:
    """Search indexed PDF documents for relevant information."""
    
    def search_pdf_knowledge(query: str, top_k: int = 3) -> str:
        if retriever is None:
            return "PDF knowledge base not initialized. Index documents first."
        
        try:
            docs = retriever.similarity_search(query, k=top_k)
            if not docs:
                return f"No relevant information found for: {query}"
            
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                chunk = f"--- Result {i} ---\nSource: {source}\nPage: {page}\nContent:\n{doc.page_content}\n"
                results.append(chunk)
            return "\n".join(results)
        except Exception as e:
            return f"Error searching PDF knowledge base: {str(e)}"
    
    return Tool(
        name="pdf_knowledge_search",
        func=search_pdf_knowledge,
        description=(
            "Search the indexed PDF knowledge base for answers grounded in your ingested documents. "
            "Use this tool when the user asks about content that should exist in uploaded PDFs, reports, manuals, or notes."
        )
    )


# =============================================================================
# Tool 2: Web Search
# =============================================================================

def create_web_search_tool(max_results: int = 5) -> BaseTool:
    """Real-time web search using DuckDuckGo."""
    search = DuckDuckGoSearchRun()
    
    def search_web(query: str) -> str:
        try:
            results = search.run(query)
            return results if results and results.strip() else f"No web results for: {query}"
        except Exception as e:
            return f"Error searching web: {str(e)}"
    
    return Tool(
        name="web_search",
        func=search_web,
        description="Search the web for current information, news, and general knowledge."
    )


# =============================================================================
# Tool 3: Wikipedia Search
# =============================================================================

def create_wikipedia_tool(top_k_results: int = 3, doc_content_chars_max: int = 4000) -> BaseTool:
    """Search Wikipedia for encyclopedic knowledge."""
    wikipedia = WikipediaAPIWrapper(
        top_k_results=top_k_results,
        doc_content_chars_max=doc_content_chars_max
    )
    wiki_search = WikipediaQueryRun(api_wrapper=wikipedia)
    
    def search_wikipedia(query: str) -> str:
        try:
            results = wiki_search.run(query)
            return results if results and results.strip() else f"No Wikipedia results for: {query}"
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"
    
    return Tool(
        name="wikipedia_search",
        func=search_wikipedia,
        description="Search Wikipedia for encyclopedic knowledge, definitions, and factual information."
    )


# =============================================================================
# Tool 4: ArXiv Search
# =============================================================================

def create_arxiv_tool(top_k_results: int = 3, doc_content_chars_max: int = 4000) -> BaseTool:
    """Search ArXiv for academic papers and research."""
    arxiv = ArxivAPIWrapper(
        top_k_results=top_k_results,
        doc_content_chars_max=doc_content_chars_max
    )
    arxiv_search = ArxivQueryRun(api_wrapper=arxiv)
    
    def search_arxiv(query: str) -> str:
        try:
            results = arxiv_search.run(query)
            return results if results and results.strip() else f"No ArXiv papers for: {query}"
        except Exception as e:
            return f"Error searching ArXiv: {str(e)}"
    
    return Tool(
        name="arxiv_search",
        func=search_arxiv,
        description="Search ArXiv for academic papers, cutting-edge research, and scientific publications."
    )


# =============================================================================
# Tool Registry and Factory
# =============================================================================

class ToolRegistry:
    """Central registry for managing agent tools."""
    
    def __init__(self, retriever=None):
        self.retriever = retriever
        self._tools = self._initialize_tools()
    
    def _initialize_tools(self) -> Dict[str, BaseTool]:
        return {
            "pdf_knowledge_search": create_pdf_knowledge_tool(self.retriever),
            "web_search": create_web_search_tool(),
            "wikipedia_search": create_wikipedia_tool(),
            "arxiv_search": create_arxiv_tool()
        }
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return self._tools.get(tool_name)
    
    def get_all_tools(self) -> List[BaseTool]:
        return list(self._tools.values())
    
    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in self._tools.items()}


# =============================================================================
# Convenience Functions
# =============================================================================

def create_all_tools(retriever=None) -> List[BaseTool]:
    """Create all tools at once for LangGraph compatibility."""
    tools = []

    # Include PDF knowledge tool (counts as one of the 4 tools)
    tools.append(create_pdf_knowledge_tool(retriever))
    
    wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=4000)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
    wikipedia_tool.name = "wikipedia_search"
    wikipedia_tool.description = (
        "Search Wikipedia for encyclopedic knowledge, definitions, and factual summaries. "
        "Use this for well-known concepts, people, places, and historical context."
    )
    tools.append(wikipedia_tool)
    
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=4000)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    arxiv_tool.name = "arxiv_search"
    arxiv_tool.description = (
        "Search ArXiv for academic papers, research preprints, and scientific publications. "
        "Use this tool when you need citations, paper titles/authors, or research-oriented summaries from academic sources." 
    )
    tools.append(arxiv_tool)
    
    web_search_tool = DuckDuckGoSearchRun()
    web_search_tool.name = "web_search"
    web_search_tool.description = (
        "Search the public web for current information, recent updates, news, and general knowledge. "
        "Use this when the answer may have changed recently or is not in the local knowledge base."
    )
    tools.append(web_search_tool)
    
    return tools


def get_tool_by_name(tool_name: str, retriever=None) -> Optional[BaseTool]:
    """Get a specific tool by name."""
    registry = ToolRegistry(retriever)
    return registry.get_tool(tool_name)


__all__ = [
    "create_pdf_knowledge_tool",
    "create_web_search_tool",
    "create_wikipedia_tool",
    "create_arxiv_tool",
    "ToolRegistry",
    "create_all_tools",
    "get_tool_by_name",
]
