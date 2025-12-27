"""Agent Router - LLM-based agent that dynamically routes queries to tools."""

from typing import List, Optional, Dict, Any
from langchain_core.tools import BaseTool
from langchain.agents import create_agent  # Modern LangChain 1.2.0 API

# Using modern create_agent (LangChain 1.2.0+) with LangGraph under the hood
USING_LANGGRAPH = True


# =============================================================================
# Agent Prompt Template (ReAct-style)
# =============================================================================

REACT_PROMPT_TEMPLATE = """Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important guidelines:
- Always provide thoughtful reasoning before taking actions
- Use the most appropriate tool for each query type
- For general knowledge, prefer Wikipedia over web search
- For academic/research questions, use ArXiv
- For current events/news, use web search
- For domain-specific questions, use PDF knowledge search
- If you don't find relevant information, say so explicitly
- Summarize the information found from each tool used

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


# =============================================================================
# KnowledgeAgent Class
# =============================================================================

class KnowledgeAgent:
    """LLM-based agent using ReAct pattern to route queries to tools."""
    
    def __init__(
        self,
        llm,
        tools: List[BaseTool],
        max_iterations: int = 10,
        verbose: bool = False,
        handle_parsing_errors: bool = True,
        **kwargs
    ):
        """
        Initialize the KnowledgeAgent.
        
        Args:
            llm: Language model instance (from LLMFactory)
                Should support chat/completion interface
            tools: List of Tool instances (from tools.py)
                Each tool must have: name, description, func
            max_iterations: Maximum agent reasoning iterations
                Prevents infinite loops (default: 10)
            verbose: If True, prints agent's reasoning steps
                Useful for debugging and demonstrations
            handle_parsing_errors: If True, gracefully handles LLM output parsing errors
            **kwargs: Additional configuration options
                Passed to AgentExecutor
        
        Raises:
            ValueError: If llm is None or tools list is empty
        
        Example:
            >>> llm = LLMFactory.create_llm()
            >>> tools = create_all_tools()
            >>> agent = KnowledgeAgent(llm, tools, verbose=True, max_iterations=5)
        """
        # Validate inputs
        if llm is None:
            raise ValueError("LLM instance is required")
        
        if not tools or len(tools) == 0:
            raise ValueError("At least one tool must be provided")
        
        # Store configuration
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.config = kwargs
        
        # Initialize agent
        self.agent_executor = self._initialize_agent(handle_parsing_errors)
    
    def _initialize_agent(self, handle_parsing_errors: bool):
        """
        Initialize the modern LangChain 1.2.0 agent with create_agent API.
        
        This method sets up the agent using the new unified API that:
        - Uses LangGraph under the hood for better control
        - Supports streaming and persistence out of the box
        - Provides better error handling and observability
        - Works seamlessly with Groq's tool calling
        
        Args:
            handle_parsing_errors: Whether to handle parsing errors gracefully
        
        Returns:
            Configured agent graph (runnable)
        
        Note:
            The new create_agent API replaces AgentExecutor and provides
            automatic LangGraph-based execution with improved performance.
            
            For Groq compatibility, we need to bind tools to the model first.
        """
        # Build system prompt for the agent
        system_prompt = """You are a helpful assistant.

    Answer the user's question directly.

    You have access to tools (Wikipedia, ArXiv, web search). Use tools ONLY when you need external facts or up-to-date information. If the question is general and you already know the answer, respond without calling tools.

    If you use tools, summarize the relevant findings and then provide a clear final answer."""

        # CRITICAL: Bind tools to Groq model for proper function calling
        # Without this, Groq generates invalid tool call format
        model_with_tools = self.llm.bind_tools(self.tools)
        
        # Create agent using modern API (LangChain 1.2.0+)
        # This automatically uses LangGraph internally
        agent_graph = create_agent(
            model=model_with_tools,  # Use bound model
            tools=self.tools,
            system_prompt=system_prompt
        )
        
        return agent_graph
    
    def run(self, query: str, **kwargs) -> str:
        """
        Execute agent reasoning and tool usage for a user query.
        
        This is the main public interface for the agent. It:
        1. Validates the input query
        2. Passes query to the agent executor
        3. Agent reasons about which tools to use
        4. Agent executes tools and observes results
        5. Agent returns final answer
        
        The agent is stateless - each call is independent. For conversational
        context, use ConversationMemoryManager separately and pass history
        in the query or via kwargs.
        
        Args:
            query: User's question or request (string)
            **kwargs: Additional parameters
                - max_iterations: Override default max iterations
                - return_intermediate_steps: Get reasoning trace
                - callbacks: LangChain callbacks for monitoring
        
        Returns:
            Agent's final answer as a string
        
        Raises:
            ValueError: If query is empty or invalid
            Exception: Propagates any unhandled errors with context
        
        Example:
            >>> agent = KnowledgeAgent(llm, tools)
            >>> 
            >>> # Simple query
            >>> response = agent.run("What is Python?")
            >>> 
            >>> # Query with options
            >>> response = agent.run(
            ...     "Latest AI news",
            ...     max_iterations=5,
            ...     return_intermediate_steps=True
            ... )
        
        TODO:
        - Add query preprocessing (spell check, expansion)
        - Support query context/history injection
        - Add response post-processing (formatting, citations)
        - Implement response streaming for long queries
        """
        # Input validation
        if not query:
            raise ValueError("Query cannot be empty")
        
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query)}")
        
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be only whitespace")
        
        # Validate kwargs to prevent conflicts
        reserved_keys = ['return_intermediate_steps']
        conflicting_keys = [k for k in kwargs if k in reserved_keys]
        if conflicting_keys:
            raise ValueError(
                f"Cannot override reserved parameters: {conflicting_keys}. "
                f"Use run_with_trace() for intermediate steps."
            )
        
        try:
            # Modern create_agent API (LangChain 1.2.0+)
            # IMPORTANT: create_agent graphs operate on message state.
            # Passing {"input": ...} can be ignored depending on the state schema.
            result = self.agent_executor.invoke(
                {"messages": [("user", query)]},
                **kwargs
            )
            
            # Extract response - modern API returns dict with "messages" key
            # containing a list of AIMessage objects
            if isinstance(result, dict) and "messages" in result:
                # Get the last AI message content
                messages = result["messages"]
                if messages and hasattr(messages[-1], 'content'):
                    answer = messages[-1].content
                else:
                    answer = ""
            elif isinstance(result, dict) and "output" in result:
                # Fallback for older format
                answer = result.get("output", "")
            else:
                answer = str(result)
            
            # Handle empty response
            if not answer or not answer.strip():
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            return answer.strip()
            
        except ValueError as e:
            # Handle parsing or validation errors
            error_msg = str(e)
            if "Could not parse" in error_msg or "parsing" in error_msg.lower():
                return (
                    "I encountered an error processing your request. "
                    "This might be due to the complexity of the query. "
                    "Please try rephrasing or breaking it into smaller questions."
                )
            raise
            
        except Exception as e:
            # Catch and provide context for unexpected errors
            error_type = type(e).__name__
            error_msg = str(e)

            # Groq models vary in tool-calling support. If tool calling fails,
            # fall back to a direct model response (no tools) instead of returning
            # a generic greeting or crashing the UI.
            if any(
                marker in error_msg.lower()
                for marker in [
                    "tool_use_failed",
                    "tool calling",
                    "failed to call a function",
                    "tool call validation failed",
                ]
            ):
                try:
                    direct = self.llm.invoke(query)
                    if hasattr(direct, "content"):
                        return str(direct.content).strip()
                    return str(direct).strip()
                except Exception:
                    # If fallback also fails, continue with original error handling
                    pass
            
            # Log error (in production, use proper logging)
            if self.verbose:
                print(f"Agent error ({error_type}): {error_msg}")
            
            # Re-raise with context (safely handle different exception types)
            try:
                raise type(e)(
                    f"Agent execution failed for query: '{query[:50]}...'. "
                    f"Original error: {error_msg}"
                ) from e
            except TypeError:
                # Exception type doesn't accept string message, re-raise original
                raise e from e
    
    def run_with_trace(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute agent with detailed execution trace.
        
        Similar to run() but returns both the answer and intermediate steps
        showing the agent's reasoning process, tool usage, and observations.
        
        Args:
            query: User's question
            **kwargs: Additional parameters
        
        Returns:
            Dictionary containing:
                - answer: Final response string
                - intermediate_steps: List of (action, observation) tuples
                - iterations: Number of reasoning steps taken
        
        Example:
            >>> result = agent.run_with_trace("What is quantum computing?")
            >>> print(result['answer'])
            >>> for step in result['intermediate_steps']:
            ...     action, observation = step
            ...     print(f"Tool: {action.tool}, Input: {action.tool_input}")
            ...     print(f"Result: {observation[:100]}...")
        
        TODO:
        - Add timing information per step
        - Include token usage statistics
        - Add tool selection reasoning
        """
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Validate kwargs to prevent conflicts
        if 'return_intermediate_steps' in kwargs:
            raise ValueError(
                "Cannot override 'return_intermediate_steps' in run_with_trace(). "
                "This method always returns intermediate steps."
            )
        
        try:
            # Modern create_agent returns message history, including tool calls.
            result = self.agent_executor.invoke(
                {"messages": [("user", query)]},
                **kwargs
            )

            messages = []
            answer = ""
            if isinstance(result, dict) and "messages" in result:
                messages = result["messages"]
                if messages and hasattr(messages[-1], "content"):
                    answer = messages[-1].content or ""

            return {
                "answer": (answer or "").strip(),
                "messages": messages,
            }
            
        except Exception as e:
            raise Exception(f"Agent trace execution failed: {str(e)}") from e
    
    def get_available_tools(self) -> List[str]:
        """
        Get names of all tools available to the agent.
        
        Returns:
            List of tool names
        
        Example:
            >>> tools = agent.get_available_tools()
            >>> print(f"Available tools: {', '.join(tools)}")
        """
        return [tool.name for tool in self.tools]
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all available tools.
        
        Returns:
            Dictionary mapping tool names to descriptions
        
        Example:
            >>> descriptions = agent.get_tool_descriptions()
            >>> for name, desc in descriptions.items():
            ...     print(f"{name}: {desc[:50]}...")
        """
        return {tool.name: tool.description for tool in self.tools}


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def create_knowledge_agent(
    llm=None,
    tools: Optional[List[BaseTool]] = None,
    retriever=None,
    verbose: bool = False,
    **kwargs
) -> KnowledgeAgent:
    """
    Convenience function to create a fully configured KnowledgeAgent.
    
    This factory handles the common setup pattern:
    1. Create LLM if not provided
    2. Create tools if not provided
    3. Initialize agent with configuration
    
    Args:
        llm: Optional LLM instance (creates default if None)
        tools: Optional list of tools (creates all tools if None)
        retriever: Optional RAG retriever for PDF knowledge tool
        verbose: Whether to print agent reasoning
        **kwargs: Additional agent configuration
    
    Returns:
        Configured KnowledgeAgent instance
    
    Example:
        >>> # Create with defaults
        >>> agent = create_knowledge_agent(verbose=True)
        >>> 
        >>> # Create with custom LLM
        >>> from app.core.llm import LLMFactory
        >>> llm = LLMFactory.create()
        >>> agent = create_knowledge_agent(llm=llm, verbose=True)
        >>> 
        >>> # Create with retriever for PDF search
        >>> from app.rag.retriever import create_rag_retriever
        >>> retriever = create_rag_retriever()
        >>> agent = create_knowledge_agent(retriever=retriever)
    
    TODO:
    - Add preset configurations (fast, accurate, research, etc.)
    - Support configuration from settings file
    - Add validation and compatibility checks
    """
    # Create LLM if not provided
    if llm is None:
        from app.core.llm import LLMFactory
        llm = LLMFactory.create()  # Uses default from settings
    
    # Create tools if not provided
    if tools is None:
        from app.agents.tools import create_all_tools
        tools = create_all_tools(retriever=retriever)
    
    # Validate we have tools
    if not tools:
        raise ValueError("No tools available. Cannot create agent without tools.")
    
    # Create and return agent
    return KnowledgeAgent(
        llm=llm,
        tools=tools,
        verbose=verbose,
        **kwargs
    )

__all__ = ["KnowledgeAgent", "create_knowledge_agent"]
