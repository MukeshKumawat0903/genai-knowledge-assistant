"""
Agent Tools Demo

This demo showcases the modular agent tools for the GenAI Knowledge Assistant.
Each tool demonstrates how to fetch information from different sources.

Features demonstrated:
1. PDF Knowledge Search (local RAG)
2. Web Search (DuckDuckGo)
3. Wikipedia Search
4. ArXiv Academic Search
5. Tool Registry management
"""

from app.agents.tools import (
    create_all_tools,
    create_pdf_knowledge_tool,
    create_web_search_tool,
    create_wikipedia_tool,
    create_arxiv_tool,
    ToolRegistry,
    get_tool_by_name
)


def demo_individual_tools():
    """
    Demo: Test each tool individually
    
    Shows how to create and use each tool separately.
    """
    print("=" * 70)
    print("DEMO 1: Individual Tool Usage")
    print("=" * 70)
    
    # 1. PDF Knowledge Search Tool
    print("\n📚 PDF Knowledge Search Tool")
    print("-" * 70)
    pdf_tool = create_pdf_knowledge_tool()  # No retriever - will show initialization message
    result = pdf_tool.run("What is machine learning?")
    print(f"Query: 'What is machine learning?'")
    print(f"Result:\n{result[:200]}...")  # Show first 200 chars
    
    # 2. Web Search Tool
    print("\n🌐 Web Search Tool")
    print("-" * 70)
    web_tool = create_web_search_tool()
    query = "latest AI breakthroughs 2024"
    print(f"Query: '{query}'")
    print("Note: Performing real web search...")
    try:
        result = web_tool.run(query)
        print(f"Result:\n{result[:300]}...")  # Show first 300 chars
    except Exception as e:
        print(f"Web search error (expected if offline): {e}")
    
    # 3. Wikipedia Tool
    print("\n📖 Wikipedia Search Tool")
    print("-" * 70)
    wiki_tool = create_wikipedia_tool()
    query = "Artificial Intelligence"
    print(f"Query: '{query}'")
    try:
        result = wiki_tool.run(query)
        print(f"Result:\n{result[:400]}...")  # Show first 400 chars
    except Exception as e:
        print(f"Wikipedia error (expected if offline): {e}")
    
    # 4. ArXiv Tool
    print("\n🎓 ArXiv Academic Search Tool")
    print("-" * 70)
    arxiv_tool = create_arxiv_tool()
    query = "transformer neural networks"
    print(f"Query: '{query}'")
    try:
        result = arxiv_tool.run(query)
        print(f"Result:\n{result[:400]}...")  # Show first 400 chars
    except Exception as e:
        print(f"ArXiv error (expected if offline): {e}")
    
    print("\n✅ Individual tools demo complete!")


def demo_tool_registry():
    """
    Demo: Using ToolRegistry for centralized tool management
    
    Shows how to use the registry pattern for managing multiple tools.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Tool Registry")
    print("=" * 70)
    
    # Create tool registry
    registry = ToolRegistry()
    
    # List all available tools
    print("\n📋 Available Tools:")
    print("-" * 70)
    for tool_name in registry.get_tool_names():
        print(f"  - {tool_name}")
    
    # Get tool descriptions
    print("\n📝 Tool Descriptions:")
    print("-" * 70)
    descriptions = registry.get_tool_descriptions()
    for name, desc in descriptions.items():
        print(f"\n{name}:")
        print(f"  {desc[:100]}...")  # First 100 chars
    
    # Get specific tool by name
    print("\n🔍 Get Specific Tool:")
    print("-" * 70)
    wiki_tool = registry.get_tool("wikipedia_search")
    print(f"Retrieved tool: {wiki_tool.name}")
    print(f"Description: {wiki_tool.description[:80]}...")
    
    # Get all tools (for agent use)
    print("\n🤖 All Tools for Agent:")
    print("-" * 70)
    all_tools = registry.get_all_tools()
    print(f"Total tools available: {len(all_tools)}")
    for tool in all_tools:
        print(f"  - {tool.name}")
    
    print("\n✅ Tool registry demo complete!")


def demo_tool_metadata():
    """
    Demo: Exploring tool metadata and structure
    
    Shows the metadata that helps agents understand and select tools.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Tool Metadata and Structure")
    print("=" * 70)
    
    # Create a sample tool
    wiki_tool = create_wikipedia_tool()
    
    print("\n🔍 Tool Attributes:")
    print("-" * 70)
    print(f"Name: {wiki_tool.name}")
    print(f"Description length: {len(wiki_tool.description)} chars")
    print(f"Description:\n{wiki_tool.description}")
    
    print("\n📊 Tool Interface:")
    print("-" * 70)
    print(f"Callable: {callable(wiki_tool.run)}")
    print(f"Can be invoked with: tool.run(query)")
    
    # Show how agent would see the tool
    print("\n🤖 Agent's View of Tool:")
    print("-" * 70)
    print(f"Tool Name: {wiki_tool.name}")
    print(f"When to use: {wiki_tool.description.split('.')[0]}")
    print(f"Input format: String query")
    print(f"Output format: String result")
    
    print("\n✅ Tool metadata demo complete!")


def demo_agent_integration_pattern():
    """
    Demo: How agents would use these tools
    
    Shows the pattern for integrating tools with LangChain agents.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Agent Integration Pattern")
    print("=" * 70)
    
    print("\n📋 Step-by-Step Agent Integration:")
    print("-" * 70)
    
    # Step 1: Create tools
    print("\n1️⃣  Create tools:")
    tools = create_all_tools()
    print(f"   Created {len(tools)} tools")
    
    # Step 2: Show tool names and descriptions (what agent sees)
    print("\n2️⃣  Tools available to agent:")
    for tool in tools:
        print(f"   - {tool.name}")
    
    # Step 3: Simulate agent's tool selection logic
    print("\n3️⃣  Agent's tool selection (simulated):")
    queries = [
        ("What is quantum computing?", "wikipedia_search"),
        ("Latest AI news", "web_search"),
        ("Research on transformers", "arxiv_search"),
        ("Company policy on remote work", "pdf_knowledge_search")
    ]
    
    for query, expected_tool in queries:
        print(f"\n   Query: '{query}'")
        print(f"   Agent thinks: \"This needs {expected_tool}\"")
        print(f"   Rationale: {_get_tool_rationale(expected_tool)}")
    
    # Step 4: Show integration code pattern
    print("\n4️⃣  Integration code pattern:")
    print("""
   # Pattern 1: With LangChain Agent
   from langchain.agents import initialize_agent, AgentType
   from langchain.chat_models import ChatOpenAI
   
   llm = ChatOpenAI(temperature=0)
   tools = create_all_tools(retriever)
   
   agent = initialize_agent(
       tools=tools,
       llm=llm,
       agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
       verbose=True
   )
   
   response = agent.run("What is machine learning?")
   
   # Pattern 2: Manual tool selection
   query = "Latest AI news"
   web_tool = get_tool_by_name("web_search")
   result = web_tool.run(query)
    """)
    
    print("\n✅ Agent integration pattern demo complete!")


def demo_tool_comparison():
    """
    Demo: Comparing different tools for the same query
    
    Shows how different tools provide different types of information.
    """
    print("\n" + "=" * 70)
    print("DEMO 5: Tool Comparison")
    print("=" * 70)
    
    query = "artificial intelligence"
    print(f"\n🔎 Query: '{query}'")
    print("\nHow different tools would respond:")
    print("-" * 70)
    
    # Wikipedia: Encyclopedic knowledge
    print("\n📖 Wikipedia Tool:")
    print("   Response type: Encyclopedic definition and history")
    print("   Best for: General knowledge, definitions, historical context")
    print("   Strengths: Comprehensive, well-sourced, stable information")
    
    # Web Search: Current information
    print("\n🌐 Web Search Tool:")
    print("   Response type: Recent news, current trends, latest developments")
    print("   Best for: Current events, breaking news, trending topics")
    print("   Strengths: Up-to-date, diverse sources, real-time")
    
    # ArXiv: Academic research
    print("\n🎓 ArXiv Tool:")
    print("   Response type: Research papers, technical details, academic insights")
    print("   Best for: Scientific questions, cutting-edge research, technical depth")
    print("   Strengths: Peer-reviewed, authoritative, technical")
    
    # PDF Knowledge: Domain-specific
    print("\n📚 PDF Knowledge Tool:")
    print("   Response type: Company docs, internal policies, domain-specific info")
    print("   Best for: Internal knowledge, specific documents, proprietary info")
    print("   Strengths: Customized, controlled, domain-specific")
    
    print("\n💡 Tool Selection Guide:")
    print("-" * 70)
    print("  General knowledge → Wikipedia")
    print("  Current events → Web Search")
    print("  Research/Academic → ArXiv")
    print("  Company/Domain specific → PDF Knowledge")
    
    print("\n✅ Tool comparison demo complete!")


def _get_tool_rationale(tool_name: str) -> str:
    """Helper function to provide rationale for tool selection."""
    rationales = {
        "wikipedia_search": "General knowledge query → encyclopedic source",
        "web_search": "Needs current/recent information → web search",
        "arxiv_search": "Research/academic topic → scientific papers",
        "pdf_knowledge_search": "Domain-specific content → local knowledge base"
    }
    return rationales.get(tool_name, "Unknown tool")


def main():
    """Run all demos."""
    print("\n" + "🚀" * 35)
    print("AGENT TOOLS - COMPREHENSIVE DEMO")
    print("🚀" * 35)
    
    print("\n📌 Note: Some demos may show errors if offline or if LangChain")
    print("   dependencies are not installed. This is expected behavior.")
    
    try:
        # Run all demos
        demo_individual_tools()
        demo_tool_registry()
        demo_tool_metadata()
        demo_agent_integration_pattern()
        demo_tool_comparison()
        
        print("\n" + "=" * 70)
        print("🎉 ALL DEMOS COMPLETED!")
        print("=" * 70)
        
        print("\n📚 Key Takeaways:")
        print("  ✓ 4 specialized tools for different information sources")
        print("  ✓ LangChain Tool interface compatibility")
        print("  ✓ Clear names and descriptions for agent reasoning")
        print("  ✓ Lightweight wrappers (no answer generation)")
        print("  ✓ Easy integration with LangChain agents")
        print("  ✓ Registry pattern for centralized management")
        
        print("\n🔗 Next Steps:")
        print("  1. Create a LangChain agent with these tools")
        print("  2. Index documents for PDF knowledge search")
        print("  3. Test with real queries")
        print("  4. Implement tool usage logging")
        print("  5. Add result ranking and caching")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
