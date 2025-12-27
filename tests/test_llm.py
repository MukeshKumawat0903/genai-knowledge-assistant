"""
LLM Abstraction Layer Test/Demo Script

This script demonstrates how to use the LLM factory and abstraction layer.
It shows how easy it is to switch between providers.

Usage:
    # Make sure .env is configured with LLM_PROVIDER and API keys
    python test_llm.py
"""

from app.core.llm import LLMFactory
from app.utils.config import get_settings


def main():
    """Test the LLM abstraction layer."""
    
    print("=" * 70)
    print("LLM Abstraction Layer - Test & Demo")
    print("=" * 70)
    print()
    
    try:
        # Get settings
        settings = get_settings()
        
        print(f"📋 Configuration:")
        print(f"   • Provider: {settings.llm_provider}")
        print(f"   • Model: {settings.llm_model_name}")
        print(f"   • Temperature: {settings.llm_temperature}")
        print(f"   • Max Tokens: {settings.llm_max_tokens or 'Default'}")
        print()
        
        # Create LLM using factory
        print("🏭 Creating LLM using Factory pattern...")
        llm = LLMFactory.create()
        print(f"✅ Successfully created: {type(llm).__name__}")
        print()
        
        # Test a simple query
        print("💬 Testing LLM with a simple query...")
        print("   Query: 'Explain RAG in one sentence.'")
        print()
        
        test_prompt = "Explain Retrieval-Augmented Generation (RAG) in one sentence."
        
        print("   🤔 Generating response...")
        response = llm.invoke(test_prompt)
        
        print()
        print("   📝 Response:")
        print(f"   {response.content}")
        print()
        
        # Show how to use with chat messages
        print("=" * 70)
        print("💡 Usage Pattern - Chat Messages:")
        print("=" * 70)
        print()
        print("from langchain_core.messages import HumanMessage, SystemMessage")
        print()
        print("messages = [")
        print("    SystemMessage(content='You are a helpful AI assistant.'),")
        print("    HumanMessage(content='What is machine learning?')")
        print("]")
        print()
        print("response = llm.invoke(messages)")
        print("print(response.content)")
        print()
        
        # Show how to switch providers
        print("=" * 70)
        print("🔄 How to Switch Providers:")
        print("=" * 70)
        print()
        print("This repo is configured for Groq-only.")
        print()
        print("Just update your .env file:")
        print()
        print("   LLM_PROVIDER=groq")
        print("   LLM_MODEL_NAME=llama-3.3-70b-versatile")
        print("   GROQ_API_KEY=your-key-here")
        print()
        print("No code changes needed - the factory reads Settings.")
        print()
        
        # Show integration example
        print("=" * 70)
        print("🔗 Integration in Your RAG Pipeline:")
        print("=" * 70)
        print()
        print("from app.core.llm import LLMFactory")
        print("from langchain_core.prompts import ChatPromptTemplate")
        print()
        print("# Create LLM")
        print("llm = LLMFactory.create()")
        print()
        print("# Create RAG prompt")
        print("prompt = ChatPromptTemplate.from_messages([")
        print("    ('system', 'Use this context: {context}'),")
        print("    ('human', '{question}')")
        print("])")
        print()
        print("# Create chain")
        print("chain = prompt | llm")
        print()
        print("# Use it")
        print("response = chain.invoke({")
        print("    'context': 'Retrieved documents...',")
        print("    'question': 'What is the capital of France?'")
        print("})")
        print()
        
        print("=" * 70)
        print("✅ Test completed successfully!")
        print("=" * 70)
        print()
        
    except ValueError as e:
        print()
        print("❌ Configuration Error:")
        print(f"   {str(e)}")
        print()
        print("💡 Solutions:")
        if "Groq" in str(e):
            print("   1. Get free Groq API key: https://console.groq.com/keys")
            print("   2. Add to .env: GROQ_API_KEY=your-key-here")
        else:
            print("   1. Check your .env file configuration")
            print("   2. Run: python tests/test_config.py")
        print()
        return 1
    
    except Exception as e:
        print()
        print(f"❌ Error: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("💡 Troubleshooting:")
        print("   • Ensure your .env file is configured")
        print("   • Run: python tests/test_config.py")
        print("   • Check that API keys are valid")
        print()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
