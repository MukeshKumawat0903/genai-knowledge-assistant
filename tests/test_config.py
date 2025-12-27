"""
Configuration System Test/Demo Script

This script demonstrates how to use the centralized configuration system.
Run this to verify your .env file is set up correctly.

Usage:
    python test_config.py
"""

from app.utils.config import get_settings, reload_settings


def main():
    """Test and display current configuration."""
    
    print("=" * 70)
    print("GenAI Knowledge Assistant - Configuration Test")
    print("=" * 70)
    print()
    
    try:
        # Get settings (singleton - only loads once)
        settings = get_settings()
        
        print("✅ Configuration loaded successfully!")
        print()
        
        # Display Application Settings
        print("📋 Application Settings:")
        print(f"   • App Name: {settings.app_name}")
        print(f"   • Environment: {settings.environment}")
        print(f"   • Debug Mode: {settings.debug}")
        print()
        
        # Display LLM Configuration
        print("🤖 LLM Configuration:")
        print(f"   • Provider: {settings.llm_provider}")
        print(f"   • Model: {settings.llm_model_name}")
        print(f"   • Temperature: {settings.llm_temperature}")
        print(f"   • Max Tokens: {settings.llm_max_tokens or 'Default'}")
        
        # Check API key (without exposing it)
        api_key = settings.get_api_key()
        if api_key:
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            print(f"   • API Key: {masked_key} ✓")
        else:
            print(f"   • API Key: ❌ NOT SET")
        print()
        
        # Display Embedding Configuration
        print("🔢 Embedding Configuration:")
        print(f"   • Provider: {settings.embedding_provider}")
        print(f"   • Model: {settings.embedding_model_name}")
        print()
        
        # Display Vector Store Configuration
        print("💾 Vector Store Configuration:")
        print(f"   • Type: {settings.vector_store_type}")
        print(f"   • Path: {settings.vector_store_path}")
        print(f"   • Collection: {settings.collection_name}")
        print()
        
        # Display RAG Configuration
        print("🔍 RAG Configuration:")
        print(f"   • Chunk Size: {settings.chunk_size} characters")
        print(f"   • Chunk Overlap: {settings.chunk_overlap} characters")
        print(f"   • Retriever Top-K: {settings.retriever_top_k}")
        print(f"   • Similarity Threshold: {settings.similarity_threshold}")
        print()
        
        # Display Agent Configuration
        print("🤖 Agent Configuration:")
        print(f"   • Max Iterations: {settings.agent_max_iterations}")
        print(f"   • Tools Enabled: {settings.enable_agent_tools}")
        print()
        
        # Display Data Paths
        print("📁 Data Paths:")
        print(f"   • Data Directory: {settings.data_dir}")
        print(f"   • Raw Data: {settings.raw_data_dir}")
        print(f"   • Processed Data: {settings.processed_data_dir}")
        print()
        
        print("=" * 70)
        print("✅ All configuration validated successfully!")
        print("=" * 70)
        print()
        print("💡 Tips:")
        print("   • Edit .env file to change settings")
        print("   • Run this script again to verify changes")
        print("   • See .env.example for all available options")
        print()
        
        # Show how to use in code
        print("📝 Usage in your code:")
        print()
        print("   from app.utils.config import get_settings")
        print()
        print("   settings = get_settings()")
        print("   print(settings.llm_provider)")
        print("   print(settings.chunk_size)")
        print()
        
    except ValueError as e:
        print()
        print("❌ Configuration Error:")
        print(f"   {str(e)}")
        print()
        print("💡 Solutions:")
        print("   1. Copy .env.example to .env")
        print("   2. Set required API keys in .env")
        print("   3. Check all values are in valid ranges")
        print()
        return 1
    
    except Exception as e:
        print()
        print(f"❌ Unexpected Error: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
