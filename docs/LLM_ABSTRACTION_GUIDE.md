# LLM Abstraction Layer - Quick Reference

## 🎯 Purpose

Clean, extensible abstraction for LLM providers that prevents vendor lock-in and makes it trivial to switch between cloud and local models.

## 🏗️ Architecture

```
BaseLLM (Abstract)
    ↓
    ├── GroqLLM (Cloud - Fast inference)
    └── [Future: OpenAILLM, AnthropicLLM, GoogleLLM]
    
LLMFactory
    └── create() → Returns configured LangChain chat model
```

## 🚀 Quick Start

### 1. Configure Provider (in .env)

```bash
# Option A: Use Groq (Cloud - Free tier available)
LLM_PROVIDER=groq
LLM_MODEL_NAME=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.7
GROQ_API_KEY=your-groq-api-key
```

### 2. Use the Factory

```python
from app.core.llm import LLMFactory

# Create LLM - factory reads from Settings
llm = LLMFactory.create()

# Use it like any LangChain chat model
response = llm.invoke("What is RAG?")
print(response.content)
```

## 📚 Usage Examples

### Basic Text Generation

```python
from app.core.llm import LLMFactory

llm = LLMFactory.create()
response = llm.invoke("Explain machine learning in simple terms.")
print(response.content)
```

### Chat with Messages

```python
from app.core.llm import LLMFactory
from langchain_core.messages import HumanMessage, SystemMessage

llm = LLMFactory.create()

messages = [
    SystemMessage(content="You are a helpful AI assistant specialized in Python."),
    HumanMessage(content="How do I read a CSV file in pandas?")
]

response = llm.invoke(messages)
print(response.content)
```

### Integration with LangChain Chains

```python
from app.core.llm import LLMFactory
from langchain_core.prompts import ChatPromptTemplate

# Create LLM
llm = LLMFactory.create()

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {topic}."),
    ("human", "{question}")
])

# Create chain using LCEL (LangChain Expression Language)
chain = prompt | llm

# Invoke
response = chain.invoke({
    "topic": "artificial intelligence",
    "question": "What is a neural network?"
})

print(response.content)
```

### RAG Chain Example

```python
from app.core.llm import LLMFactory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

llm = LLMFactory.create()

# RAG prompt template
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on this context:\n\n{context}"),
    ("human", "{question}")
])

# Create RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
)

# Use it
answer = rag_chain.invoke("What is the capital of France?")
print(answer.content)
```

## 🔄 Switching Providers

**Zero code changes required!** Just update your `.env` file:

### Switch to Groq (Cloud)

```bash
LLM_PROVIDER=groq
LLM_MODEL_NAME=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_your_key_here
```
The factory automatically handles provider-specific initialization.

## 🎓 Design Patterns Used

### 1. Abstract Base Class Pattern
- `BaseLLM` defines the contract
- All implementations must provide `get_llm()`
- Ensures consistent interface across providers

### 2. Factory Pattern
- `LLMFactory.create()` centralizes object creation
- Decouples consumer code from concrete implementations
- Easy to add new providers without breaking existing code

### 3. Dependency Inversion
- High-level code depends on `BaseLLM` abstraction
- Not on concrete implementations like `GroqLLM`
- Follows SOLID principles

## 🔌 Adding New Providers

Template for adding a new provider:

```python
class NewProviderLLM(BaseLLM):
    """
    NewProvider LLM implementation.
    
    Description of the provider and its models.
    """
    
    def get_llm(self) -> ChatNewProvider:
        """Create and return ChatNewProvider instance."""
        settings = get_settings()
        
        # Validate API key
        if not settings.newprovider_api_key:
            raise ValueError(
                "NewProvider API key not found. "
                "Set NEWPROVIDER_API_KEY in .env"
            )
        
        # Return LangChain-compatible chat model
        return ChatNewProvider(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.newprovider_api_key,
        )
```

Then update `LLMFactory.create()`:

```python
elif provider == "newprovider":
    return NewProviderLLM().get_llm()
```

## 🧪 Testing

```bash
# Test configuration
python tests/test_config.py

# Test LLM abstraction
python tests/test_llm.py
```

## 🎯 Interview-Ready Points

### Question: "Why use an abstraction layer?"

**Answer:**
1. **Vendor Lock-in Prevention**: Easy to switch providers
2. **Consistent Interface**: All LLMs expose same API
3. **Testability**: Can mock `BaseLLM` for unit tests
4. **Maintainability**: Provider changes isolated to one class
5. **Scalability**: Add providers without changing consumer code

### Question: "Explain the Factory pattern"

**Answer:**
The Factory pattern is a creational design pattern that provides an interface for creating objects without specifying their exact class. In our case:

- `LLMFactory.create()` reads configuration
- Returns the appropriate LLM instance
- Consumer code never directly instantiates `GroqLLM`
- Makes it easy to add providers or change creation logic

### Question: "How does this follow SOLID principles?"

**Answer:**
- **S**ingle Responsibility: Each class has one job
  - `GroqLLM`: Initialize Groq
  - `LLMFactory`: Create LLMs
- **O**pen/Closed: Open for extension (add providers), closed for modification
- **L**iskov Substitution: Any `BaseLLM` subclass works interchangeably
- **I**nterface Segregation: `BaseLLM` has minimal interface
- **D**ependency Inversion: Depend on `BaseLLM` abstraction, not concrete classes

## 📊 Supported Models

### Groq (Fast Cloud Inference)
- `llama-3.3-70b-versatile` (Recommended)
- `llama-3.1-70b-versatile`
- `mixtral-8x7b-32768`
- `gemma2-9b-it`

## 🔗 Resources

- **Groq Console**: https://console.groq.com/keys
- **LangChain Docs**: https://python.langchain.com/docs/

## 💡 Best Practices

1. **Always use the factory**: `LLMFactory.create()` instead of direct instantiation
2. **Keep API keys in .env**: Never hardcode credentials
3. **Handle errors gracefully**: Factory provides clear error messages
4. **Test with different providers**: Ensure your code works with any LLM
5. **Use type hints**: Makes code self-documenting and catches errors early

## 🐛 Troubleshooting

### "Groq API key not found"
```bash
# Get free API key
Visit: https://console.groq.com/keys

# Add to .env
GROQ_API_KEY=gsk_your_key_here
```

### "Unsupported LLM provider"
```bash
# Check .env file
LLM_PROVIDER must be: groq
```

---

**Summary**: This abstraction layer provides a clean, extensible, production-ready interface for working with multiple LLM providers while preventing vendor lock-in and following software engineering best practices.
