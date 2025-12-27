# LLM Abstraction - Quick Reference Card

## 🚀 One-Line Usage

```python
from app.core.llm import LLMFactory
llm = LLMFactory.create()  # That's it!
```

## 📋 Configuration (.env)

### Groq (Cloud - Recommended)
```bash
LLM_PROVIDER=groq
LLM_MODEL_NAME=llama-3.3-70b-versatile
GROQ_API_KEY=your-key-here
```

## 💡 Common Usage Patterns

### Basic Query
```python
llm = LLMFactory.create()
response = llm.invoke("What is machine learning?")
print(response.content)
```

### Chat Messages
```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain RAG in one sentence.")
]
response = llm.invoke(messages)
```

### RAG Chain
```python
from langchain_core.prompts import ChatPromptTemplate

llm = LLMFactory.create()
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use this context: {context}"),
    ("human", "{question}")
])

chain = prompt | llm
response = chain.invoke({
    "context": "...",
    "question": "..."
})
```

## 🔄 Switch Providers

Just edit `.env` - no code changes needed!

| Provider | .env Setting | API Key Needed? |
|----------|-------------|-----------------|
| Groq     | `LLM_PROVIDER=groq` | Yes (free) |

## 🎯 Architecture (3 Classes)

```
BaseLLM (abstract)
   ↓
   ├── GroqLLM
   └── [Future providers]

LLMFactory.create() → Returns ready-to-use LLM
```

## 🧪 Testing

```powershell
python tests/test_config.py  # Verify setup
python tests/test_llm.py     # Test LLM (makes real API call)
```

## 🆘 Quick Troubleshooting

| Error | Solution |
|-------|----------|
| "API key not found" | Add `GROQ_API_KEY` to .env |
| "Unsupported provider" | Check `LLM_PROVIDER` in .env |

## 📚 Documentation Files

- **INSTALLATION.md** - Setup guide
- **docs/LLM_ABSTRACTION_GUIDE.md** - Complete usage guide
- **docs/ARCHITECTURE_DIAGRAM.md** - Visual diagrams
- **docs/reference/IMPLEMENTATION_SUMMARY.md** - What was built

## 🎓 Interview Points

**Q: Why abstraction?**
A: Prevents vendor lock-in, consistent interface, easy testing

**Q: Design patterns?**
A: Factory (creation), Abstract Base Class (interface), Singleton (config)

**Q: SOLID principles?**
A: Single Responsibility (✓), Open/Closed (✓), Liskov (✓), Interface Segregation (✓), Dependency Inversion (✓)

**Q: How to add provider?**
A: 
1. Create class: `class NewLLM(BaseLLM)`
2. Update factory: `elif provider == "new": return NewLLM().get_llm()`
3. Update config validation
4. Done! No consumer code changes needed

## 🔑 Key Files

```
app/core/llm.py          # Main implementation
test_llm.py              # Test script
.env                     # Your configuration
docs/                    # Complete documentation
```

## ⚡ Supported Models

### Groq
- llama-3.3-70b-versatile ⭐ (Recommended)
- llama-3.1-70b-versatile
- mixtral-8x7b-32768

## 🎉 Get Started

```powershell
# 1. Install
pip install -r requirements.txt

# 2. Configure
New-Item -ItemType File -Path .env -Force | Out-Null
# Edit .env with your settings (LLM_PROVIDER=groq, GROQ_API_KEY=...)

# 3. Test
python tests/test_config.py
python tests/test_llm.py

# 4. Use
from app.core.llm import LLMFactory
llm = LLMFactory.create()
```

---

**Remember**: Configuration in .env, usage is always the same!
