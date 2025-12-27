# Embedding Manager - Quick Reference

## 🚀 One-Line Usage

```python
from app.core.embeddings import EmbeddingManager

manager = EmbeddingManager()
embeddings = manager.get_embeddings()  # That's it!
```

## 📋 Configuration (.env)

```bash
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

## 💡 Common Patterns

### Embed Single Text
```python
vector = embeddings.embed_query("What is machine learning?")
print(f"Dimensions: {len(vector)}")  # 384
```

### Embed Multiple Texts (Faster)
```python
texts = ["Text 1", "Text 2", "Text 3"]
vectors = embeddings.embed_documents(texts)  # Batch processing
```

### Calculate Similarity
```python
import numpy as np

similarity = np.dot(vec1, vec2)  # Works because normalized
# Range: 0 (unrelated) to 1 (identical)
```

### Use in Vector Store
```python
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

results = vector_store.similarity_search("query", k=5)
```

## 🏗️ Architecture (Simple)

```
EmbeddingManager
    ↓ get_embeddings()
HuggingFaceEmbeddings (cached)
    ↓ embed_query() / embed_documents()
Vectors [0.23, -0.45, ..., 0.12]
```

## 🎯 Lazy Initialization

```python
manager = EmbeddingManager()        # No model loaded
embeddings = manager.get_embeddings()  # Loads here (10-30s first time)
embeddings2 = manager.get_embeddings() # Instant (cached)
```

## 📊 Model Options

| Model | Dims | Size | Speed | Best For |
|-------|------|------|-------|----------|
| all-MiniLM-L6-v2 ⭐ | 384 | 80MB | Fast | General use |
| all-mpnet-base-v2 | 768 | 420MB | Medium | Better quality |

⭐ = Recommended

## 🧪 Testing

```powershell
# Verify config
python test_config.py

# Test embeddings
python test_embeddings.py

# Interactive demo
jupyter notebook notebooks/embedding_demo.ipynb
```

## 🆘 Quick Fixes

| Error | Fix |
|-------|-----|
| "Model name not configured" | Add `EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2` to .env |
| "Failed to initialize" | Check internet, try smaller model |
| Slow first load | Expected (downloads model), later calls instant |

## 🎓 Interview Points

**Q: Why lazy initialization?**
A: Models are 80-500MB. Load only when needed, cache for reuse.

**Q: Why normalize embeddings?**
A: Unit vectors enable fast similarity: `dot(a, b)` instead of `cosine(a, b)`

**Q: How to scale?**
A: GPU (10x faster), batch processing, cloud embeddings for huge scale

## 📚 Files

```
app/core/embeddings.py           # Implementation
test_embeddings.py               # Test script
notebooks/embedding_demo.ipynb   # Interactive demo
docs/EMBEDDING_GUIDE.md          # Full guide
```

## 🔗 Complete RAG Example

```python
from app.core.embeddings import EmbeddingManager
from app.core.llm import LLMFactory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Setup
embeddings = EmbeddingManager().get_embeddings()
llm = LLMFactory.create()

# Create vector store
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# RAG prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use this context: {context}"),
    ("human", "{question}")
])

# Chain
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt
    | llm
)

# Use
response = chain.invoke("What is RAG?")
print(response.content)
```

---

**Remember**: Load once, cache, reuse everywhere!
