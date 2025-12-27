# Installation & Setup Guide

## Prerequisites

- Python 3.9 or higher
- pip package manager

## Quick Setup (5 minutes)

### Step 1: Clone or Navigate to Project

```powershell
cd genai-knowledge-assistant
```

### Step 2: Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### Step 3: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes depending on your internet speed.

### Step 4: Configure Environment

```powershell
# Create a .env file in the project root (do NOT commit it)
New-Item -ItemType File -Path .env -Force | Out-Null

# Edit .env file with your preferred editor
notepad .env  # Or use VS Code: code .env
```

**Minimum configuration:**

```bash
# LLM provider used in this repo
LLM_PROVIDER=groq

# If using Groq (Cloud - Free tier available)
GROQ_API_KEY=your-groq-api-key-here

# Model configuration
LLM_MODEL_NAME=llama-3.3-70b-versatile  # for Groq

# Vector store (supported in this repo: faiss, chroma)
VECTOR_STORE_TYPE=faiss
```

### Step 5: Get API Keys (If Using Cloud LLMs)

#### Groq (Recommended - Fast & Free)
1. Visit: https://console.groq.com/keys
2. Sign up for free account
3. Create new API key
4. Copy and paste into `.env` file

### Step 6: Verify Installation

```powershell
# Test configuration
python tests/test_config.py

# Test LLM (Groq makes an actual API call)
python tests/test_llm.py
```

## Troubleshooting

### "pip install" fails

**Problem**: Some packages fail to install

**Solutions**:
```powershell
# Update pip first
python -m pip install --upgrade pip

# Install with verbose output to see errors
pip install -r requirements.txt -v

# Install problematic packages individually
pip install langchain langchain-groq
```

### "Import could not be resolved"

**Problem**: IDE shows import errors

**Solution**: Make sure virtual environment is activated and packages are installed
```powershell
# Check which Python is active
Get-Command python

# Should show path to venv\Scripts\python.exe
# If not, activate venv again:
.\venv\Scripts\activate

# Verify packages
pip list | Select-String langchain
```

### "Groq API key not found"

**Problem**: API key not loaded from .env

**Solutions**:
1. Verify `.env` file exists in project root
2. Check no extra spaces: `GROQ_API_KEY=your-key` (no spaces around `=`)
3. Restart Python if environment was just edited
4. Run `python tests/test_config.py` to verify

## Package Overview

### Core Dependencies
- `langchain` - LLM orchestration framework
- `langchain-groq` - Groq LLM integration
- `streamlit` - Web UI framework
- `python-dotenv` - Environment variable management

### Vector Stores
- `faiss-cpu` - Fast similarity search (local)
- `chromadb` - Persistent vector store

> Note: A Pinecone backend is mentioned in code/docs as a future extension,
> but it is not implemented end-to-end in this repo.

### Document Processing
- `pypdf` - PDF parsing
- `reportlab` - PDF generation (used by tests)
- `beautifulsoup4` - Web scraping
- `youtube-transcript-api` - YouTube transcripts

### Utilities
- `sentence-transformers` - Text embeddings
- `tiktoken` - Token counting utilities
- `numpy` - Numeric utilities
- `pydantic` - Settings/config validation

## Development (Optional)

Run the unit tests:

```powershell
python -m pytest -q
```

## Environment Variables Reference

Key variables:

```bash
# Application
APP_NAME="GenAI Knowledge Assistant"
ENVIRONMENT=development  # or production
DEBUG=true

# LLM Configuration
LLM_PROVIDER=groq
LLM_MODEL_NAME=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096

# API Keys (only needed for Groq)
GROQ_API_KEY=

# Vector Store
VECTOR_STORE_TYPE=faiss  # supported: faiss, chroma
VECTOR_STORE_PATH=./data/vector_store

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_TOP_K=5
SIMILARITY_THRESHOLD=0.7

# Agent Configuration
AGENT_MAX_ITERATIONS=10
ENABLE_AGENT_TOOLS=true
```

## Next Steps

After successful installation:

1. ✅ **Test your setup**: Run `python tests/test_llm.py`
2. 📚 **Read documentation**: Check `docs/LLM_ABSTRACTION_GUIDE.md`
3. 🚀 **Start building**: Explore `app/` modules
4. 💻 **Run the app**: `streamlit run run.py`

## Getting Help

- Check [LLM_ABSTRACTION_GUIDE.md](docs/LLM_ABSTRACTION_GUIDE.md) for LLM usage
- Review [README.md](README.md) for architecture overview
- Run test scripts for diagnostic information
- Open an issue if problems persist

## System Requirements

**Minimum**:
- CPU: Dual-core processor
- RAM: 4GB
- Storage: 2GB for dependencies
- Internet: Required for cloud LLMs

**Cloud LLMs** (Groq):
- CPU: Any modern processor
- RAM: 2GB
- No GPU needed
- Stable internet connection required

---

**Ready to start?** Run `python tests/test_config.py` to verify your setup!
