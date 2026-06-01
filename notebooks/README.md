# 📚 GenAI Knowledge Assistant - Learning Notebooks

Welcome to the comprehensive educational series for mastering Retrieval-Augmented Generation (RAG), Large Language Models, and Agentic AI systems!

## 🚀 Quick Start

1. **Start Here**: Open [`00_START_HERE.ipynb`](00_START_HERE.ipynb) for the complete learning roadmap
2. **Check Setup**: Run the setup verification cell to ensure your environment is ready
3. **Follow the Path**: Complete notebooks in numerical order for best learning experience

## 📖 Notebook Series Overview

### 🟢 Beginner Track: Foundations (Start Here!)

| Notebook                                                | Topic              | Duration | Description                                                    |
| ------------------------------------------------------- | ------------------ | -------- | -------------------------------------------------------------- |
| **[00_START_HERE.ipynb](00_START_HERE.ipynb)**       | Navigation & Setup | 10 min   | Learning path overview, prerequisites check, concept map       |
| **[01_embeddings.ipynb](01_embeddings.ipynb)**       | Text Embeddings    | 25 min   | Convert text to vectors, semantic similarity, cosine distance  |
| **[02_text_chunking.ipynb](02_text_chunking.ipynb)** | Text Chunking      | 30 min   | Document splitting strategies, chunk size vs overlap tradeoffs |
| **[03_vector_stores.ipynb](03_vector_stores.ipynb)** | Vector Databases   | 35 min   | FAISS vs Chroma comparison, similarity search, persistence     |
| **[04_llm_basics.ipynb](04_llm_basics.ipynb)**       | LLM Abstraction    | 30 min   | Factory pattern, provider switching, parameter tuning          |

**Total**: ~2.5 hours

### 🟡 Intermediate Track: Data Pipelines

| Notebook                                                        | Topic                  | Duration | Description                                                 |
| --------------------------------------------------------------- | ---------------------- | -------- | ----------------------------------------------------------- |
| **[05_pdf_ingestion.ipynb](05_pdf_ingestion.ipynb)**         | PDF Loading            | 25 min   | Extract text from PDFs, metadata handling, batch processing |
| **[06_web_ingestion.ipynb](06_web_ingestion.ipynb)**         | Web Scraping           | 30 min   | Scrape web pages, clean HTML, extract main content          |
| **[07_youtube_ingestion.ipynb](07_youtube_ingestion.ipynb)** | YouTube Transcripts    | 25 min   | Extract video transcripts, captions, metadata               |
| **[08_document_indexing.ipynb](08_document_indexing.ipynb)** | Full Indexing Pipeline | 40 min   | End-to-end: Load → Chunk → Embed → Store                 |

**Total**: ~2 hours

### 🔴 Advanced Track: RAG & Agents

| Notebook                                                              | Topic                 | Duration | Description                                                |
| --------------------------------------------------------------------- | --------------------- | -------- | ---------------------------------------------------------- |
| **[09_retrieval_strategies.ipynb](09_retrieval_strategies.ipynb)** | Retrieval Techniques  | 35 min   | Similarity search, MMR, score thresholds, hybrid retrieval |
| **[10_rag_chain.ipynb](10_rag_chain.ipynb)**                       | Complete RAG Pipeline | 45 min   | Query → Retrieve → Generate with source attribution      |
| **[11_conversational_rag.ipynb](11_conversational_rag.ipynb)**     | Conversational AI     | 40 min   | Multi-turn conversations, memory, context management       |
| **[12_rag_evaluation.ipynb](12_rag_evaluation.ipynb)**             | RAG Metrics           | 40 min   | Precision@K, Recall@K, MRR, quality measurement            |
| **[13_agent_tools.ipynb](13_agent_tools.ipynb)**                   | Agentic Workflows     | 50 min   | ReAct pattern, tool creation, agent routing                |

**Total**: ~3.5 hours

**🎯 Grand Total: ~8 hours of hands-on learning**

## 🗺️ Learning Path Map

```
START HERE
    │
    ├─> 01_embeddings ──┬─> 03_vector_stores ──┬─> 08_document_indexing ──┬─> 09_retrieval_strategies
    │                   │                       │                          │
    │                   │                       │                          └─> 10_rag_chain ───┬─> 11_conversational_rag
    │                   │                       │                                               │
    └─> 02_text_chunking┴───────────────────────┘                                              ├─> 12_rag_evaluation
                                                                                                │
    04_llm_basics ──────────────────────────────────────────────────────────────────────────────┘
                                                                                                │
    05_pdf_ingestion ──┬                                                                       │
    06_web_ingestion ──┼─> 08_document_indexing                                               │
    07_youtube_ingestion┘                                                                      │
                                                                                                │
                                                                                                └─> 13_agent_tools
```

## 🎯 Prerequisites

### Required

- ✅ Python 3.11+ installed
- ✅ All dependencies: `pip install -r ../requirements.txt`
- ✅ `.env` file configured (copy from `.env.example`)
- ✅ API keys set up (GROQ_API_KEY for LLM notebooks)

### Recommended Knowledge

- Basic Python programming
- Understanding of functions, classes, and imports
- Familiarity with Jupyter notebooks

### Optional (Helpful)

- Machine learning concepts
- Vector mathematics (dot products, cosine similarity)
- Web APIs and HTTP requests

## ⚙️ Setup Instructions

### 1. Install Dependencies

```bash
cd ../  # Go to project root
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file in project root:

```bash
# Required for LLM notebooks (04, 10, 11, 13)
GROQ_API_KEY=your_api_key_here

# Optional - defaults work fine
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_TYPE=faiss  # or chroma
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

Get a free Groq API key: https://console.groq.com/

### 3. Launch Jupyter

```bash
# From project root
jupyter notebook notebooks/

# Or from notebooks folder
cd notebooks/
jupyter notebook
```

### 4. Start Learning

Open `00_START_HERE.ipynb` and run the setup check cell!

## 📂 Additional Resources

### In This Repository

- **`/docs`** - Detailed technical documentation and guides
- **`/docs/quick-references`** - Quick reference cards for each component
- **`/app`** - Source code for all components (browse while learning!)
- **`/tests`** - Unit tests showing usage examples
- **`/demos`** - Standalone demo scripts

### Helpful Documentation Files

- **Architecture**: `../docs/ARCHITECTURE_DIAGRAM.md`
- **Embeddings**: `../docs/EMBEDDING_GUIDE.md`
- **LLM Guide**: `../docs/LLM_ABSTRACTION_GUIDE.md`
- **RAG Chain**: `../docs/CHAIN_QUICK_REF.md`
- **Full Index**: `../docs/INDEX.md`

## 🎓 Learning Tips

1. **Execute Every Cell** - Don't just read! Run the code and see results
2. **Experiment Freely** - Modify parameters and observe what changes
3. **Take Notes** - Add markdown cells with your observations
4. **Follow the Order** - Notebooks build on previous concepts
5. **Reference Code** - Look at `app/` source code for implementation details
6. **Check Tests** - See `tests/` for more usage examples
7. **Ask Questions** - Add comments and revisit unclear sections

## 🔧 Troubleshooting

### Common Issues

**ImportError: No module named 'app'**

```python
# Run this in the first cell of any notebook
import sys, os
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

**Missing API Key**

- Ensure `.env` file exists in project root
- Check `GROQ_API_KEY` is set
- Restart Jupyter kernel after adding keys

**Model Download Slow**

- First-time embedding model download takes 1-2 minutes
- Model is cached after first use (~80MB)

**Vector Store Errors**

- Clear old Chroma data: Delete `../data/vector_store/chroma/`
- FAISS errors: Reinstall with `pip install --upgrade faiss-cpu`

### Getting Help

1. Check notebook comments and documentation strings
2. Review `../docs/` folder for detailed guides
3. Look at test files in `../tests/` for examples
4. Check GitHub issues (if applicable)

## 🎯 What You'll Master

By completing this series, you'll understand:

✅ **Embeddings** - How text becomes searchable vectors
✅ **Text Chunking** - Optimal document splitting strategies
✅ **Vector Stores** - Efficient similarity search at scale
✅ **LLM Integration** - Provider-agnostic LLM usage
✅ **Data Ingestion** - Load PDFs, web pages, YouTube transcripts
✅ **RAG Pipelines** - Build complete retrieval-augmented systems
✅ **Conversational AI** - Multi-turn chat with memory
✅ **Evaluation** - Measure and improve RAG quality
✅ **Agentic AI** - Build reasoning agents with tools

## 📊 Progress Tracking

Create a checklist as you go:

- [ ] 00 - START_HERE ✓ Setup verified
- [ ] 01 - Embeddings ✓ Understand vectors
- [ ] 02 - Text Chunking ✓ Split documents
- [ ] 03 - Vector Stores ✓ Search semantically
- [ ] 04 - LLM Basics ✓ Use language models
- [ ] 05 - PDF Ingestion ✓ Load PDFs
- [ ] 06 - Web Ingestion ✓ Scrape web
- [ ] 07 - YouTube Ingestion ✓ Extract transcripts
- [ ] 08 - Document Indexing ✓ Full pipeline
- [ ] 09 - Retrieval Strategies ✓ Advanced search
- [ ] 10 - RAG Chain ✓ Complete RAG
- [ ] 11 - Conversational RAG ✓ Multi-turn chat (Coming soon)
- [ ] 12 - RAG Evaluation ✓ Measure quality (Coming soon)
- [ ] 13 - Agent Tools ✓ Reasoning agents (Coming soon)

## 🌟 After Completion

Once you've mastered the notebooks:

1. **Build Your Own RAG System** - Apply concepts to your domain
2. **Experiment with Models** - Try different LLMs and embeddings
3. **Optimize Performance** - Tune parameters for your use case
4. **Deploy to Production** - Use Streamlit UI (`app/ui/chat_app.py`)
5. **Contribute Back** - Improve notebooks or documentation

## 📝 Notebook Status

| Status                  | Notebooks                                    |
| ----------------------- | -------------------------------------------- |
| ✅**Complete**    | 00-10 (All core foundations and RAG)         |
| 📋**Planned**     | 11-13 (Conversational, Evaluation, Agents)   |

> **Note**: Notebooks 04-13 have structured templates. You can start with 00-03 immediately and work through the core concepts. Additional notebooks will follow the same pattern demonstrated in 01-03.

## 💬 Questions or Feedback?

- Add markdown cells in notebooks with questions
- Check existing documentation in `../docs/`
- Review source code in `../app/`
- Run demos in `../demos/` folder

---

**🎉 Happy Learning! Let's build amazing AI systems together!**

---
---
### ✅ Complete Notebook Series (00-13)

**Foundation (00-04):**
00_START_HERE.ipynb - Master navigation
01_embeddings.ipynb - Text-to-vector conversion
02_text_chunking.ipynb - Document splitting
03_vector_stores.ipynb - FAISS vs Chroma
04_llm_basics.ipynb - LLM factory pattern

**Data Ingestion (05-07):**
05_pdf_ingestion.ipynb - PDF loading
06_web_ingestion.ipynb - Web scraping
07_youtube_ingestion.ipynb - Video transcripts

**RAG Pipeline (08-10):**
08_document_indexing.ipynb - ✨ NEW! End-to-end Load→Chunk→Embed→Store
09_retrieval_strategies.ipynb - ✨ NEW! Similarity, MMR, thresholds, top-k tuning
10_rag_chain.ipynb - Complete RAG workflow

**Advanced (11-13): Coming next**
11_conversational_rag.ipynb - Multi-turn conversations
12_rag_evaluation.ipynb - Metrics and quality
13_agent_tools.ipynb - Agentic workflows
