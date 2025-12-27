# RAG Evaluator - Quick Reference

## Overview
Lightweight, manual-evaluation-friendly RAG retrieval evaluator focused on retrieval quality assessment without requiring labeled datasets or LLMs.

## File Structure
```
app/rag/evaluator.py (1115 lines)
├── Data Classes (RetrievalMetrics, QueryEvaluationResult)
├── RAGEvaluator Class (Core evaluator)
├── Helper Functions
└── TODO Comments (Future enhancements)
```

## Core Philosophy

**Focus: RETRIEVAL Quality, Not Generation Quality**
- Evaluate which documents were retrieved
- Compare against manually defined relevant documents
- No LLM calls required for scoring
- No model training needed
- Beginner-friendly and interview-safe

## Key Metrics

### 1. Precision@K
**What fraction of retrieved docs are relevant?**
```python
Precision@K = (# relevant docs in top-k) / k
```
- High precision = few false positives (irrelevant docs)
- Range: 0.0 to 1.0

### 2. Recall@K
**What fraction of relevant docs were retrieved?**
```python
Recall@K = (# relevant docs in top-k) / (total # relevant docs)
```
- High recall = few false negatives (missed docs)
- Range: 0.0 to 1.0

### 3. F1@K
**Harmonic mean of precision and recall**
```python
F1 = 2 * (precision * recall) / (precision + recall)
```
- Balances precision and recall
- Single metric for overall performance

### 4. Hit Rate@K
**Was at least one relevant doc retrieved?**
```python
Hit Rate = 1.0 if any relevant doc in top-k, else 0.0
```
- Binary success metric
- Useful for "did we find anything useful?" questions

### 5. MRR (Mean Reciprocal Rank)
**How early does the first relevant doc appear?**
```python
MRR = 1 / (rank of first relevant document)
```
- Rewards early relevant results
- Range: 0.0 to 1.0

## Basic Usage

### Single Query Evaluation
```python
from app.rag.evaluator import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator(verbose=True)

# Define relevance (manual assessment)
relevant_docs = {"doc1", "doc3", "doc7"}  # These are relevant

# Documents retrieved by your system
retrieved_docs = ["doc3", "doc5", "doc1", "doc9", "doc2"]

# Evaluate
result = evaluator.evaluate_query(
    query="What is Python?",
    retrieved_doc_ids=retrieved_docs,
    relevant_doc_ids=relevant_docs,
    k=5
)

# Access metrics
print(result.metrics)
print(f"Precision@5: {result.metrics.precision_at_k:.3f}")
print(f"Recall@5: {result.metrics.recall_at_k:.3f}")
print(f"F1@5: {result.metrics.f1_at_k:.3f}")

# Analyze errors
print(f"False Positives: {result.retrieved_but_not_relevant}")
print(f"False Negatives: {result.relevant_but_not_retrieved}")
```

### Batch Evaluation
```python
# Evaluate multiple queries
test_cases = [
    {
        "query": "What is Python?",
        "retrieved_doc_ids": ["doc1", "doc2", "doc3"],
        "relevant_doc_ids": {"doc1", "doc3"}
    },
    {
        "query": "Explain recursion",
        "retrieved_doc_ids": ["doc5", "doc6", "doc7"],
        "relevant_doc_ids": {"doc6"}
    },
    {
        "query": "What is OOP?",
        "retrieved_doc_ids": ["doc8", "doc9", "doc10"],
        "relevant_doc_ids": {"doc8", "doc10"}
    }
]

# Run batch evaluation
results = evaluator.evaluate_batch(test_cases, k=3)

# Access aggregate metrics
agg = results["aggregate_metrics"]
print(f"Average Precision@3: {agg['precision_at_k']['mean']:.3f}")
print(f"Average Recall@3: {agg['recall_at_k']['mean']:.3f}")
print(f"Average F1@3: {agg['f1_at_k']['mean']:.3f}")

# Identify problematic queries
print(f"\nBest query: {results['best_query']['query']}")
print(f"  F1: {results['best_query']['f1_at_k']:.3f}")

print(f"\nWorst query: {results['worst_query']['query']}")
print(f"  F1: {results['worst_query']['f1_at_k']:.3f}")
```

### Configuration Comparison
**Compare chunk sizes, k values, vector stores, etc.**

```python
# Same queries, different retrieval configurations
configurations = {
    "chunk_256": [
        {"query": "Q1", "retrieved_doc_ids": ["d1", "d2", "d3"], "relevant_doc_ids": {"d1"}},
        {"query": "Q2", "retrieved_doc_ids": ["d4", "d5", "d6"], "relevant_doc_ids": {"d5"}}
    ],
    "chunk_512": [
        {"query": "Q1", "retrieved_doc_ids": ["d1", "d7", "d8"], "relevant_doc_ids": {"d1"}},
        {"query": "Q2", "retrieved_doc_ids": ["d5", "d9", "d10"], "relevant_doc_ids": {"d5"}}
    ],
    "chunk_1024": [
        {"query": "Q1", "retrieved_doc_ids": ["d2", "d7", "d1"], "relevant_doc_ids": {"d1"}},
        {"query": "Q2", "retrieved_doc_ids": ["d5", "d11", "d12"], "relevant_doc_ids": {"d5"}}
    ]
}

# Compare configurations
comparison = evaluator.compare_configurations(configurations, k=3)

# See winner
print(f"🏆 Best configuration: {comparison['winner']}")
print(f"   F1 Score: {comparison['analysis']['winner_score']:.3f}")

# See comparison table
for metric, scores in comparison['comparison'].items():
    print(f"\n{metric}:")
    for config, score in scores.items():
        print(f"  {config}: {score:.3f}")

# Get recommendation
print(f"\n{comparison['analysis']['recommendation']}")
```

## Use Cases

### 1. Compare Chunk Sizes
```python
# Evaluate: Does chunk size 256, 512, or 1024 work best?
configs = {
    "chunk_256": test_queries_with_chunk_256_retrieval,
    "chunk_512": test_queries_with_chunk_512_retrieval,
    "chunk_1024": test_queries_with_chunk_1024_retrieval
}
comparison = evaluator.compare_configurations(configs, k=5)
print(f"Best chunk size: {comparison['winner']}")
```

### 2. Compare Top-K Values
```python
# Evaluate: Should I retrieve 3, 5, or 10 documents?
configs = {
    "k=3": test_queries_with_k3,
    "k=5": test_queries_with_k5,
    "k=10": test_queries_with_k10
}
comparison = evaluator.compare_configurations(configs)
```

### 3. Compare Vector Stores
```python
# Evaluate: FAISS vs Chroma vs Pinecone
configs = {
    "FAISS": test_queries_faiss_results,
    "Chroma": test_queries_chroma_results,
    "Pinecone": test_queries_pinecone_results
}
comparison = evaluator.compare_configurations(configs, k=5)
```

### 4. Debug Poor RAG Responses
```python
# Why is my RAG giving bad answers?
result = evaluator.evaluate_query(
    query="problematic query",
    retrieved_doc_ids=retrieved,
    relevant_doc_ids=relevant,
    k=5
)

# Check retrieval quality
if result.metrics.recall_at_k < 0.5:
    print("❌ Problem: Low recall - relevant docs not being retrieved")
    print(f"Missed documents: {result.relevant_but_not_retrieved}")

if result.metrics.precision_at_k < 0.5:
    print("❌ Problem: Low precision - too many irrelevant docs")
    print(f"Irrelevant docs retrieved: {result.retrieved_but_not_relevant}")

if result.metrics.mrr < 0.3:
    print("❌ Problem: Relevant docs ranked too low")
```

## Data Classes

### RetrievalMetrics
```python
@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    hit_rate_at_k: float
    mrr: float
    k: int
    num_retrieved: int
    num_relevant: int
    num_relevant_retrieved: int
```

### QueryEvaluationResult
```python
@dataclass
class QueryEvaluationResult:
    query: str
    retrieved_doc_ids: List[str]
    relevant_doc_ids: Set[str]
    metrics: RetrievalMetrics
    retrieved_but_not_relevant: List[str]  # False positives
    relevant_but_not_retrieved: List[str]  # False negatives
    metadata: Dict[str, Any]
```

## Helper Functions

### create_relevance_mapping()
Convert parallel lists to relevance set:
```python
from app.rag.evaluator import create_relevance_mapping

doc_ids = ["doc1", "doc2", "doc3", "doc4"]
labels = [True, False, True, False]  # Boolean relevance

relevant = create_relevance_mapping(doc_ids, labels)
print(relevant)  # {"doc1", "doc3"}
```

### format_evaluation_report()
Generate human-readable report:
```python
from app.rag.evaluator import format_evaluation_report

result = evaluator.evaluate_query(...)
report = format_evaluation_report(result)
print(report)
```

## Edge Cases Handled

✅ **Empty retrieved documents**
```python
retrieved = []
result = evaluator.evaluate_query(query, retrieved, relevant, k=5)
# Returns 0.0 for all metrics
```

✅ **Empty relevant documents**
```python
relevant = set()
result = evaluator.evaluate_query(query, retrieved, relevant, k=5)
# precision=0.0, recall=1.0 (nothing to find)
```

✅ **k larger than retrieved**
```python
retrieved = ["doc1", "doc2"]  # Only 2 docs
result = evaluator.evaluate_query(query, retrieved, relevant, k=10)
# Uses k=2 (actual number retrieved)
```

✅ **No relevant docs retrieved**
```python
retrieved = ["doc1", "doc2", "doc3"]
relevant = {"doc7", "doc8"}
result = evaluator.evaluate_query(query, retrieved, relevant, k=3)
# precision=0.0, recall=0.0, mrr=0.0, hit_rate=0.0
```

## Interview Tips

### Data Science Mindset
- **Explain metrics clearly**: "Precision is about quality, recall is about completeness"
- **Discuss trade-offs**: "Higher k increases recall but decreases precision"
- **Justify choices**: "F1 balances precision and recall, MRR prioritizes ranking"

### Practical Application
- **Show configuration comparison**: "I compared chunk sizes to optimize retrieval"
- **Discuss evaluation strategy**: "Manual relevance assessment for test set"
- **Explain debugging**: "Low recall indicated embedding model issue"

### Code Quality
- **Type hints**: All functions have clear type annotations
- **Docstrings**: Complete with examples
- **Edge cases**: Handled gracefully with validation

## Future Enhancements (TODO)

### 1. LLM-Based Relevance Assessment
```python
# TODO: Automated relevance labeling
def assess_relevance_with_llm(query: str, document: str, llm) -> float:
    """Use LLM to score document relevance without manual labels."""
    pass
```

### 2. RAGAS Integration
```python
# TODO: Comprehensive RAG evaluation
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_relevancy

def evaluate_with_ragas(...):
    """Evaluate answer quality, not just retrieval."""
    pass
```

### 3. Answer Faithfulness
```python
# TODO: Check if LLM answer is grounded in context
def evaluate_faithfulness(answer: str, context: str, llm) -> float:
    """Verify answer claims are supported by retrieved context."""
    pass
```

### 4. Diversity Metrics
```python
# TODO: Measure document diversity
def calculate_diversity(retrieved_docs: List[str]) -> float:
    """Assess diversity to avoid redundant retrievals."""
    pass
```

## Best Practices

### ✅ Manual Relevance Assessment
- Start with small test set (10-20 queries)
- Manually label relevant documents
- Use domain expertise for labeling
- Document labeling criteria

### ✅ Test Set Construction
- Cover different query types
- Include edge cases
- Represent real user queries
- Balance difficulty levels

### ✅ Metric Selection
- **Precision**: When false positives are costly
- **Recall**: When missing relevant docs is costly
- **F1**: When you want balance
- **MRR**: When ranking matters
- **Hit Rate**: For binary success/failure

### ✅ Configuration Comparison
- Keep queries constant across configs
- Test multiple k values
- Document configuration details
- Use statistical significance testing

## Summary

**What's Ready:**
✅ Complete RAG evaluator (1115 lines)
✅ 5 core metrics (P@K, R@K, F1@K, Hit Rate, MRR)
✅ Single query evaluation
✅ Batch evaluation with aggregation
✅ Configuration comparison
✅ Helper functions
✅ Edge case handling
✅ Comprehensive docstrings
✅ Type hints throughout
✅ Interview-ready code

**No Dependencies:**
❌ No LLM calls for scoring
❌ No model training
❌ No labeled datasets required
❌ No heavy ML libraries

**Focus:**
🎯 Retrieval quality assessment
🎯 Manual evaluation friendly
🎯 Configuration comparison
🎯 Beginner-friendly
🎯 Data science best practices
