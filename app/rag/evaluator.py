"""
RAG Evaluator - Retrieval Quality Assessment

Provides manual-evaluation tools for assessing RAG retrieval quality.

Core Metrics:
- Precision@K: Fraction of retrieved docs that are relevant
- Recall@K: Fraction of relevant docs that were retrieved
- F1@K: Harmonic mean of precision and recall
- MRR: Rank position of first relevant document
- Hit Rate@K: Whether at least one relevant doc was found

Use Cases:
- Compare chunk sizes, top-k values, vector stores
- Evaluate retrieval effectiveness with manual relevance judgments
- Identify retrieval failures and optimization opportunities
"""
       │
       ▼
┌──────────────┐      ┌─────────────────┐
│  Retriever   │─────▶│ Retrieved Docs  │
└──────────────┘      └────────┬────────┘
                               │
                               ▼
┌─────────────────┐    ┌──────────────┐
│  Relevant Docs  │───▶│  Evaluator   │
│  (Manual List)  │    └──────┬───────┘
└─────────────────┘           │
                              ▼
                       ┌──────────────┐
                       │   Metrics    │
                       │ (P@K, R@K)   │
                       └──────────────┘

Example Usage:
    >>> from app.rag.evaluator import RAGEvaluator
    >>> 
    >>> # Define what's relevant (manual)
    >>> relevant_doc_ids = ["doc1", "doc3", "doc7"]
    >>> 
    >>> # Retrieve documents
    >>> retrieved = retriever.retrieve("What is Python?", k=5)
    >>> 
    >>> # Evaluate
    >>> evaluator = RAGEvaluator()
    >>> result = evaluator.evaluate_query(
    ...     query="What is Python?",
    ...     retrieved_docs=retrieved,
    ...     relevant_doc_ids=relevant_doc_ids,
    ...     k=5
    ... )
    >>> 
    >>> print(f"Precision@5: {result['precision_at_k']:.2f}")
    >>> print(f"Recall@5: {result['recall_at_k']:.2f}")
"""

from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RetrievalMetrics:
    """
    Container for retrieval evaluation metrics.
    
    Attributes:
        precision_at_k: Fraction of retrieved docs that are relevant
        recall_at_k: Fraction of relevant docs that were retrieved
        f1_at_k: Harmonic mean of precision and recall
        hit_rate_at_k: Whether at least one relevant doc was retrieved
        mrr: Mean Reciprocal Rank (position of first relevant doc)
        k: Number of documents retrieved
        num_retrieved: Actual number of documents retrieved
        num_relevant: Total number of relevant documents
        num_relevant_retrieved: Number of relevant docs in retrieved set
    
    Example:
        >>> metrics = RetrievalMetrics(
        ...     precision_at_k=0.6,  # 3 out of 5 retrieved are relevant
        ...     recall_at_k=0.75,    # 3 out of 4 relevant docs retrieved
        ...     f1_at_k=0.67,
        ...     hit_rate_at_k=1.0,   # At least one relevant doc found
        ...     mrr=0.5,             # First relevant at position 2
        ...     k=5,
        ...     num_retrieved=5,
        ...     num_relevant=4,
        ...     num_relevant_retrieved=3
        ... )
    """
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    hit_rate_at_k: float
    mrr: float
    k: int
    num_retrieved: int
    num_relevant: int
    num_relevant_retrieved: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "f1_at_k": self.f1_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "mrr": self.mrr,
            "k": self.k,
            "num_retrieved": self.num_retrieved,
            "num_relevant": self.num_relevant,
            "num_relevant_retrieved": self.num_relevant_retrieved
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Retrieval Metrics (k={self.k}):\n"
            f"  Precision@{self.k}: {self.precision_at_k:.3f}\n"
            f"  Recall@{self.k}: {self.recall_at_k:.3f}\n"
            f"  F1@{self.k}: {self.f1_at_k:.3f}\n"
            f"  Hit Rate@{self.k}: {self.hit_rate_at_k:.3f}\n"
            f"  MRR: {self.mrr:.3f}\n"
            f"  Retrieved: {self.num_relevant_retrieved}/{self.num_retrieved} relevant"
        )


@dataclass
class QueryEvaluationResult:
    """
    Complete evaluation result for a single query.
    
    Attributes:
        query: The query text
        retrieved_doc_ids: IDs of retrieved documents (in order)
        relevant_doc_ids: IDs of documents marked as relevant
        metrics: RetrievalMetrics object
        retrieved_but_not_relevant: Docs retrieved but not relevant
        relevant_but_not_retrieved: Relevant docs that weren't retrieved
        metadata: Additional information
    """
    query: str
    retrieved_doc_ids: List[str]
    relevant_doc_ids: Set[str]
    metrics: RetrievalMetrics
    retrieved_but_not_relevant: List[str] = field(default_factory=list)
    relevant_but_not_retrieved: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "relevant_doc_ids": list(self.relevant_doc_ids),
            "metrics": self.metrics.to_dict(),
            "retrieved_but_not_relevant": self.retrieved_but_not_relevant,
            "relevant_but_not_retrieved": self.relevant_but_not_retrieved,
            "metadata": self.metadata
        }


# =============================================================================
# RAG Evaluator Class
# =============================================================================

class RAGEvaluator:
    """
    Lightweight evaluator for RAG retrieval quality.
    
    This class focuses on retrieval effectiveness without requiring:
    - Labeled datasets
    - Model training
    - LLM calls for scoring
    - Heavy ML dependencies
    
    It supports manual evaluation where you specify which documents
    are relevant for a query, then measures how well the retriever
    performs.
    
    Key Methods:
    - precision_at_k(): Fraction of retrieved docs that are relevant
    - recall_at_k(): Fraction of relevant docs that were retrieved
    - evaluate_query(): Complete evaluation for one query
    - evaluate_batch(): Evaluate multiple queries at once
    - compare_configurations(): Compare different retrieval settings
    
    Attributes:
        verbose: Whether to print detailed information
    
    Example:
        >>> evaluator = RAGEvaluator(verbose=True)
        >>> 
        >>> # Manual relevance assessment
        >>> relevant = {"doc1", "doc3", "doc7"}
        >>> retrieved = ["doc3", "doc5", "doc1", "doc9", "doc2"]
        >>> 
        >>> # Evaluate
        >>> result = evaluator.evaluate_query(
        ...     query="What is Python?",
        ...     retrieved_doc_ids=retrieved,
        ...     relevant_doc_ids=relevant,
        ...     k=5
        ... )
        >>> print(result.metrics)
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize RAG evaluator.
        
        Args:
            verbose: If True, prints detailed evaluation information
        """
        self.verbose = verbose
    
    def precision_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K: fraction of retrieved docs that are relevant.
        
        Precision@K = (# relevant docs in top-k) / k
        
        This answers: "Of the K documents I retrieved, how many are actually relevant?"
        
        High precision = low false positive rate (few irrelevant docs retrieved)
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs (in rank order)
            relevant_doc_ids: Set of document IDs marked as relevant
            k: Number of top documents to consider
        
        Returns:
            Precision score between 0.0 and 1.0
        
        Example:
            >>> evaluator = RAGEvaluator()
            >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
            >>> relevant = {"doc1", "doc3", "doc7"}
            >>> 
            >>> # 2 out of 5 retrieved docs are relevant
            >>> precision = evaluator.precision_at_k(retrieved, relevant, k=5)
            >>> print(f"Precision@5: {precision}")  # 0.4
        
        Edge Cases:
            - If k=0 or no docs retrieved: returns 0.0
            - If k > len(retrieved): uses actual number retrieved
            - If relevant_doc_ids is empty: returns 0.0
        """
        # Edge case: no documents or invalid k
        if k <= 0 or not retrieved_doc_ids:
            return 0.0
        
        # Edge case: no relevant documents defined
        if not relevant_doc_ids:
            return 0.0
        
        # Take only top-k documents
        top_k_retrieved = retrieved_doc_ids[:k]
        
        # Count how many are relevant
        num_relevant_in_topk = sum(
            1 for doc_id in top_k_retrieved 
            if doc_id in relevant_doc_ids
        )
        
        # Precision = relevant / retrieved
        precision = num_relevant_in_topk / len(top_k_retrieved)
        
        if self.verbose:
            print(f"Precision@{k}: {num_relevant_in_topk}/{len(top_k_retrieved)} = {precision:.3f}")
        
        return precision
    
    def recall_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K: fraction of relevant docs that were retrieved.
        
        Recall@K = (# relevant docs in top-k) / (total # relevant docs)
        
        This answers: "Of all the relevant documents, how many did I find?"
        
        High recall = low false negative rate (few relevant docs missed)
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs (in rank order)
            relevant_doc_ids: Set of document IDs marked as relevant
            k: Number of top documents to consider
        
        Returns:
            Recall score between 0.0 and 1.0
        
        Example:
            >>> evaluator = RAGEvaluator()
            >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
            >>> relevant = {"doc1", "doc3", "doc7"}
            >>> 
            >>> # 2 out of 3 relevant docs were retrieved
            >>> recall = evaluator.recall_at_k(retrieved, relevant, k=5)
            >>> print(f"Recall@5: {recall}")  # 0.667
        
        Edge Cases:
            - If no relevant docs defined: returns 1.0 (nothing to find)
            - If k=0 or no docs retrieved: returns 0.0
            - If k > len(retrieved): uses actual number retrieved
        """
        # Edge case: no relevant documents defined
        if not relevant_doc_ids:
            return 1.0  # Nothing to find = perfect recall
        
        # Edge case: no documents retrieved or invalid k
        if k <= 0 or not retrieved_doc_ids:
            return 0.0
        
        # Take only top-k documents
        top_k_retrieved = retrieved_doc_ids[:k]
        
        # Count how many relevant docs were found
        num_relevant_retrieved = sum(
            1 for doc_id in top_k_retrieved 
            if doc_id in relevant_doc_ids
        )
        
        # Recall = found / total relevant
        recall = num_relevant_retrieved / len(relevant_doc_ids)
        
        if self.verbose:
            print(f"Recall@{k}: {num_relevant_retrieved}/{len(relevant_doc_ids)} = {recall:.3f}")
        
        return recall
    
    def f1_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate F1@K: harmonic mean of precision and recall.
        
        F1 = 2 * (precision * recall) / (precision + recall)
        
        F1 balances precision and recall. Use when you want a single
        metric that considers both false positives and false negatives.
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs
            relevant_doc_ids: Set of relevant document IDs
            k: Number of top documents to consider
        
        Returns:
            F1 score between 0.0 and 1.0
        
        Example:
            >>> evaluator = RAGEvaluator()
            >>> f1 = evaluator.f1_at_k(retrieved, relevant, k=5)
        """
        precision = self.precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        recall = self.recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        
        # Edge case: both are zero
        if precision + recall == 0:
            return 0.0
        
        # Harmonic mean
        f1 = 2 * (precision * recall) / (precision + recall)
        
        if self.verbose:
            print(f"F1@{k}: {f1:.3f}")
        
        return f1
    
    def mean_reciprocal_rank(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        MRR = 1 / (rank of first relevant document)
        
        This measures how early the first relevant document appears
        in the results. Higher is better.
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs (in rank order)
            relevant_doc_ids: Set of relevant document IDs
        
        Returns:
            MRR score between 0.0 and 1.0
        
        Example:
            >>> evaluator = RAGEvaluator()
            >>> retrieved = ["doc2", "doc3", "doc1", "doc5"]
            >>> relevant = {"doc1", "doc7"}
            >>> 
            >>> # First relevant doc (doc1) is at position 3 (rank 3)
            >>> mrr = evaluator.mean_reciprocal_rank(retrieved, relevant)
            >>> print(f"MRR: {mrr}")  # 1/3 = 0.333
        
        Edge Cases:
            - If no relevant docs retrieved: returns 0.0
            - If no relevant docs defined: returns 0.0
            - Rank starts at 1 (first position)
        """
        # Edge cases
        if not relevant_doc_ids or not retrieved_doc_ids:
            return 0.0
        
        # Find position of first relevant document (1-indexed)
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant_doc_ids:
                mrr = 1.0 / rank
                if self.verbose:
                    print(f"MRR: 1/{rank} = {mrr:.3f} (first relevant at position {rank})")
                return mrr
        
        # No relevant document found
        if self.verbose:
            print(f"MRR: 0.0 (no relevant documents retrieved)")
        return 0.0
    
    def hit_rate_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate Hit Rate@K: whether at least one relevant doc was found.
        
        Hit Rate@K = 1.0 if any relevant doc in top-k, else 0.0
        
        This is a binary metric: did we find anything useful?
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs
            relevant_doc_ids: Set of relevant document IDs
            k: Number of top documents to consider
        
        Returns:
            1.0 if hit, 0.0 otherwise
        
        Example:
            >>> evaluator = RAGEvaluator()
            >>> hit_rate = evaluator.hit_rate_at_k(retrieved, relevant, k=5)
        """
        if k <= 0 or not retrieved_doc_ids or not relevant_doc_ids:
            return 0.0
        
        top_k = retrieved_doc_ids[:k]
        hit = any(doc_id in relevant_doc_ids for doc_id in top_k)
        
        hit_rate = 1.0 if hit else 0.0
        
        if self.verbose:
            print(f"Hit Rate@{k}: {hit_rate} ({'HIT' if hit else 'MISS'})")
        
        return hit_rate
    
    def evaluate_query(
        self,
        query: str,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Union[Set[str], List[str]],
        k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QueryEvaluationResult:
        """
        Perform complete evaluation for a single query.
        
        This is the main evaluation method that computes all metrics
        and provides detailed analysis.
        
        Args:
            query: The query text
            retrieved_doc_ids: List of retrieved document IDs (in rank order)
            relevant_doc_ids: Set or list of document IDs marked as relevant
            k: Number of top documents to evaluate (default: all retrieved)
            metadata: Optional additional information
        
        Returns:
            QueryEvaluationResult with all metrics and analysis
        
        Example:
            >>> evaluator = RAGEvaluator(verbose=True)
            >>> 
            >>> result = evaluator.evaluate_query(
            ...     query="What is Python?",
            ...     retrieved_doc_ids=["doc3", "doc5", "doc1", "doc9", "doc2"],
            ...     relevant_doc_ids={"doc1", "doc3", "doc7"},
            ...     k=5
            ... )
            >>> 
            >>> print(result.metrics)
            >>> print(f"Missed: {result.relevant_but_not_retrieved}")
        
        Edge Cases:
            - Handles empty retrieved or relevant lists
            - Handles k larger than number of retrieved docs
            - Converts relevant_doc_ids to set if list provided
        """
        # Convert relevant to set if needed
        if isinstance(relevant_doc_ids, list):
            relevant_doc_ids = set(relevant_doc_ids)
        
        # Default k to all retrieved documents
        if k is None:
            k = len(retrieved_doc_ids)
        
        # Limit k to actual number retrieved
        k = min(k, len(retrieved_doc_ids))
        
        if self.verbose:
            print(f"\nEvaluating query: '{query}'")
            print(f"Retrieved: {len(retrieved_doc_ids)} docs")
            print(f"Relevant: {len(relevant_doc_ids)} docs")
            print(f"Evaluating top-{k}\n")
        
        # Calculate all metrics
        precision = self.precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        recall = self.recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        f1 = self.f1_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        hit_rate = self.hit_rate_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        mrr = self.mean_reciprocal_rank(retrieved_doc_ids, relevant_doc_ids)
        
        # Analyze what was retrieved vs what was relevant
        top_k_retrieved = retrieved_doc_ids[:k]
        retrieved_set = set(top_k_retrieved)
        
        num_relevant_retrieved = len(retrieved_set & relevant_doc_ids)
        retrieved_but_not_relevant = list(retrieved_set - relevant_doc_ids)
        relevant_but_not_retrieved = list(relevant_doc_ids - retrieved_set)
        
        # Create metrics object
        metrics = RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            f1_at_k=f1,
            hit_rate_at_k=hit_rate,
            mrr=mrr,
            k=k,
            num_retrieved=len(top_k_retrieved),
            num_relevant=len(relevant_doc_ids),
            num_relevant_retrieved=num_relevant_retrieved
        )
        
        # Create result object
        result = QueryEvaluationResult(
            query=query,
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            metrics=metrics,
            retrieved_but_not_relevant=retrieved_but_not_relevant,
            relevant_but_not_retrieved=relevant_but_not_retrieved,
            metadata=metadata or {}
        )
        
        if self.verbose:
            print(f"\nFalse Positives (retrieved but not relevant): {retrieved_but_not_relevant}")
            print(f"False Negatives (relevant but not retrieved): {relevant_but_not_retrieved}")
        
        return result

    
    def evaluate_batch(
        self,
        queries_and_relevance: List[Dict[str, Any]],
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple queries and compute aggregate statistics.
        
        This method is useful for:
        - Evaluating a retriever on a test set
        - Computing overall performance metrics
        - Identifying problematic queries
        
        Args:
            queries_and_relevance: List of dictionaries, each containing:
                - "query": Query text
                - "retrieved_doc_ids": List of retrieved document IDs
                - "relevant_doc_ids": Set/list of relevant document IDs
                - "metadata" (optional): Additional information
            k: Number of top documents to evaluate (default: all retrieved)
        
        Returns:
            Dictionary containing:
                - "individual_results": List of QueryEvaluationResult objects
                - "aggregate_metrics": Average metrics across all queries
                - "best_query": Query with highest F1
                - "worst_query": Query with lowest F1
                - "num_queries": Total number of queries evaluated
        
        Example:
            >>> evaluator = RAGEvaluator()
            >>> 
            >>> test_cases = [
            ...     {
            ...         "query": "What is Python?",
            ...         "retrieved_doc_ids": ["doc1", "doc2", "doc3"],
            ...         "relevant_doc_ids": {"doc1", "doc3"}
            ...     },
            ...     {
            ...         "query": "Explain recursion",
            ...         "retrieved_doc_ids": ["doc5", "doc6", "doc7"],
            ...         "relevant_doc_ids": {"doc6"}
            ...     }
            ... ]
            >>> 
            >>> results = evaluator.evaluate_batch(test_cases, k=3)
            >>> print(f"Average Precision@3: {results['aggregate_metrics']['precision_at_k']:.3f}")
        """
        if not queries_and_relevance:
            raise ValueError("queries_and_relevance cannot be empty")
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Batch Evaluation: {len(queries_and_relevance)} queries")
            print(f"{'='*70}")
        
        # Evaluate each query
        individual_results = []
        for idx, item in enumerate(queries_and_relevance, 1):
            if self.verbose:
                print(f"\n[{idx}/{len(queries_and_relevance)}] Evaluating...")
            
            result = self.evaluate_query(
                query=item["query"],
                retrieved_doc_ids=item["retrieved_doc_ids"],
                relevant_doc_ids=item["relevant_doc_ids"],
                k=k,
                metadata=item.get("metadata", {})
            )
            individual_results.append(result)
        
        # Compute aggregate statistics
        metrics_lists = defaultdict(list)
        for result in individual_results:
            metrics_lists["precision_at_k"].append(result.metrics.precision_at_k)
            metrics_lists["recall_at_k"].append(result.metrics.recall_at_k)
            metrics_lists["f1_at_k"].append(result.metrics.f1_at_k)
            metrics_lists["hit_rate_at_k"].append(result.metrics.hit_rate_at_k)
            metrics_lists["mrr"].append(result.metrics.mrr)
        
        aggregate_metrics = {
            "precision_at_k": {
                "mean": statistics.mean(metrics_lists["precision_at_k"]),
                "median": statistics.median(metrics_lists["precision_at_k"]),
                "std": statistics.stdev(metrics_lists["precision_at_k"]) if len(metrics_lists["precision_at_k"]) > 1 else 0.0,
                "min": min(metrics_lists["precision_at_k"]),
                "max": max(metrics_lists["precision_at_k"])
            },
            "recall_at_k": {
                "mean": statistics.mean(metrics_lists["recall_at_k"]),
                "median": statistics.median(metrics_lists["recall_at_k"]),
                "std": statistics.stdev(metrics_lists["recall_at_k"]) if len(metrics_lists["recall_at_k"]) > 1 else 0.0,
                "min": min(metrics_lists["recall_at_k"]),
                "max": max(metrics_lists["recall_at_k"])
            },
            "f1_at_k": {
                "mean": statistics.mean(metrics_lists["f1_at_k"]),
                "median": statistics.median(metrics_lists["f1_at_k"]),
                "std": statistics.stdev(metrics_lists["f1_at_k"]) if len(metrics_lists["f1_at_k"]) > 1 else 0.0,
                "min": min(metrics_lists["f1_at_k"]),
                "max": max(metrics_lists["f1_at_k"])
            },
            "hit_rate_at_k": {
                "mean": statistics.mean(metrics_lists["hit_rate_at_k"]),
                "median": statistics.median(metrics_lists["hit_rate_at_k"]),
                "std": statistics.stdev(metrics_lists["hit_rate_at_k"]) if len(metrics_lists["hit_rate_at_k"]) > 1 else 0.0,
                "min": min(metrics_lists["hit_rate_at_k"]),
                "max": max(metrics_lists["hit_rate_at_k"])
            },
            "mrr": {
                "mean": statistics.mean(metrics_lists["mrr"]),
                "median": statistics.median(metrics_lists["mrr"]),
                "std": statistics.stdev(metrics_lists["mrr"]) if len(metrics_lists["mrr"]) > 1 else 0.0,
                "min": min(metrics_lists["mrr"]),
                "max": max(metrics_lists["mrr"])
            }
        }
        
        # Find best and worst queries (by F1)
        f1_scores = [(result, result.metrics.f1_at_k) for result in individual_results]
        best_query = max(f1_scores, key=lambda x: x[1])[0]
        worst_query = min(f1_scores, key=lambda x: x[1])[0]
        
        batch_results = {
            "individual_results": individual_results,
            "aggregate_metrics": aggregate_metrics,
            "best_query": {
                "query": best_query.query,
                "f1_at_k": best_query.metrics.f1_at_k,
                "precision_at_k": best_query.metrics.precision_at_k,
                "recall_at_k": best_query.metrics.recall_at_k
            },
            "worst_query": {
                "query": worst_query.query,
                "f1_at_k": worst_query.metrics.f1_at_k,
                "precision_at_k": worst_query.metrics.precision_at_k,
                "recall_at_k": worst_query.metrics.recall_at_k
            },
            "num_queries": len(queries_and_relevance)
        }
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Aggregate Results:")
            print(f"  Precision@K: {aggregate_metrics['precision_at_k']['mean']:.3f} ± {aggregate_metrics['precision_at_k']['std']:.3f}")
            print(f"  Recall@K: {aggregate_metrics['recall_at_k']['mean']:.3f} ± {aggregate_metrics['recall_at_k']['std']:.3f}")
            print(f"  F1@K: {aggregate_metrics['f1_at_k']['mean']:.3f} ± {aggregate_metrics['f1_at_k']['std']:.3f}")
            print(f"  Hit Rate@K: {aggregate_metrics['hit_rate_at_k']['mean']:.3f}")
            print(f"  MRR: {aggregate_metrics['mrr']['mean']:.3f}")
            print(f"{'='*70}\n")
        
        return batch_results
    
    def compare_configurations(
        self,
        configurations: Dict[str, List[Dict[str, Any]]],
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare different RAG configurations on the same queries.
        
        This is the primary method for comparing:
        - Different chunk sizes (256 vs 512 vs 1024)
        - Different k values (3 vs 5 vs 10)
        - Different vector stores (FAISS vs Chroma)
        - Different embedding models
        
        Args:
            configurations: Dictionary mapping configuration names to
                lists of query evaluation data. Each configuration
                should have the same queries evaluated.
                
                Example:
                {
                    "chunk_256": [...query results...],
                    "chunk_512": [...query results...],
                    "chunk_1024": [...query results...]
                }
            k: Number of top documents to evaluate
        
        Returns:
            Dictionary containing:
                - "configurations": Individual results for each configuration
                - "comparison": Side-by-side comparison of aggregate metrics
                - "winner": Configuration with best average F1
                - "analysis": Interpretation and recommendations
        
        Example:
            >>> evaluator = RAGEvaluator(verbose=True)
            >>> 
            >>> # Same queries, different retrievals
            >>> configs = {
            ...     "chunk_256": [
            ...         {"query": "Q1", "retrieved_doc_ids": [...], "relevant_doc_ids": {...}},
            ...         {"query": "Q2", "retrieved_doc_ids": [...], "relevant_doc_ids": {...}}
            ...     ],
            ...     "chunk_512": [
            ...         {"query": "Q1", "retrieved_doc_ids": [...], "relevant_doc_ids": {...}},
            ...         {"query": "Q2", "retrieved_doc_ids": [...], "relevant_doc_ids": {...}}
            ...     ]
            ... }
            >>> 
            >>> comparison = evaluator.compare_configurations(configs, k=5)
            >>> print(f"Best configuration: {comparison['winner']}")
        """
        if not configurations:
            raise ValueError("configurations cannot be empty")
        
        # Validate: all configurations should have same number of queries
        num_queries = len(next(iter(configurations.values())))
        for config_name, queries in configurations.items():
            if len(queries) != num_queries:
                raise ValueError(
                    f"All configurations must have the same number of queries. "
                    f"{config_name} has {len(queries)}, expected {num_queries}"
                )
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Configuration Comparison")
            print(f"Configurations: {list(configurations.keys())}")
            print(f"Queries per config: {num_queries}")
            print(f"{'='*70}\n")
        
        # Evaluate each configuration
        configuration_results = {}
        for config_name, queries_data in configurations.items():
            if self.verbose:
                print(f"\nEvaluating configuration: {config_name}")
                print(f"-" * 70)
            
            batch_result = self.evaluate_batch(queries_data, k=k)
            configuration_results[config_name] = batch_result
        
        # Compare aggregate metrics
        comparison_table = {
            "precision_at_k": {},
            "recall_at_k": {},
            "f1_at_k": {},
            "hit_rate_at_k": {},
            "mrr": {}
        }
        
        for config_name, result in configuration_results.items():
            agg = result["aggregate_metrics"]
            comparison_table["precision_at_k"][config_name] = agg["precision_at_k"]["mean"]
            comparison_table["recall_at_k"][config_name] = agg["recall_at_k"]["mean"]
            comparison_table["f1_at_k"][config_name] = agg["f1_at_k"]["mean"]
            comparison_table["hit_rate_at_k"][config_name] = agg["hit_rate_at_k"]["mean"]
            comparison_table["mrr"][config_name] = agg["mrr"]["mean"]
        
        # Determine winner (by F1)
        winner_config = max(
            comparison_table["f1_at_k"].items(),
            key=lambda x: x[1]
        )[0]
        
        # Generate analysis
        analysis = self._generate_comparison_analysis(
            comparison_table,
            winner_config,
            configuration_results
        )
        
        result = {
            "configurations": configuration_results,
            "comparison": comparison_table,
            "winner": winner_config,
            "analysis": analysis
        }
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Comparison Summary:")
            print(f"\nConfiguration Rankings (by F1@K):")
            for rank, (config, f1) in enumerate(
                sorted(comparison_table["f1_at_k"].items(), key=lambda x: x[1], reverse=True),
                1
            ):
                print(f"  {rank}. {config}: {f1:.3f}")
            print(f"\n🏆 Winner: {winner_config}")
            print(f"{'='*70}\n")
        
        return result
    
    def _generate_comparison_analysis(
        self,
        comparison_table: Dict[str, Dict[str, float]],
        winner: str,
        configuration_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate analysis of configuration comparison.
        
        Args:
            comparison_table: Metrics comparison table
            winner: Name of winning configuration
            configuration_results: Full results for each configuration
        
        Returns:
            Dictionary with analysis insights
        """
        # Calculate metric differences
        f1_scores = comparison_table["f1_at_k"]
        precision_scores = comparison_table["precision_at_k"]
        recall_scores = comparison_table["recall_at_k"]
        
        winner_f1 = f1_scores[winner]
        
        # Find biggest improvement
        improvements = {}
        for config, f1 in f1_scores.items():
            if config != winner:
                improvements[config] = winner_f1 - f1
        
        if improvements:
            biggest_improvement = max(improvements.values())
        else:
            biggest_improvement = 0.0
        
        # Identify trade-offs
        trade_offs = []
        for config, precision in precision_scores.items():
            recall = recall_scores[config]
            if config != winner:
                if precision > precision_scores[winner] and recall < recall_scores[winner]:
                    trade_offs.append(f"{config} has higher precision but lower recall than {winner}")
                elif precision < precision_scores[winner] and recall > recall_scores[winner]:
                    trade_offs.append(f"{config} has lower precision but higher recall than {winner}")
        
        analysis = {
            "winner_score": winner_f1,
            "improvement_over_next_best": biggest_improvement,
            "trade_offs": trade_offs,
            "recommendation": (
                f"Use {winner} configuration for best overall performance (F1: {winner_f1:.3f}). "
                f"It achieves {biggest_improvement:.3f} improvement over the next best alternative."
            )
        }
        
        return analysis


# =============================================================================
# Helper Functions
# =============================================================================

def create_relevance_mapping(
    doc_ids: List[str],
    relevance_labels: List[bool]
) -> Set[str]:
    """
    Helper function to create relevance set from parallel lists.
    
    Args:
        doc_ids: List of document IDs
        relevance_labels: List of boolean relevance labels (True=relevant)
    
    Returns:
        Set of relevant document IDs
    
    Example:
        >>> doc_ids = ["doc1", "doc2", "doc3", "doc4"]
        >>> labels = [True, False, True, False]
        >>> relevant = create_relevance_mapping(doc_ids, labels)
        >>> print(relevant)  # {"doc1", "doc3"}
    """
    if len(doc_ids) != len(relevance_labels):
        raise ValueError("doc_ids and relevance_labels must have same length")
    
    return {doc_id for doc_id, is_relevant in zip(doc_ids, relevance_labels) if is_relevant}


def format_evaluation_report(result: QueryEvaluationResult) -> str:
    """
    Format evaluation result as human-readable report.
    
    Args:
        result: QueryEvaluationResult object
    
    Returns:
        Formatted string report
    
    Example:
        >>> report = format_evaluation_report(result)
        >>> print(report)
    """
    report = []
    report.append(f"Query: {result.query}")
    report.append(f"\n{result.metrics}")
    
    if result.retrieved_but_not_relevant:
        report.append(f"\nFalse Positives: {result.retrieved_but_not_relevant}")
    
    if result.relevant_but_not_retrieved:
        report.append(f"\nFalse Negatives: {result.relevant_but_not_retrieved}")
    
    return "\n".join(report)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "RAGEvaluator",
    "RetrievalMetrics",
    "QueryEvaluationResult",
    "create_relevance_mapping",
    "format_evaluation_report"
]
