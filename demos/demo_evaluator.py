"""
Demo: RAG Evaluator - Retrieval Quality Assessment

This demo showcases the RAG evaluator's capabilities for assessing
retrieval quality without requiring labeled datasets or LLMs.

Demonstrations:
1. Single query evaluation
2. Batch evaluation with aggregation
3. Configuration comparison (chunk sizes)
4. Error analysis (false positives/negatives)
5. Metric interpretation

Run:
    python demo_evaluator.py
"""

from app.rag.evaluator import (
    RAGEvaluator,
    create_relevance_mapping,
    format_evaluation_report
)


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_single_query_evaluation():
    """Demonstrate single query evaluation."""
    print_section("Demo 1: Single Query Evaluation")
    
    print("Scenario: Evaluating retrieval for 'What is Python?'\n")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(verbose=True)
    
    # Manual relevance assessment
    # Suppose we've manually reviewed our docs and determined these are relevant:
    relevant_docs = {"doc1", "doc3", "doc7", "doc12"}
    print(f"Manually identified relevant docs: {relevant_docs}\n")
    
    # Documents retrieved by our system
    retrieved_docs = ["doc3", "doc5", "doc1", "doc9", "doc2"]
    print(f"System retrieved (in rank order): {retrieved_docs}\n")
    
    # Evaluate
    result = evaluator.evaluate_query(
        query="What is Python?",
        retrieved_doc_ids=retrieved_docs,
        relevant_doc_ids=relevant_docs,
        k=5
    )
    
    # Print formatted report
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    print(format_evaluation_report(result))
    
    # Interpret results
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"\n✅ Found 2 out of 4 relevant documents (Recall@5: {result.metrics.recall_at_k:.3f})")
    print(f"✅ 2 out of 5 retrieved docs were relevant (Precision@5: {result.metrics.precision_at_k:.3f})")
    print(f"✅ First relevant doc at position 1 (MRR: {result.metrics.mrr:.3f})")
    print(f"\n❌ Missed relevant docs: {result.relevant_but_not_retrieved}")
    print(f"❌ Retrieved irrelevant docs: {result.retrieved_but_not_relevant}")


def demo_batch_evaluation():
    """Demonstrate batch evaluation."""
    print_section("Demo 2: Batch Evaluation")
    
    print("Scenario: Evaluating retrieval across multiple queries\n")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(verbose=False)  # Less verbose for batch
    
    # Test cases (simulated)
    test_cases = [
        {
            "query": "What is Python?",
            "retrieved_doc_ids": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "relevant_doc_ids": {"doc1", "doc3", "doc7"}
        },
        {
            "query": "Explain recursion",
            "retrieved_doc_ids": ["doc10", "doc11", "doc12", "doc13", "doc14"],
            "relevant_doc_ids": {"doc11", "doc12"}
        },
        {
            "query": "What is OOP?",
            "retrieved_doc_ids": ["doc20", "doc21", "doc22", "doc23", "doc24"],
            "relevant_doc_ids": {"doc20", "doc22", "doc24"}
        },
        {
            "query": "How does garbage collection work?",
            "retrieved_doc_ids": ["doc30", "doc31", "doc32", "doc33", "doc34"],
            "relevant_doc_ids": {"doc31", "doc34"}
        },
        {
            "query": "What are design patterns?",
            "retrieved_doc_ids": ["doc40", "doc41", "doc42", "doc43", "doc44"],
            "relevant_doc_ids": {"doc40", "doc41", "doc42"}
        }
    ]
    
    print(f"Evaluating {len(test_cases)} queries...\n")
    
    # Run batch evaluation
    results = evaluator.evaluate_batch(test_cases, k=5)
    
    # Print aggregate statistics
    agg = results["aggregate_metrics"]
    
    print("="*70)
    print("AGGREGATE RESULTS")
    print("="*70)
    print(f"\n📊 Precision@5:")
    print(f"   Mean: {agg['precision_at_k']['mean']:.3f}")
    print(f"   Std:  {agg['precision_at_k']['std']:.3f}")
    print(f"   Range: [{agg['precision_at_k']['min']:.3f}, {agg['precision_at_k']['max']:.3f}]")
    
    print(f"\n📊 Recall@5:")
    print(f"   Mean: {agg['recall_at_k']['mean']:.3f}")
    print(f"   Std:  {agg['recall_at_k']['std']:.3f}")
    print(f"   Range: [{agg['recall_at_k']['min']:.3f}, {agg['recall_at_k']['max']:.3f}]")
    
    print(f"\n📊 F1@5:")
    print(f"   Mean: {agg['f1_at_k']['mean']:.3f}")
    print(f"   Std:  {agg['f1_at_k']['std']:.3f}")
    print(f"   Range: [{agg['f1_at_k']['min']:.3f}, {agg['f1_at_k']['max']:.3f}]")
    
    print(f"\n📊 Hit Rate@5: {agg['hit_rate_at_k']['mean']:.3f}")
    print(f"📊 MRR: {agg['mrr']['mean']:.3f}")
    
    # Identify best and worst queries
    print("\n" + "="*70)
    print("QUERY ANALYSIS")
    print("="*70)
    print(f"\n🏆 Best performing query:")
    print(f"   Query: {results['best_query']['query']}")
    print(f"   F1@5: {results['best_query']['f1_at_k']:.3f}")
    print(f"   Precision@5: {results['best_query']['precision_at_k']:.3f}")
    print(f"   Recall@5: {results['best_query']['recall_at_k']:.3f}")
    
    print(f"\n⚠️  Worst performing query:")
    print(f"   Query: {results['worst_query']['query']}")
    print(f"   F1@5: {results['worst_query']['f1_at_k']:.3f}")
    print(f"   Precision@5: {results['worst_query']['precision_at_k']:.3f}")
    print(f"   Recall@5: {results['worst_query']['recall_at_k']:.3f}")


def demo_configuration_comparison():
    """Demonstrate configuration comparison."""
    print_section("Demo 3: Configuration Comparison")
    
    print("Scenario: Comparing chunk sizes (256 vs 512 vs 1024)\n")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(verbose=False)
    
    # Shared queries across configurations
    shared_queries = ["Query 1", "Query 2", "Query 3"]
    shared_relevant = [
        {"doc1", "doc2"},
        {"doc5", "doc6"},
        {"doc10", "doc11"}
    ]
    
    # Different retrievals for each configuration
    configurations = {
        "chunk_256": [
            {"query": shared_queries[0], "retrieved_doc_ids": ["doc1", "doc3", "doc4"], "relevant_doc_ids": shared_relevant[0]},
            {"query": shared_queries[1], "retrieved_doc_ids": ["doc5", "doc7", "doc8"], "relevant_doc_ids": shared_relevant[1]},
            {"query": shared_queries[2], "retrieved_doc_ids": ["doc10", "doc12", "doc13"], "relevant_doc_ids": shared_relevant[2]}
        ],
        "chunk_512": [
            {"query": shared_queries[0], "retrieved_doc_ids": ["doc1", "doc2", "doc3"], "relevant_doc_ids": shared_relevant[0]},
            {"query": shared_queries[1], "retrieved_doc_ids": ["doc6", "doc7", "doc8"], "relevant_doc_ids": shared_relevant[1]},
            {"query": shared_queries[2], "retrieved_doc_ids": ["doc11", "doc12", "doc13"], "relevant_doc_ids": shared_relevant[2]}
        ],
        "chunk_1024": [
            {"query": shared_queries[0], "retrieved_doc_ids": ["doc2", "doc3", "doc4"], "relevant_doc_ids": shared_relevant[0]},
            {"query": shared_queries[1], "retrieved_doc_ids": ["doc5", "doc6", "doc9"], "relevant_doc_ids": shared_relevant[1]},
            {"query": shared_queries[2], "retrieved_doc_ids": ["doc10", "doc11", "doc14"], "relevant_doc_ids": shared_relevant[2]}
        ]
    }
    
    print("Comparing 3 configurations on 3 shared queries...\n")
    
    # Compare configurations
    comparison = evaluator.compare_configurations(configurations, k=3)
    
    # Print comparison table
    print("="*70)
    print("COMPARISON TABLE")
    print("="*70)
    
    print(f"\n{'Metric':<20} {'chunk_256':<15} {'chunk_512':<15} {'chunk_1024':<15}")
    print("-" * 70)
    
    for metric_name, scores in comparison['comparison'].items():
        print(f"{metric_name:<20} ", end="")
        for config in ["chunk_256", "chunk_512", "chunk_1024"]:
            print(f"{scores[config]:<15.3f} ", end="")
        print()
    
    # Print winner and analysis
    print("\n" + "="*70)
    print("WINNER & RECOMMENDATION")
    print("="*70)
    print(f"\n🏆 Winner: {comparison['winner']}")
    print(f"   F1 Score: {comparison['analysis']['winner_score']:.3f}")
    print(f"   Improvement: +{comparison['analysis']['improvement_over_next_best']:.3f}")
    
    print(f"\n💡 Recommendation:")
    print(f"   {comparison['analysis']['recommendation']}")
    
    if comparison['analysis']['trade_offs']:
        print(f"\n⚖️  Trade-offs:")
        for trade_off in comparison['analysis']['trade_offs']:
            print(f"   - {trade_off}")


def demo_error_analysis():
    """Demonstrate error analysis."""
    print_section("Demo 4: Error Analysis")
    
    print("Scenario: Debugging poor retrieval quality\n")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(verbose=True)
    
    # Simulated poor retrieval
    relevant_docs = {"doc1", "doc2", "doc3", "doc4", "doc5"}  # 5 relevant docs
    retrieved_docs = ["doc10", "doc11", "doc2", "doc12", "doc13"]  # Only 1 relevant retrieved
    
    print(f"Expected relevant docs: {relevant_docs}")
    print(f"System retrieved: {retrieved_docs}\n")
    
    # Evaluate
    result = evaluator.evaluate_query(
        query="Test query with poor retrieval",
        retrieved_doc_ids=retrieved_docs,
        relevant_doc_ids=relevant_docs,
        k=5
    )
    
    # Diagnose issues
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    if result.metrics.recall_at_k < 0.5:
        print("\n❌ LOW RECALL DETECTED")
        print(f"   Only {result.metrics.num_relevant_retrieved}/{result.metrics.num_relevant} relevant docs retrieved")
        print(f"   Missed documents: {result.relevant_but_not_retrieved}")
        print("\n   Possible causes:")
        print("   1. Embedding model not capturing semantic meaning")
        print("   2. Query preprocessing issues")
        print("   3. Chunk size too small/large")
        print("   4. Wrong similarity metric")
    
    if result.metrics.precision_at_k < 0.5:
        print("\n❌ LOW PRECISION DETECTED")
        print(f"   Only {result.metrics.num_relevant_retrieved}/{result.metrics.k} retrieved docs are relevant")
        print(f"   Irrelevant docs retrieved: {result.retrieved_but_not_relevant}")
        print("\n   Possible causes:")
        print("   1. Too many documents retrieved (k too high)")
        print("   2. Similar but not relevant docs ranked high")
        print("   3. Noisy index with irrelevant content")
    
    if result.metrics.mrr < 0.3:
        print("\n❌ LOW MRR DETECTED")
        print(f"   First relevant doc appears late (MRR: {result.metrics.mrr:.3f})")
        print("\n   Possible causes:")
        print("   1. Ranking function not optimal")
        print("   2. Relevant docs have lower similarity scores")
        print("   3. Need re-ranking stage")


def demo_metric_interpretation():
    """Demonstrate metric interpretation."""
    print_section("Demo 5: Metric Interpretation")
    
    print("Understanding when to use which metric:\n")
    
    scenarios = [
        {
            "name": "Medical Diagnosis RAG",
            "description": "Retrieve medical literature for diagnosis support",
            "metrics": ["Recall", "MRR"],
            "reasoning": "Cannot miss relevant papers (high recall). Need important papers ranked high (MRR)."
        },
        {
            "name": "Customer Support Chatbot",
            "description": "Retrieve FAQ articles for user questions",
            "metrics": ["Precision", "Hit Rate"],
            "reasoning": "Must avoid irrelevant answers (high precision). Just need one good answer (hit rate)."
        },
        {
            "name": "Legal Document Search",
            "description": "Find relevant case law for legal research",
            "metrics": ["F1", "Recall"],
            "reasoning": "Balance finding all cases (recall) with avoiding noise (precision). F1 for overall quality."
        },
        {
            "name": "E-commerce Product Search",
            "description": "Retrieve relevant products for search queries",
            "metrics": ["Precision", "MRR"],
            "reasoning": "Users see top results (precision, MRR). Less critical to find all relevant products."
        }
    ]
    
    for scenario in scenarios:
        print(f"📋 Scenario: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Key Metrics: {', '.join(scenario['metrics'])}")
        print(f"   Why: {scenario['reasoning']}\n")
    
    print("="*70)
    print("METRIC SUMMARY")
    print("="*70)
    print("""
📊 Precision@K: Use when false positives are costly
   - Customer-facing applications
   - When users see few results
   - Quality over quantity

📊 Recall@K: Use when false negatives are costly
   - Medical/legal applications
   - Research and discovery
   - When missing info is dangerous

📊 F1@K: Use when you want balance
   - General purpose RAG
   - When precision and recall equally important
   - Comparing different systems

📊 Hit Rate@K: Use for binary success/failure
   - "Did we find anything useful?"
   - Simple success metrics
   - Quick quality checks

📊 MRR: Use when ranking matters
   - Users focus on top results
   - Order of results important
   - Search quality assessment
""")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("  RAG EVALUATOR DEMO")
    print("  Retrieval Quality Assessment Without LLMs")
    print("="*70)
    
    demo_single_query_evaluation()
    demo_batch_evaluation()
    demo_configuration_comparison()
    demo_error_analysis()
    demo_metric_interpretation()
    
    print("\n" + "="*70)
    print("  DEMO COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("✅ Evaluate retrieval without LLMs or labeled datasets")
    print("✅ Compare configurations (chunk size, k, vector stores)")
    print("✅ Diagnose retrieval problems with error analysis")
    print("✅ Choose appropriate metrics for your use case")
    print("\nNext Steps:")
    print("1. Create your own test set with manual relevance labels")
    print("2. Evaluate your RAG retriever")
    print("3. Compare different configurations")
    print("4. Iterate and improve based on metrics")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
