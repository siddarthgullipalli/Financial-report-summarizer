"""
Practical Example: Evaluating Your Financial RAG System
========================================================

This script demonstrates how to evaluate your RAG system using the metrics
from rag_evaluation_metrics.py

Usage:
1. Install dependencies: pip install -r requirements_eval.txt
2. Run: python test_rag_evaluation.py
"""

from rag_evaluation_metrics import RAGEvaluator


def create_test_dataset():
    """
    Create a test dataset with questions, ground truth answers, and relevant docs

    Returns:
        List of test cases
    """
    test_cases = [
        {
            "id": 1,
            "question": "What was Apple's revenue growth in fiscal 2023?",
            "reference_answer": "Apple Inc. reported total revenue of $394 billion in fiscal year 2023, representing approximately 15% year-over-year growth from $343 billion in fiscal 2022.",
            "relevant_doc_ids": ["Apple_2023_Item7_0", "Apple_2023_Item7_1"],
            "company": "Apple Inc."
        },
        {
            "id": 2,
            "question": "What are the main risk factors for technology companies?",
            "reference_answer": "Key risk factors include intense competition in cloud services, cybersecurity threats, economic uncertainty affecting IT spending, supply chain disruptions, and regulatory compliance challenges.",
            "relevant_doc_ids": ["Tech_Co_Item1A_0", "Tech_Co_Item1A_2"],
            "company": "Multiple"
        },
        {
            "id": 3,
            "question": "Compare gross margins across companies.",
            "reference_answer": "Company A has a gross margin of 43.5%, Company B has 42.0%, and Company C has 18.2%. Company A leads with the highest margin.",
            "relevant_doc_ids": ["CompanyA_Item7_1", "CompanyB_Item7_1", "CompanyC_Item7_1"],
            "company": "Multiple"
        }
    ]

    return test_cases


def evaluate_single_query(evaluator, test_case, rag_system_output):
    """
    Evaluate a single RAG query

    Args:
        evaluator: RAGEvaluator instance
        test_case: Test case dictionary
        rag_system_output: Output from your RAG system

    Returns:
        Evaluation results
    """
    question = test_case["question"]
    reference = test_case["reference_answer"]

    # Your RAG system returns these
    generated_answer = rag_system_output["answer"]
    retrieved_contexts = rag_system_output["contexts"]
    retrieved_doc_ids = rag_system_output["doc_ids"]

    # Ground truth
    relevant_doc_ids = test_case["relevant_doc_ids"]

    print(f"\n{'='*70}")
    print(f"   Evaluating Query {test_case['id']}: {question[:50]}...")
    print(f"{'='*70}")

    # 1. Generation Quality
    print("\n📊 GENERATION QUALITY:")
    gen_metrics = evaluator.evaluate_generation(reference, generated_answer)

    for metric, value in gen_metrics.items():
        print(f"  • {metric:20s}: {value}")

    # 2. Retrieval Quality
    print("\n🔍 RETRIEVAL QUALITY:")
    retrieval_metrics = evaluator.evaluate_retrieval(
        retrieved_doc_ids,
        relevant_doc_ids,
        k_values=[1, 3, 5]
    )

    for metric, value in retrieval_metrics.items():
        print(f"  • {metric:20s}: {value}")

    # 3. RAG-Specific Metrics
    print("\n🎯 RAG-SPECIFIC METRICS:")

    faithfulness = evaluator.compute_faithfulness(generated_answer, retrieved_contexts)
    print(f"  • {'Faithfulness':20s}: {faithfulness}")

    relevance = evaluator.compute_answer_relevance(question, generated_answer)
    print(f"  • {'Answer Relevance':20s}: {relevance}")

    utilization = evaluator.compute_context_utilization(generated_answer, retrieved_contexts)
    print(f"  • {'Context Utilization':20s}: {utilization['utilization_rate']}")

    # Combine all metrics
    all_metrics = {
        **gen_metrics,
        **retrieval_metrics,
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "context_utilization": utilization['utilization_rate']
    }

    return all_metrics


def evaluate_rag_system_batch(test_cases, rag_outputs):
    """
    Evaluate your RAG system on multiple test cases

    Args:
        test_cases: List of test cases
        rag_outputs: List of RAG system outputs

    Returns:
        Summary statistics
    """
    evaluator = RAGEvaluator()

    all_results = []

    print("\n" + "="*70)
    print("   BATCH EVALUATION - FINANCIAL RAG SYSTEM")
    print("="*70)

    for test_case, rag_output in zip(test_cases, rag_outputs):
        metrics = evaluate_single_query(evaluator, test_case, rag_output)
        all_results.append(metrics)

    # Compute average metrics
    print("\n" + "="*70)
    print("   AVERAGE METRICS ACROSS ALL QUERIES")
    print("="*70)

    avg_metrics = {}

    for key in all_results[0].keys():
        if isinstance(all_results[0][key], (int, float)):
            avg_value = sum(r[key] for r in all_results) / len(all_results)
            avg_metrics[key] = round(avg_value, 2)
            print(f"  • {key:20s}: {avg_metrics[key]}")

    return avg_metrics


def example_rag_outputs():
    """
    Example outputs from your RAG system
    In practice, you would run your actual RAG system (rag.ask(), hybrid.ask_hybrid(), etc.)
    """
    outputs = [
        {
            "answer": "Apple Inc. reported revenue of $394 billion in FY2023, representing a 15% year-over-year growth from $343 billion in FY2022.",
            "contexts": [
                "Apple Inc. total revenue: $394 billion (FY2023)",
                "Year-over-year revenue growth: 15%",
                "Previous year revenue: $343 billion"
            ],
            "doc_ids": ["Apple_2023_Item7_0", "Apple_2023_Item7_1", "Apple_2022_Item7_0"]
        },
        {
            "answer": "Main risk factors include: 1) Intense competition in cloud services, 2) Cybersecurity incidents that could harm reputation, 3) Economic uncertainty reducing IT spending.",
            "contexts": [
                "Competition in cloud services is fierce...",
                "Cybersecurity incidents pose significant risks...",
                "Economic downturn may impact customer spending..."
            ],
            "doc_ids": ["Tech_Co_Item1A_0", "Tech_Co_Item1A_2", "Tech_Co_Item1A_5"]
        },
        {
            "answer": "Company A: 43.5% gross margin, Company B: 42.0%, Company C: 18.2%. Company A has the highest margin.",
            "contexts": [
                "Company A gross margin: 43.5%",
                "Company B gross margin: 42.0%",
                "Company C gross margin: 18.2%"
            ],
            "doc_ids": ["CompanyA_Item7_1", "CompanyB_Item7_1", "CompanyC_Item7_1"]
        }
    ]

    return outputs


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("   RAG EVALUATION - PRACTICAL EXAMPLE")
    print("="*70)
    print("\nThis script demonstrates how to evaluate your RAG system.")
    print("Replace 'example_rag_outputs()' with your actual RAG system calls.\n")

    # Create test dataset
    test_cases = create_test_dataset()

    # Get outputs from your RAG system
    # In practice: rag_outputs = [rag.ask(tc["question"]) for tc in test_cases]
    rag_outputs = example_rag_outputs()

    # Evaluate
    avg_metrics = evaluate_rag_system_batch(test_cases, rag_outputs)

    print("\n" + "="*70)
    print("   EVALUATION SUMMARY")
    print("="*70)
    print(f"\n✅ Evaluated {len(test_cases)} queries")
    print(f"📊 Average BLEU Score: {avg_metrics.get('BLEU', 'N/A')}")
    print(f"📊 Average ROUGE-L: {avg_metrics.get('ROUGE-L', 'N/A')}")
    print(f"📊 Average Recall@5: {avg_metrics.get('Recall@5', 'N/A')}")
    print(f"📊 Average Faithfulness: {avg_metrics.get('faithfulness', 'N/A')}")

    print("\n" + "="*70)
    print("✅ Evaluation complete!")
    print("="*70)


# =============================================================================
# INTEGRATION WITH YOUR NOTEBOOK
# =============================================================================

def integrate_with_chromadb_rag(rag_system, hybrid_search, test_question):
    """
    Example: How to integrate evaluation with your ChromaDB RAG system

    Args:
        rag_system: Your FinBERTFinancialRAG instance
        hybrid_search: Your HybridSearch instance
        test_question: Question to evaluate
    """
    evaluator = RAGEvaluator()

    # 1. Run your RAG system
    print(f"Question: {test_question}")

    # Get answer using hybrid search
    indices = hybrid_search.hybrid_search(test_question, top_k=5)

    # Get retrieved contexts
    retrieved_contexts = [rag_system.chunks[idx] for idx in indices]
    retrieved_metadata = [rag_system.chunk_metadata[idx] for idx in indices]

    # Generate answer (simplified - use your actual generation code)
    answer = hybrid_search.ask_hybrid(test_question)

    # 2. Define ground truth (you need to create this manually)
    reference_answer = "Your ground truth answer here..."
    relevant_doc_ids = ["doc1", "doc2"]  # IDs of truly relevant documents

    # 3. Evaluate
    print("\n📊 EVALUATION RESULTS:")

    # Generation quality
    gen_metrics = evaluator.evaluate_generation(reference_answer, answer)
    print("\nGeneration Metrics:")
    for k, v in gen_metrics.items():
        print(f"  {k}: {v}")

    # RAG-specific
    faithfulness = evaluator.compute_faithfulness(answer, retrieved_contexts)
    print(f"\nFaithfulness: {faithfulness}")

    relevance = evaluator.compute_answer_relevance(
        test_question,
        answer,
        embedder=rag_system.embedder  # Use your FinBERT embedder
    )
    print(f"Answer Relevance: {relevance}")

    return gen_metrics
