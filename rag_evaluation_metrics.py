"""
RAG Performance Evaluation Metrics
===================================

This script provides comprehensive evaluation metrics for RAG systems including:
- Generation quality (BLEU, ROUGE, METEOR, BERTScore)
- Retrieval quality (Precision@K, Recall@K, MRR, NDCG)
- RAG-specific metrics (Faithfulness, Answer Relevance, Context Utilization)

Author: Financial Report Summarizer Project
Date: 2024
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re


class RAGEvaluator:
    """Comprehensive evaluation suite for RAG systems"""

    def __init__(self):
        """Initialize evaluator with optional model-based metrics"""
        self.bleu_available = False
        self.rouge_available = False
        self.bertscore_available = False

        # Try importing optional dependencies
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            self.sentence_bleu = sentence_bleu
            self.smoothing = SmoothingFunction()
            self.bleu_available = True
        except ImportError:
            print("⚠️  NLTK not available - BLEU metric disabled. Install: pip install nltk")

        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.rouge_available = True
        except ImportError:
            print("⚠️  rouge-score not available - ROUGE metric disabled. Install: pip install rouge-score")

        try:
            from bert_score import score as bertscore
            self.bertscore = bertscore
            self.bertscore_available = True
        except ImportError:
            print("⚠️  BERTScore not available - BERTScore metric disabled. Install: pip install bert-score")

    # =========================================================================
    # GENERATION QUALITY METRICS
    # =========================================================================

    def compute_bleu(self, reference: str, hypothesis: str, max_n: int = 4) -> Dict[str, float]:
        """
        Compute BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)

        Args:
            reference: Ground truth answer
            hypothesis: Generated answer
            max_n: Maximum n-gram size (default: 4)

        Returns:
            Dictionary with BLEU scores for each n-gram
        """
        if not self.bleu_available:
            return {"error": "NLTK not installed"}

        # Tokenize
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        scores = {}

        # Compute BLEU for different n-grams
        for n in range(1, max_n + 1):
            weights = tuple([1.0/n] * n + [0.0] * (max_n - n))
            score = self.sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=weights,
                smoothing_function=self.smoothing.method1
            )
            scores[f'BLEU-{n}'] = round(score * 100, 2)

        # Overall BLEU-4 (standard)
        scores['BLEU'] = scores['BLEU-4']

        return scores

    def compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)

        Args:
            reference: Ground truth answer
            hypothesis: Generated answer

        Returns:
            Dictionary with ROUGE F1 scores
        """
        if not self.rouge_available:
            return {"error": "rouge-score not installed"}

        scores = self.rouge_scorer.score(reference, hypothesis)

        return {
            'ROUGE-1': round(scores['rouge1'].fmeasure * 100, 2),
            'ROUGE-2': round(scores['rouge2'].fmeasure * 100, 2),
            'ROUGE-L': round(scores['rougeL'].fmeasure * 100, 2),
        }

    def compute_bertscore(self, references: List[str], hypotheses: List[str],
                         model_type: str = 'microsoft/deberta-xlarge-mnli') -> Dict[str, float]:
        """
        Compute BERTScore for semantic similarity

        Args:
            references: List of ground truth answers
            hypotheses: List of generated answers
            model_type: Model for computing embeddings

        Returns:
            Dictionary with precision, recall, F1
        """
        if not self.bertscore_available:
            return {"error": "bert-score not installed"}

        P, R, F1 = self.bertscore(hypotheses, references, model_type=model_type, verbose=False)

        return {
            'BERTScore-P': round(P.mean().item() * 100, 2),
            'BERTScore-R': round(R.mean().item() * 100, 2),
            'BERTScore-F1': round(F1.mean().item() * 100, 2),
        }

    def compute_exact_match(self, reference: str, hypothesis: str) -> float:
        """
        Compute exact match score (1 if exactly same, 0 otherwise)

        Args:
            reference: Ground truth answer
            hypothesis: Generated answer

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        # Normalize whitespace and case
        ref_normalized = ' '.join(reference.lower().split())
        hyp_normalized = ' '.join(hypothesis.lower().split())

        return 1.0 if ref_normalized == hyp_normalized else 0.0

    def compute_f1_score(self, reference: str, hypothesis: str) -> float:
        """
        Compute token-level F1 score

        Args:
            reference: Ground truth answer
            hypothesis: Generated answer

        Returns:
            F1 score (0-100)
        """
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())

        if len(hyp_tokens) == 0:
            return 0.0

        common = ref_tokens & hyp_tokens

        if len(common) == 0:
            return 0.0

        precision = len(common) / len(hyp_tokens)
        recall = len(common) / len(ref_tokens)

        f1 = 2 * (precision * recall) / (precision + recall)

        return round(f1 * 100, 2)

    # =========================================================================
    # RETRIEVAL QUALITY METRICS
    # =========================================================================

    def compute_precision_at_k(self, retrieved_docs: List[str],
                              relevant_docs: List[str], k: int) -> float:
        """
        Compute Precision@K

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by rank)
            relevant_docs: List of truly relevant document IDs
            k: Number of top documents to consider

        Returns:
            Precision@K score (0-1)
        """
        top_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)

        num_relevant_in_top_k = len([doc for doc in top_k if doc in relevant_set])

        return num_relevant_in_top_k / k if k > 0 else 0.0

    def compute_recall_at_k(self, retrieved_docs: List[str],
                           relevant_docs: List[str], k: int) -> float:
        """
        Compute Recall@K

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by rank)
            relevant_docs: List of truly relevant document IDs
            k: Number of top documents to consider

        Returns:
            Recall@K score (0-1)
        """
        if len(relevant_docs) == 0:
            return 0.0

        top_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)

        num_relevant_in_top_k = len([doc for doc in top_k if doc in relevant_set])

        return num_relevant_in_top_k / len(relevant_docs)

    def compute_mrr(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Compute Mean Reciprocal Rank

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by rank)
            relevant_docs: List of truly relevant document IDs

        Returns:
            MRR score (0-1)
        """
        relevant_set = set(relevant_docs)

        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in relevant_set:
                return 1.0 / rank

        return 0.0

    def compute_ndcg_at_k(self, retrieved_docs: List[str],
                         relevance_scores: Dict[str, int], k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by rank)
            relevance_scores: Dictionary mapping doc_id to relevance score (0-3)
            k: Number of top documents to consider

        Returns:
            NDCG@K score (0-1)
        """
        # DCG@K
        dcg = 0.0
        for rank, doc in enumerate(retrieved_docs[:k], start=1):
            rel = relevance_scores.get(doc, 0)
            dcg += (2**rel - 1) / np.log2(rank + 1)

        # IDCG@K (ideal DCG)
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels, start=1):
            idcg += (2**rel - 1) / np.log2(rank + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    # =========================================================================
    # RAG-SPECIFIC METRICS
    # =========================================================================

    def compute_faithfulness(self, answer: str, retrieved_contexts: List[str]) -> float:
        """
        Compute faithfulness: whether answer is grounded in retrieved context

        Simple version: measures token overlap between answer and context
        Advanced version would use NLI model

        Args:
            answer: Generated answer
            retrieved_contexts: List of retrieved document chunks

        Returns:
            Faithfulness score (0-1)
        """
        answer_tokens = set(answer.lower().split())

        # Combine all contexts
        all_context = ' '.join(retrieved_contexts).lower()
        context_tokens = set(all_context.split())

        if len(answer_tokens) == 0:
            return 0.0

        # How many answer tokens appear in context?
        grounded_tokens = answer_tokens & context_tokens

        faithfulness = len(grounded_tokens) / len(answer_tokens)

        return round(faithfulness, 3)

    def compute_answer_relevance(self, question: str, answer: str,
                                embedder=None) -> float:
        """
        Compute answer relevance: how well answer addresses the question

        Args:
            question: User query
            answer: Generated answer
            embedder: Sentence embedding model (e.g., FinBERT)

        Returns:
            Relevance score (0-1) based on cosine similarity
        """
        if embedder is None:
            # Fallback: simple token overlap
            q_tokens = set(question.lower().split())
            a_tokens = set(answer.lower().split())

            if len(q_tokens) == 0 or len(a_tokens) == 0:
                return 0.0

            overlap = q_tokens & a_tokens
            return len(overlap) / len(q_tokens)

        else:
            # Use embeddings for semantic similarity
            q_embedding = embedder.encode([question], convert_to_numpy=True)[0]
            a_embedding = embedder.encode([answer], convert_to_numpy=True)[0]

            # Cosine similarity
            similarity = np.dot(q_embedding, a_embedding) / (
                np.linalg.norm(q_embedding) * np.linalg.norm(a_embedding)
            )

            return round(float(similarity), 3)

    def compute_context_utilization(self, answer: str, retrieved_contexts: List[str],
                                   cited_sources: List[str] = None) -> Dict[str, float]:
        """
        Compute how much of the retrieved context is actually used

        Args:
            answer: Generated answer
            retrieved_contexts: All chunks retrieved
            cited_sources: Sources explicitly cited in answer (optional)

        Returns:
            Dictionary with utilization metrics
        """
        num_retrieved = len(retrieved_contexts)

        if num_retrieved == 0:
            return {"utilization_rate": 0.0, "avg_overlap": 0.0}

        # Count how many contexts have significant overlap with answer
        answer_tokens = set(answer.lower().split())
        utilized_count = 0
        total_overlap = 0.0

        for context in retrieved_contexts:
            context_tokens = set(context.lower().split())
            overlap = len(answer_tokens & context_tokens)

            if overlap > 5:  # Threshold: at least 5 common tokens
                utilized_count += 1

            total_overlap += overlap / len(context_tokens) if len(context_tokens) > 0 else 0

        return {
            "utilization_rate": round(utilized_count / num_retrieved, 3),
            "avg_overlap": round(total_overlap / num_retrieved, 3)
        }

    # =========================================================================
    # COMPREHENSIVE EVALUATION
    # =========================================================================

    def evaluate_generation(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Run all generation quality metrics

        Args:
            reference: Ground truth answer
            hypothesis: Generated answer

        Returns:
            Dictionary with all metrics
        """
        results = {}

        # N-gram metrics
        if self.bleu_available:
            results.update(self.compute_bleu(reference, hypothesis))

        if self.rouge_available:
            results.update(self.compute_rouge(reference, hypothesis))

        # Exact match and F1
        results['Exact_Match'] = self.compute_exact_match(reference, hypothesis)
        results['F1_Score'] = self.compute_f1_score(reference, hypothesis)

        return results

    def evaluate_retrieval(self, retrieved_docs: List[str],
                          relevant_docs: List[str],
                          relevance_scores: Dict[str, int] = None,
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Run all retrieval quality metrics

        Args:
            retrieved_docs: Retrieved document IDs (ordered)
            relevant_docs: Ground truth relevant documents
            relevance_scores: Optional graded relevance (for NDCG)
            k_values: K values to evaluate

        Returns:
            Dictionary with all retrieval metrics
        """
        results = {}

        for k in k_values:
            results[f'Precision@{k}'] = round(
                self.compute_precision_at_k(retrieved_docs, relevant_docs, k), 3
            )
            results[f'Recall@{k}'] = round(
                self.compute_recall_at_k(retrieved_docs, relevant_docs, k), 3
            )

            if relevance_scores:
                results[f'NDCG@{k}'] = round(
                    self.compute_ndcg_at_k(retrieved_docs, relevance_scores, k), 3
                )

        results['MRR'] = round(self.compute_mrr(retrieved_docs, relevant_docs), 3)

        return results

    def evaluate_rag_end_to_end(self,
                               question: str,
                               answer: str,
                               reference: str,
                               retrieved_contexts: List[str],
                               relevant_doc_ids: List[str],
                               retrieved_doc_ids: List[str],
                               embedder=None) -> Dict[str, any]:
        """
        Comprehensive end-to-end RAG evaluation

        Args:
            question: User query
            answer: Generated answer
            reference: Ground truth answer
            retrieved_contexts: Retrieved text chunks
            relevant_doc_ids: Ground truth relevant document IDs
            retrieved_doc_ids: Actually retrieved document IDs
            embedder: Optional embedding model for semantic similarity

        Returns:
            Comprehensive evaluation report
        """
        report = {
            "generation_metrics": self.evaluate_generation(reference, answer),
            "retrieval_metrics": self.evaluate_retrieval(retrieved_doc_ids, relevant_doc_ids),
            "rag_specific": {
                "faithfulness": self.compute_faithfulness(answer, retrieved_contexts),
                "answer_relevance": self.compute_answer_relevance(question, answer, embedder),
                "context_utilization": self.compute_context_utilization(answer, retrieved_contexts)
            }
        }

        return report


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":

    print("="*70)
    print("   RAG EVALUATION METRICS - DEMONSTRATION")
    print("="*70)
    print()

    # Initialize evaluator
    evaluator = RAGEvaluator()

    # Sample data
    question = "What was Apple's revenue growth in 2023?"

    reference_answer = "Apple reported revenue of $394 billion in FY2023, representing 15% growth."

    generated_answer = "Apple Inc. achieved $394 billion in revenue for fiscal 2023, a 15% increase year-over-year."

    retrieved_contexts = [
        "Apple Inc. reported total revenue of $394 billion...",
        "The company saw 15% year-over-year revenue growth...",
        "iPhone sales contributed 52% of total revenue..."
    ]

    # Example 1: Generation Quality Metrics
    print("\n" + "="*70)
    print("   GENERATION QUALITY METRICS")
    print("="*70)

    gen_metrics = evaluator.evaluate_generation(reference_answer, generated_answer)

    for metric, value in gen_metrics.items():
        print(f"{metric:20s}: {value}")

    # Example 2: Retrieval Quality Metrics
    print("\n" + "="*70)
    print("   RETRIEVAL QUALITY METRICS")
    print("="*70)

    retrieved_doc_ids = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant_doc_ids = ['doc1', 'doc2', 'doc7']  # doc7 was not retrieved

    retrieval_metrics = evaluator.evaluate_retrieval(retrieved_doc_ids, relevant_doc_ids)

    for metric, value in retrieval_metrics.items():
        print(f"{metric:20s}: {value}")

    # Example 3: RAG-Specific Metrics
    print("\n" + "="*70)
    print("   RAG-SPECIFIC METRICS")
    print("="*70)

    faithfulness = evaluator.compute_faithfulness(generated_answer, retrieved_contexts)
    print(f"{'Faithfulness':20s}: {faithfulness}")

    relevance = evaluator.compute_answer_relevance(question, generated_answer)
    print(f"{'Answer Relevance':20s}: {relevance}")

    utilization = evaluator.compute_context_utilization(generated_answer, retrieved_contexts)
    print(f"{'Utilization Rate':20s}: {utilization['utilization_rate']}")
    print(f"{'Avg Overlap':20s}: {utilization['avg_overlap']}")

    print("\n" + "="*70)
    print("✅ Evaluation complete!")
    print("="*70)
