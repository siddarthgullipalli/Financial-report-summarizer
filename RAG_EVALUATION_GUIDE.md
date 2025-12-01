# RAG Performance Evaluation Guide

## 📋 Overview

This guide explains how to evaluate your Financial Report Summarizer RAG system using industry-standard metrics.

---

## 🎯 Why Evaluate RAG Systems?

RAG systems have two components that need evaluation:
1. **Retrieval**: Did we find the right documents?
2. **Generation**: Did we produce a good answer?

---

## 📊 Metrics Explained

### **1. BLEU (Bilingual Evaluation Understudy)**

**What it measures:** How similar is the generated text to the reference answer (based on n-gram overlap)

**Formula:**
```
BLEU = BP × exp(Σ(log(p_n)))
where p_n = precision of n-grams
```

**Range:** 0-100 (higher is better)

**Example:**
- Reference: "Apple revenue was $394 billion"
- Generated: "Apple's revenue reached $394 billion"
- BLEU-1: ~75 (good word overlap)
- BLEU-4: ~40 (some 4-grams don't match)

**Best for:** Exact wording comparison

**Limitations:** Doesn't capture semantic meaning well

---

### **2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**

**What it measures:** Overlap between generated and reference text (focuses on recall)

**Types:**
- **ROUGE-1:** Unigram (single word) overlap
- **ROUGE-2:** Bigram (2-word phrase) overlap
- **ROUGE-L:** Longest Common Subsequence

**Formula (ROUGE-1):**
```
Recall = (matching words) / (words in reference)
Precision = (matching words) / (words in generated)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Range:** 0-100 (higher is better)

**Example:**
- Reference: "Apple revenue grew 15% to $394 billion"
- Generated: "Apple achieved $394 billion revenue, up 15%"
- ROUGE-1: ~80 (most words match)
- ROUGE-L: ~70 (good sequence overlap)

**Best for:** Summarization tasks (perfect for your financial summarizer!)

---

### **3. BERTScore**

**What it measures:** Semantic similarity using contextual embeddings

**How it works:**
1. Embed each word using BERT
2. Compute pairwise cosine similarity
3. Match words with highest similarity
4. Average across all words

**Range:** 0-1 (higher is better)

**Example:**
- Reference: "revenue increased"
- Generated: "sales grew"
- Traditional metrics: 0% match
- BERTScore: ~0.85 (captures synonyms!)

**Best for:** When meaning matters more than exact wording

**Advantage:** Understands "revenue" ≈ "sales", "increased" ≈ "grew"

---

### **4. Precision@K & Recall@K** (Retrieval Metrics)

**Precision@K:** What fraction of top-K retrieved docs are relevant?
```
Precision@K = (relevant docs in top-K) / K
```

**Recall@K:** What fraction of all relevant docs are in top-K?
```
Recall@K = (relevant docs in top-K) / (total relevant docs)
```

**Example:**
- Retrieved top-5: [doc1, doc2, doc3, doc4, doc5]
- Actually relevant: [doc1, doc2, doc7]
- Precision@5 = 2/5 = 0.40 (40% of retrieved are relevant)
- Recall@5 = 2/3 = 0.67 (caught 67% of relevant docs)

**Best for:** Understanding retrieval quality

---

### **5. MRR (Mean Reciprocal Rank)**

**What it measures:** How quickly you find the first relevant document

**Formula:**
```
MRR = 1 / (rank of first relevant doc)
```

**Example:**
- Retrieved: [doc3, doc7, doc1, doc2]
- Relevant: [doc1, doc2]
- First relevant doc (doc7) is at rank 2
- MRR = 1/2 = 0.5

**Best for:** When you care about finding ANY relevant doc quickly

---

### **6. NDCG (Normalized Discounted Cumulative Gain)**

**What it measures:** Quality of ranking with graded relevance

**How it works:**
1. Assign relevance scores (0=not relevant, 1=somewhat, 2=relevant, 3=highly relevant)
2. Compute DCG = Σ (2^relevance - 1) / log2(rank + 1)
3. Normalize by ideal ranking (IDCG)

**Example:**
- Retrieved: [doc_A(rel=3), doc_B(rel=1), doc_C(rel=0)]
- Ideal: [doc_A(rel=3), doc_D(rel=2), doc_B(rel=1)]
- NDCG@3 ≈ 0.85 (good but not perfect)

**Best for:** When some docs are more relevant than others

---

### **7. Faithfulness / Groundedness** (RAG-Specific)

**What it measures:** Is the answer supported by retrieved context?

**Simple version:**
```
Faithfulness = (answer tokens in context) / (total answer tokens)
```

**Advanced version:** Use NLI (Natural Language Inference) model

**Example:**
- Context: "Apple revenue: $394B"
- Answer: "Apple made $394 billion" → Faithful ✅
- Answer: "Apple made $500 billion" → Hallucination ❌

**Critical for:** Preventing hallucinations in financial data!

---

### **8. Answer Relevance** (RAG-Specific)

**What it measures:** Does the answer address the question?

**Method:**
```
Relevance = cosine_similarity(question_embedding, answer_embedding)
```

**Example:**
- Question: "What was Apple's revenue?"
- Answer: "Apple's revenue was $394B" → Relevant ✅
- Answer: "Apple is headquartered in Cupertino" → Not relevant ❌

---

### **9. Context Utilization**

**What it measures:** How much retrieved context is actually used?

**Formula:**
```
Utilization = (contexts with significant overlap) / (total retrieved contexts)
```

**Example:**
- Retrieved 5 chunks, but only 2 mentioned in answer
- Utilization = 2/5 = 0.40 (40%)

**Insight:** Low utilization → retrieving irrelevant docs

---

## 🚀 How to Use

### **Step 1: Install Dependencies**

```bash
pip install -r requirements_eval.txt
```

### **Step 2: Create Test Dataset**

You need:
- Test questions
- Ground truth answers (reference answers)
- Relevant document IDs

Example:
```python
test_case = {
    "question": "What was Apple's revenue growth?",
    "reference_answer": "Apple revenue grew 15% to $394 billion",
    "relevant_doc_ids": ["Apple_2023_Item7_0", "Apple_2023_Item7_1"]
}
```

### **Step 3: Run Your RAG System**

```python
# Your existing code
answer = hybrid.ask_hybrid("What was Apple's revenue growth?")

# Get retrieved chunks and IDs
indices = hybrid.hybrid_search(question, top_k=5)
retrieved_contexts = [rag.chunks[idx] for idx in indices]
```

### **Step 4: Evaluate**

```python
from rag_evaluation_metrics import RAGEvaluator

evaluator = RAGEvaluator()

# Generation quality
metrics = evaluator.evaluate_generation(
    reference="Apple revenue grew 15% to $394 billion",
    hypothesis=answer
)

print(metrics)
# Output: {'BLEU-1': 75.2, 'ROUGE-1': 82.3, 'ROUGE-L': 78.1, ...}
```

### **Step 5: Interpret Results**

**Good RAG System:**
- BLEU > 40
- ROUGE-L > 50
- Recall@5 > 0.8
- Faithfulness > 0.9
- Answer Relevance > 0.7

**Your Results (from Progress Report 2):**
- Retrieval Success: 37.5% → 100% ✅ (after hybrid search)
- Response Time: -31.6% faster ✅ (with re-ranking)

---

## 📖 Complete Example

```python
from rag_evaluation_metrics import RAGEvaluator

# Initialize
evaluator = RAGEvaluator()

# Test data
question = "What was Apple's revenue?"
reference = "Apple reported $394 billion revenue in FY2023"
generated = "Apple's FY2023 revenue reached $394 billion"

retrieved_contexts = [
    "Apple Inc. total revenue: $394 billion",
    "Fiscal year 2023 financial results..."
]

# Evaluate
results = evaluator.evaluate_generation(reference, generated)

print("BLEU:", results['BLEU'])          # ~85
print("ROUGE-L:", results['ROUGE-L'])    # ~90
print("F1 Score:", results['F1_Score'])  # ~88

# Check faithfulness
faith = evaluator.compute_faithfulness(generated, retrieved_contexts)
print("Faithfulness:", faith)  # ~0.95 (well-grounded!)
```

---

## 🎨 Visualization Example

```python
import matplotlib.pyplot as plt

methods = ['Basic RAG', 'Hybrid', 'Few-Shot', 'Re-Ranking']
bleu_scores = [25.3, 68.5, 72.1, 71.8]
rouge_scores = [30.2, 75.3, 78.9, 76.2]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].bar(methods, bleu_scores)
ax[0].set_title('BLEU Scores')
ax[0].set_ylabel('Score')

ax[1].bar(methods, rouge_scores, color='orange')
ax[1].set_title('ROUGE-L Scores')
ax[1].set_ylabel('Score')

plt.tight_layout()
plt.savefig('rag_evaluation.png')
```

---

## 📚 Recommended Metrics for Your Project

For your **Financial Report Summarizer**, focus on:

1. **ROUGE-L** - Best for summarization quality
2. **BERTScore** - Captures financial terminology semantics
3. **Recall@5** - Ensures you find relevant financial data
4. **Faithfulness** - Critical! Prevent hallucinated financial figures
5. **Answer Relevance** - Make sure you answer the actual question

---

## 🔗 Integration with Your Notebook

Add this cell to your `GENAI_PROJECT_CHROMADB.ipynb`:

```python
# Cell: Evaluate RAG Performance

from rag_evaluation_metrics import RAGEvaluator

evaluator = RAGEvaluator()

# Test question
question = "What are the main business activities?"
reference = "Your ground truth answer..."

# Get answer from your system
answer = hybrid.ask_hybrid(question)

# Evaluate
metrics = evaluator.evaluate_generation(reference, answer)

print("📊 Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value}")
```

---

## 📖 References

1. **BLEU:** Papineni et al. (2002) - "BLEU: a Method for Automatic Evaluation of Machine Translation"
2. **ROUGE:** Lin (2004) - "ROUGE: A Package for Automatic Evaluation of Summaries"
3. **BERTScore:** Zhang et al. (2020) - "BERTScore: Evaluating Text Generation with BERT"
4. **RAG Evaluation:** Lewis et al. (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

---

## ✅ Quick Command Reference

```bash
# Install
pip install -r requirements_eval.txt

# Run test
python test_rag_evaluation.py

# Import in notebook
from rag_evaluation_metrics import RAGEvaluator
evaluator = RAGEvaluator()
```

---

**Questions?** Check `test_rag_evaluation.py` for working examples!
