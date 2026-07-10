# Evaluation Datasets and Metrics

This document details the datasets and metrics used to evaluate the MedKG-RAG Chatbot's performance across two phases: Retrieval and Generation.

## 1. Datasets Used

The project utilizes high-quality biomedical datasets for evaluation:

### 1.1. BioASQ Task B (Primary Dataset)
BioASQ Task B is the world's leading benchmark for Biomedical Semantic Indexing and Question Answering. Questions and answers are created and curated by biomedical experts.
- **Validation Split**: 500 questions. Used for hyperparameter tuning.
- **Test Split**: 500 questions. Used for final system performance evaluation.

### 1.2. MedAESQA (Secondary/External Benchmark)
This dataset consists of complex clinical questions, used to evaluate factual accuracy and grounding capabilities to mitigate LLM hallucination.
- **Size**: ~42 questions.
- *Note*: Used exclusively as an independent, external test set for the Generation phase. It is strictly not used for Retrieval evaluation or parameter tuning to test zero-shot robustness.

### 1.3. PubMedQA (Optional)
Optionally used to evaluate generation quality based on oracle context provided by abstracts, bypassing the retrieval system.

---

## 2. Core Evaluation Metrics

The RAG system is divided into two independently evaluated phases:

### 2.1. Generation Phase
- **Primary Metric**: **ROUGE-SU4-F1**
  - **Description**: This is the official metric used in BioASQ Task B for "ideal answer" (paragraph-sized summary) questions. ROUGE-SU4 measures the overlap of n-grams (specifically skip-bigrams with a maximum distance of 4 words) between the AI-generated answer and the expert gold reference.
- **Auxiliary Metrics (Ragas)**: Context Precision, Context Recall, Faithfulness, Answer Correctness, Answer Relevancy.
- *Project Enhancement*: LLM preambles are stripped prior to calculation to prevent artificial deflation of ROUGE scores.

### 2.2. Retrieval Phase
Based on the **BioASQ Task B (Phase A)** evaluation standard:

#### 2.2.1. Document Retrieval Metric (PMID Matching)
- **Primary Metric**: **Mean Average Precision (MAP)**
  - MAP calculates the mean of the Average Precision (AP) across all queries. It evaluates not just whether the system retrieved the correct document (Precision), but also its **ranking**. Documents closer to rank 1 yield a higher MAP.
  - In BioASQ, this is typically truncated at top 10 (MAP@10) due to competition submission limits.
- **Auxiliary Metrics**: Precision@K, Recall@K, F1@K, GMAP@K, MRR (measured at K = 5, 10, 20).

#### 2.2.2. Snippet Retrieval Metric
Snippet metrics evaluate the system's ability to extract the exact text passage (snippet) containing the necessary information from an article, rather than just returning the entire article (PMID).

**Detailed Explanation of Snippet Metrics:**

1. **Official BioASQ Standard:**
   - **Primary Metric:** **Mean F-measure**.
   - **Mechanism:** BioASQ evaluates snippets based on **character-level offsets**.
   - **Rationale for F-measure:** Because a single question might have multiple overlapping gold snippets, Average Precision becomes difficult to interpret with partial overlaps. The character-based F-measure is therefore chosen as the most robust and fair metric.

2. **Project-Specific Implementation (Proxy Snippet Metric):**
   - In the MedKG-RAG project, we do **not** calculate overlap using BioASQ's strict character offsets. Instead, we use a proxy metric: **Snippet Precision@K** and **Snippet Recall@K**.
   - **Matching Logic:** A gold snippet is considered "covered" if the entire `gold_snippet` text appears as a substring within the retrieved `parent_chunk` text (sharing the same PMID).
   - Although less granular than the character-offset F-measure, this provides a fast and highly effective proxy to ensure the retrieved chunks contain the necessary evidence text during development.
