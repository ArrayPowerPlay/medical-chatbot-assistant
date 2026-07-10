# Hyperparameters Configuration

This directory contains the hyperparameter configurations for the MedKG-RAG Chatbot pipeline:
- `retrieval.json`: Hyperparameters for the retrieval phase (e.g., Vector top-K, BM25 top-K, RRF constants).
- `generation.json`: Hyperparameters for the generation phase (e.g., Context limits, LLM temperature).

## Origin and Rationale

**These hyperparameters are exclusively tuned on the BioASQ Task B dataset (Validation Split of 500 questions).** 

Previously, there were separate configuration files for different datasets (e.g., `bioasq_*.json` and `medaesqa_*.json`). They have been consolidated into a single unified set for the following scientific reasons:

1. **Zero-Shot Generalization Test:** The MedAESQA dataset is specifically employed as an *external test set* to evaluate the factual accuracy and robustness of the system on complex, out-of-distribution clinical questions. If we were to tune hyperparameters specifically for MedAESQA (which only contains ~42 questions), we would risk overfitting to a very small sample size and breaking its purpose as a pure, unseen test set.
2. **Robustness Validation:** By applying the BioASQ-optimized parameters directly to MedAESQA without modifications, any metrics obtained accurately reflect the system's "zero-shot transfer" capabilities. It proves that the RAG pipeline is robust enough to handle new query distributions in a real-world scenario where the query format is unpredictable.

Therefore, across all evaluation scripts in the project, the system strictly loads these BioASQ-tuned parameters to maintain academic rigor and prevent data leakage or overfitting.
