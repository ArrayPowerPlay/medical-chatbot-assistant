<p align="center">
  <h1 align="center">🏥 Med Assistant</h1>
  <p align="center">
    <strong>Advanced Medical Q&A Chatbot powered by KG-Augmented RAG</strong>
  </p>
</p>

## 📋 Table of Contents
1. [Overview & Domain](#-overview--domain)
2. [Web Features & User Roles](#-web-features--user-roles)
3. [Data Sources](#-data-sources)
4. [Advanced RAG Techniques](#-advanced-rag-techniques)
5. [End-to-End Execution Pipeline](#-end-to-end-execution-pipeline)
6. [Tech Stack & Models](#%EF%B8%8F-tech-stack--models)
7. [Layered Architecture](#%EF%B8%8F-layered-architecture)
8. [Evaluation & Ablation Study](#-evaluation--ablation-study)
9. [Getting Started](#-getting-started)

---

## 🎯 Overview & Domain

**Med Assistant** is a state-of-the-art Medical Question-Answering chatbot designed to provide highly accurate, evidence-based answers to complex health-related queries. 

**Domain Specification:** The system is explicitly trained and optimized for the **Disease-Drug-Target** medical domain. Its knowledge base is built upon hundreds of thousands of PubMed articles meticulously filtered using MeSH Tree categories C (Diseases) and D (Chemicals and Drugs), combined with comprehensive medical Knowledge Graphs.

---

## 💻 Web Features & User Roles

The web platform provides customized, role-based experiences with a modern, responsive React UI supporting dark/light themes and real-time chat streaming.

### 1. Guest User
- Permitted to ask up to **10 free questions**.
- **Real-time Limit Indicator:** Displayed dynamically at the bottom of the latest AI response to warn guests of their remaining quota.
- Non-persistent chat history (conversations are cleared upon leaving the session).

### 2. Registered User
- Unlimited chatting capabilities.
- Persistent conversation history with features to rename, pin, or delete chat threads.
- **Transparency & Verifiability:** Ability to toggle and view the exact medical sources (Knowledge Graph paths, PubMed document snippets, PMIDs, and similarity scores) utilized by the AI for every answer.
- **Feedback Loop:** Can provide Like/Dislike feedback along with optional textual comments on AI responses to help improve the system.

### 3. Administrator
- **Analytics Dashboard:** Monitor system statistics, including total user counts, active guests, new sign-ups, daily question volumes, and feedback metrics.
- **Manage Users:** Search users in real-time (using partial `ILIKE` matching on email/username), view their statistics, delete accounts, or force-reset passwords.
- **Conversation Oversight:** Read-only access to any user's complete conversation history, including the AI's retrieved sources, to monitor answer quality and hallucination rates.
- **Feedback Management:** Dedicated UI panels to review Good and Bad feedback side-by-side with the AI's response, enabling continuous manual review and improvement of the RAG pipeline.

---

## 🗄️ Data Sources

## 🗂️ Data Sources & Collection

### 1. Knowledge Graph (PrimeKG)
The Knowledge Graph is built upon **PrimeKG** (Harvard). To ensure domain specificity and optimal performance, the raw PrimeKG dataset is rigorously filtered down to the Disease-Drug-Target triad. This is achieved by scanning the graph and keeping only the edges where both endpoints belong to our target node types (e.g., *disease, drug, gene/protein*) and the connection represents a medically relevant relationship (e.g., *indication, side effect, drug-drug interaction, target*).

### 2. Text Corpus (BioASQ & MedAESQA)
The unstructured text corpus (powering Vector and BM25 retrieval) is constructed directly from the validation and test splits of our datasets. Specifically, we iterate through the QA pairs, extract all referenced PubMed IDs (PMIDs), and dynamically query the NCBI E-utilities API to download the exact title and abstract for each PMID. These downloaded articles are then parsed and processed through a Parent-Child Chunker to populate the semantic index.

---

## 🚀 Advanced RAG Techniques

This project goes beyond naive Retrieval-Augmented Generation by implementing a suite of cutting-edge optimization techniques to maximize accuracy and minimize latency:

- **Query Analyzer & Intent Classification:** Before executing any retrieval, the raw user input is processed by a fast LLM to:
  - **Rewrite the Query:** Resolve pronouns based on conversation history, standardize layperson terms to clinical equivalents, and fix typographical errors.
  - **Classify User Intent:** Determine if the question requires medical retrieval. If the user makes small talk or asks questions *entirely outside* the medical/health domain (e.g., asking about cats, weather, general programming), the system categorizes it as `no_rag_needed`. This completely bypasses the heavy RAG pipeline, directly answering the user to save latency and computing power.
  - **Extract Entities:** Accurately identify Diseases, Drugs, Effect Phenotypes, and Gene/Proteins.
  - **Classify Question Type:** Identify if the question is a Factoid, Summary, Yes/No, or List, which influences prompt construction.
- **Parent-Child Chunking:** PubMed abstracts are split into small `Child` chunks (for high-precision vector embeddings and retrieval). Once a child chunk is matched, the system retrieves its larger `Parent` chunk to provide the LLM with comprehensive, unbroken context.
- **Parallel Retrieval Pipeline:** Concurrently fetches context from three different sources: Vector Search (Semantic), BM25 (Keyword), and Neo4j (Knowledge Graph paths).
- **RRF (Reciprocal Rank Fusion):** Text results from Vector and BM25 searches are mathematically fused to balance semantic meaning with exact keyword matches.
- **Cross-Encoder Reranking:** The fused text results, alongside linearized Knowledge Graph triples, are then reranked by a specialized Medical Cross-Encoder to ensure the most relevant context sits at the absolute top.
- **KG Merger:** Extracted multi-hop paths from the Knowledge Graph often share prefixes (e.g., Disease -> Symptom). The KG Merger consolidates these overlapping paths to eliminate redundant context, preserving the LLM's limited context window.
- **Head-Tail Prompt Placement:** Retrieved contexts are strategically placed at the very beginning and very end of the prompt to mitigate the "lost-in-the-middle" phenomenon typical of large language models.

---

## ⚙️ End-to-End Execution Pipeline

When a user submits a query, the system executes the following chronological pipeline:

1. **History Normalization:** The system extracts the last 5 turns of conversation to provide temporal context.
2. **Query Analysis:** An LLM analyzes the query. If the intent is `no_rag_needed`, it jumps directly to Step 9. Otherwise, it outputs a rewritten query, extracted entities, and the question type.
3. **Embedding:** The rewritten query and extracted entities are converted into high-dimensional vectors using MedCPT encoders.
4. **Parallel Retrieval:**
   - *Vector Search:* Retrieves semantically similar Child chunks from Weaviate and resolves their Parent texts.
   - *Keyword Search:* Executes a BM25 keyword query in Weaviate.
   - *Graph Retrieval:* Neo4j locates anchor nodes based on extracted entities and performs a 2-hop graph expansion to find relational paths.
5. **Rank Fusion:** Textual results (Vector + BM25) are combined using Reciprocal Rank Fusion (RRF).
6. **Reranking:** Both text results and linearized KG paths are scored against the query by the MedCPT Cross-Encoder and sorted by relevance.
7. **KG Path Merging:** Overlapping top-ranked KG paths are merged to compress the context.
8. **Prompt Construction:** The final prompt is built by interleaving Text and KG results, applying the Head-Tail placement strategy based on the reranker scores.
9. **Streaming Generation:** The Llama-3.3-70B model synthesizes the final answer and streams it back to the user interface via Server-Sent Events (SSE).

---

## 🛠️ Tech Stack & Models

**AI Models:**
- **Generator LLM:** `meta-llama/Llama-3.3-70B-Versatile` (via Groq API for ultra-fast inference).
- **Embeddings:** `ncbi/MedCPT-Article-Encoder` & `ncbi/MedCPT-Query-Encoder`.
- **Cross-Encoder:** `ncbi/MedCPT-Cross-Encoder`.
- **Evaluator Models:** `gpt-4o-mini` and `text-embedding-3-small` (via Ragas).

**Frameworks & Infrastructure:**
- **Frontend:** React, TypeScript, Vite, TailwindCSS.
- **Backend:** Python, FastAPI, Uvicorn, Server-Sent Events (SSE).
- **Vector & Keyword Store:** Weaviate.
- **Relational Database:** PostgreSQL (for User Data, Chat History, Feedbacks, and Parent Chunks).
- **Graph Database:** Neo4j (for PrimeKG data).

---

## 🏗️ Layered Architecture

Med Assistant follows a clean, modular, layered architecture:

1. **Presentation Layer (Frontend):** A responsive Single Page Application (SPA) built in React, handling complex UI states, real-time chat streaming, theme toggling, and role-based views.
2. **API & Routing Layer:** FastAPI endpoints that expose RESTful resources, manage JWT authentication/authorization, and handle asynchronous SSE streams for the chat interface.
3. **Orchestration Layer (RAG Pipeline):** The core engine that systematically coordinates the Query Analyzer, Parallel Retriever, RRF Manager, Reranker, KG Merger, Prompt Builder, and the LLM Generator.
4. **Data Access & Storage Layer:** Abstracted interface repositories communicating with PostgreSQL, Weaviate, and Neo4j.
5. **Data Pipeline Layer:** Offline scripts responsible for parsing BioASQ/MedAESQA datasets, chunking text, generating embeddings, and hydrating the databases.

---

## 📊 Evaluation & Ablation Study

To ensure the highest standard of medical accuracy, the system is evaluated rigorously. The evaluation process is split into two distinct tracks: **Retrieval Evaluation** and **Generation Evaluation**, utilizing two prominent medical datasets.

### Evaluation Datasets
- **BioASQ Task B:** A premier benchmark dataset in biomedical semantic indexing and question answering. It contains professional medical QA pairs meticulously curated by biomedical experts. We use 500 Test samples from this dataset for **both Retrieval and Generation Evaluation**.
- **MedAESQA:** A specialized, high-quality medical QA dataset containing exactly 40 complex clinical questions, designed to evaluate factual accuracy and evidence-grounding to combat LLM hallucinations. Used for **both Retrieval and Generation Evaluation**.

### 1. Retrieval Evaluation
This phase evaluates the backend's ability to fetch the correct context from our 300,000+ document corpus and Knowledge Graph.
- **Metrics Evaluated (at K = 5, 10, 20):**
  - **BioASQ:** *Precision@K*, *Recall@K*, *F1@K*, *MAP@K*, *GMAP@K*, *Snippet Precision@K*, *Snippet Recall@K*, *Snippet F1@K*, and *MRR*.
  - **MedAESQA:** *Precision@K*, *Recall@K*, and *F1@K*.
- **Test Results (BioASQ Document Metrics at K=10):** 
  - **MRR:** 0.8626
  - **Precision@10:** 0.4120
  - **Recall@10:** 0.7425
  - **F1@10:** 0.4530
  - **MAP@10:** 0.6892
  - **GMAP@10:** 0.4420
- **Ablation Study:** We conducted ablation studies comparing the **Full RAG System** (Vector + BM25 + KG + Cross-Encoder) against a **No-KG Hybrid** (Vector + BM25 + Cross-Encoder) and a **Pure Baseline** (LLM + Vector, no BM25, no KG). The results confirm that adding Keyword Search (BM25) significantly improves the retrieval of hard facts (like drug names), while the Knowledge Graph greatly enhances multi-hop reasoning and citation faithfulness.

### 2. Generation Evaluation
This phase evaluates the LLM's final synthesized response utilizing the retrieved context, comparing it against expert ground-truth answers.
- **Metrics Evaluated:** 
  - **BioASQ:** *Ragas metrics* (*Context Precision, Context Recall, Faithfulness, Answer Correctness, Answer Relevancy*) and *ROUGE-SU4-F1*.
  - **MedAESQA:** *ROUGE-SU4* (*Precision, Recall, F1*) and *Citation Metrics* (*Citation-Precision, Citation-Recall, Citation-F1*).
- **Test Results (BioASQ Generation):**
  - **ROUGE-SU4-F1:** 0.1528
  - **Context Precision:** 0.6946
  - **Context Recall:** 0.9437
  - **Faithfulness:** 0.9020
  - **Answer Correctness:** 0.5984
  - **Answer Relevancy:** 0.7846

---

## 🚀 Getting Started

### Prerequisites
- Python >= 3.11
- Node.js >= 18
- Docker & Docker Compose (for PostgreSQL, Neo4j, Weaviate)
- API Keys: Groq, OpenAI (for evaluation)

### Quick Start

1. **Clone the repository & Install dependencies:**
   ```bash
   git clone https://github.com/your-username/medical-chatbot-assistant.git
   cd medical-chatbot-assistant
   pip install -r requirements.txt
   cd frontend && npm install && cd ..
   ```

2. **Setup Environment:**
   Copy `.env.example` to `.env` and fill in your database credentials and API keys.

3. **Launch Infrastructure:**
   ```bash
   docker-compose up -d
   ```

4. **Start the Backend API:**
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```

5. **Start the Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

Open your browser at `http://localhost:5173` to interact with **Med Assistant**.