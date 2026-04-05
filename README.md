<p align="center">
  <h1 align="center">🏥 MedKG-RAG Chatbot</h1>
  <p align="center">
    <strong>Medical Q&A Chatbot powered by Knowledge Graph RAG</strong>
  </p>
  <p align="center">
    Combines Vector Search · BM25 · Medical Knowledge Graph · HGT · Cross-Encoder Reranking · Llama 70B
  </p>
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Data Pipeline](#data-pipeline)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**MedKG-RAG Chatbot** is an advanced medical question-answering system that leverages **Knowledge Graph RAG (Retrieval-Augmented Generation)** to provide accurate, context-rich answers about diseases, symptoms, treatments, and medical knowledge.

### Key Features

- 🔍 **Triple Retrieval Pipeline** — Parallel Vector Search (FAISS), Keyword Search (BM25), and Knowledge Graph (Neo4j) retrieval
- 🧠 **Medical Knowledge Graph** — Neo4j-based medical KG enhanced with HGT (Heterogeneous Graph Transformers)
- 📊 **Two-Stage Reranking** — RRF fusion + Cross-Encoder reranking for high-precision results
- 🤖 **Llama 70B Generation** — State-of-the-art LLM for medical answer synthesis
- 🌐 **Modern Web UI** — Clean, responsive chat interface with dark/light theme
- ☁️ **Cloud GPU Inference** — Modal platform for Cross-Encoder and LLM inference

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Query Analysis  │
                    │  (Entity Extract) │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                │
   ┌────────────────┐ ┌────────────┐          │
   │  Vector Search │ │ BM25 Search│          │
   │  (FAISS +      │ │ (Keyword)  │          │
   │  MedCPT-Encode)│ │            │          │
   └───────┬────────┘ └─────┬──────┘          │ ┌──────────────────┐
           │                │                 └─►  KG Subgraph     │
           └───────┬────────┘                   │  Retrieval (Neo4j)│
                   ▼                            │  hop=2 + HGT     │
         ┌───────────────────┐                  └───────┬──────────┘
         │   RRF (Text Search)│                         │ 
         │    (Vector + BM25) │               ┌─────────▼──────────┐
         └────────┬───────────┘               │ Rule-based Linear. │
                  │                           │ (Subgraph → Text)  │
                  │                           └────────┬───────────┘
                  └────────────────┬───────────────────┘
                                   ▼
                          ┌───────────────────┐
                          │   Cross-Encoder   │
                          │     Reranking     │
                          │  (Text Search + KG)│
                          └────────┬──────────┘
                                   ▼
                          ┌───────────────────┐
                          │ Head-Tail Prompt  │
                          │ Placement         │
                          └────────┬──────────┘
                                   ▼
                          ┌───────────────────┐
                          │   Llama 70B       │
                          │   (Groq API)      │
                          └────────┬──────────┘
                                   ▼
                          ┌───────────────────┐
                          │   Answer          │
                          └───────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Embedding** | `ncbi/MedCPT-Article-Encoder` |
| **Vector Store** | FAISS (with cosine similarity) |
| **Keyword Search** | BM25 (rank-bm25) |
| **Knowledge Graph** | Neo4j |
| **Graph ML** | HGT via PyTorch Geometric |
| **Cross-Encoder** | `ncbi/MedCPT-Cross-Encoder` |
| **LLM** | Llama 70B |
| **Cloud GPU** | Modal |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS |
| **Chunking** | Recursive Character Splitter + LLM contextual enrichment |

---

## Getting Started

### Prerequisites

- **Python** >= 3.13
- **Neo4j** (local or Docker)
- **Modal** account (for cloud GPU inference)
- **CUDA-capable GPU** (optional, for local embedding)
- **uv** (recommended package manager)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/medical-chatbot-assistant.git
cd medical-chatbot-assistant

# 2. Create virtual environment & install dependencies
uv sync

# 3. Copy environment template
cp .env.example .env
# Edit .env with your API keys (GROQ_API_KEY, HF_TOKEN) and database URIs

# 4. Start Infrastructure (via Docker)
docker-compose -f docker/docker-compose.yml up -d

# 5. Build Knowledge Graph (PrimeKG)
uv run python scripts/build_kg.py

# 6. Preprocess BioASQ Data (Task A & B)
# Download 300,000 articles (Domain: Disease-Drug-Target)
uv run python src/dataset_builder/preprocess_bioasq_taskA.py

# Process QA pairs (500 Test / 500 Validation)
uv run python src/dataset_builder/preprocess_bioasq_taskB.py

# 7. Ingest documents into vector store & BM25
uv run python src/dataset_builder/index_builder.py

# 8. Train HGT model (offline, one-time)
uv run python scripts/train_hgt.py

# 9. Deploy Modal services
modal deploy modal_deployments/cross_encoder_service.py

# 10. Start the application
uv run uvicorn api.main:app --reload --port 8000
```

Open your browser at `http://localhost:8000` to access the chatbot.

---

## Configuration

Create a `.env` file based on `.env.example`. Ensure `HF_TOKEN` is provided for streaming datasets.

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Modal
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret

# FAISS
FAISS_INDEX_PATH=./vectorstore/faiss_index

# Pipeline
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
KG_HOP_DEPTH=2

# LLM
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.3
```

---

## Data Pipeline

### 1. BioASQ Task A: Document Corpus
- **Source**: `jmhb/pubmed_bioasq_2022` (Parquet)
- **Scale**: 300,000 articles
- **Filtering**: Strictly filtered by MeSH Tree Numbers (Categories C and D) to focus on the **Disease-Drug-Target** domain.
- **Process**: `uv run python src/dataset_builder/preprocess_bioasq_taskA.py`
- **Output**: Appended to `data/corpus/corpus.jsonl` with automatic deduplication.

### 2. BioASQ Task B: Q&A Preprocessing
- **Source**: BioASQ Professional Training Set
- **Scale**: 1,000 valid medical QA samples.
- **Splitting**:
  - **Test Set**: 500 samples (`data/test/test.jsonl`)
  - **Validation Set**: 500 samples (`data/val/val.jsonl`)
- **Process**: `uv run python src/dataset_builder/preprocess_bioasq_taskB.py`

### 3. Vector & Keyword Indexing
- **Vector Search**: FAISS index with `MedCPT-Article-Encoder`.
- **Keyword Search**: Elasticsearch BM25 index.
- **Chunking**: 1200 characters with 200 overlap, enhanced with Gemini 2.5 contextual summaries.
- **Process**: `uv run python src/dataset_builder/index_builder.py`

### 4. Knowledge Graph (PrimeKG)
- **Source**: PrimeKG (Disease-Symptom-Drug subset).
- **Storage**: Neo4j.
- **Graph ML**: HGT model trained to provide semantic graph embeddings.
- **Process**: `uv run python scripts/build_kg.py`

---

## Deployment

### Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

### Modal (Cloud GPU)

```bash
# Deploy cross-encoder reranking service
modal deploy modal_deployments/cross_encoder_service.py

# Deploy Llama 70B inference service
modal deploy modal_deployments/llm_service.py
```

---

## Project Structure

See [PROJECT_ARCHITECTURE.md](./PROJECT_ARCHITECTURE.md) for the complete directory tree.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-retrieval-method`)
3. Commit your changes (`git commit -m 'Add new retrieval method'`)
4. Push to the branch (`git push origin feature/new-retrieval-method`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
#   m e d i c a l - c h a t b o t - a s s i s t a n t  
 #   m e d i c a l - c h a t b o t - a s s i s t a n t  
 #   m e d i c a l - c h a t b o t - a s s i s t a n t  
 #   m e d i c a l - c h a t b o t - a s s i s t a n t  
 