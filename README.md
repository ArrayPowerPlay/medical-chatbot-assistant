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
            ▼                ▼                ▼
   ┌────────────────┐ ┌────────────┐ ┌──────────────────┐
   │  Vector Search │ │ BM25 Search│ │  KG Subgraph     │
   │  (FAISS +      │ │ (Keyword)  │ │  Retrieval (Neo4j)│
   │  MedCPT-Encode)│ │            │ │  hop=2 + HGT     │
   └───────┬────────┘ └─────┬──────┘ └────────┬─────────┘
           │                │          ┌───────▼──────────┐
           │                │          │ LLM Linearization│
           │                │          │ (Subgraph → Text)│
           │                │          └───────┬──────────┘
           └────────────────┼──────────────────┘
                            ▼
                  ┌───────────────────┐
                  │   RRF Reranking   │
                  │   (1st pass)      │
                  └────────┬──────────┘
                           ▼
                  ┌───────────────────┐
                  │  Cross-Encoder    │
                  │  Reranking        │
                  │  (MedCPT, Modal)  │
                  └────────┬──────────┘
                           ▼
                  ┌───────────────────┐
                  │ Head-Tail Prompt  │
                  │ Placement         │
                  └────────┬──────────┘
                           ▼
                  ┌───────────────────┐
                  │   Llama 70B       │
                  │   (Modal GPU)     │
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
git clone https://github.com/your-username/rag-project.git
cd rag-project

# 2. Create virtual environment & install dependencies
uv sync

# 3. Copy environment template
cp .env.example .env
# Edit .env with your API keys and database URIs

# 4. Start Neo4j (via Docker)
docker-compose -f docker/docker-compose.yml up neo4j -d

# 5. Build Knowledge Graph
uv run python scripts/build_kg.py

# 6. Train HGT model (offline, one-time)
uv run python scripts/train_hgt.py

# 7. Ingest documents into vector store
uv run python scripts/ingest_documents.py

# 8. Deploy Modal services
modal deploy modal_deployments/cross_encoder_service.py
modal deploy modal_deployments/llm_service.py

# 9. Start the application
uv run uvicorn api.main:app --reload --port 8000
```

Open your browser at `http://localhost:8000` to access the chatbot.

---

## Configuration

Create a `.env` file based on `.env.example`:

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

### 1. Document Ingestion

```bash
# Place medical documents (PDF, TXT) in data/raw/
uv run python scripts/ingest_documents.py
```

- Loads documents from `data/raw/`
- Splits using recursive character splitter (1200 chars, 200 overlap)
- LLM generates contextual summaries prepended to each chunk
- Embeds with MedCPT-Article-Encoder → stored in FAISS

### 2. Knowledge Graph Construction

```bash
uv run python scripts/build_kg.py
```

- Parses medical data into entities (Disease, Symptom, Drug, Treatment, etc.)
- Creates nodes and relationships in Neo4j
- Schema: `(:Disease)-[:HAS_SYMPTOM]->(:Symptom)`, `(:Drug)-[:TREATS]->(:Disease)`, etc.

### 3. HGT Training (Offline)

```bash
uv run python scripts/train_hgt.py
```

- Loads graph from Neo4j into PyG HeteroData
- Trains Heterogeneous Graph Transformer
- Saves model checkpoint to `models/hgt/`

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