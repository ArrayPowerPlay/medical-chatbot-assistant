# 🧠 OVERVIEW — MedKG-RAG Chatbot

> **Purpose**: This file contains ALL essential context for the project.
> Reading this single file should be sufficient to fully understand the project's scope, architecture, and implementation details when starting a new session.

---

## 1. Project Identity

- **Name**: MedKG-RAG Chatbot (Medical Knowledge Graph RAG)
- **Goal**: Build a web-based medical Q&A chatbot that answers questions about diseases, symptoms, treatments using KG-RAG technology
- **Type**: Academic/Research project
- **Language (Code)**: Python 3.13+ (backend), Vanilla HTML/CSS/JS (frontend)
- **Language (Data/UI)**: English only
- **Package Manager**: uv
- **Backend Framework**: FastAPI
- **Project Root**: `d:\Program Files\VSCode\rag-project`

---

## 2. Pipeline Architecture

The system has **two main phases**: **Retrieval** and **Generation**.

### 2.1 Retrieval Phase — 3 Parallel Streams + Stage-based Reranking

> **IMPORTANT: Retrieval Flow**
> 1. Run 3 retrieval streams in **parallel**: Vector Search, Keyword Search, and KG Search.
> 2. **RRF (Reciprocal Rank Fusion)** merges ONLY **Vector Search + BM25** → produces **Text Search** results.
> 3. **Cross-Encoder** (MedCPT-Cross-Encoder) reranks the **combined pool** of **Text Search** (from RRF) AND **KG Search** (linearized subgraphs) together into a single final ranked list.
> 4. Final context is assembled from the reranked list using head-tail placement.

#### Stream 1: Vector Search (Semantic)
| Setting | Value |
|---|---|
| **Vector DB** | Weaviate (Local Docker, Cosine Similarity) |
| **Embedding Model** | `ncbi/MedCPT-Article-Encoder` (for documents) |
| **Query Model** | `ncbi/MedCPT-Query-Encoder` (for queries) |
| **Chunking** | **Parent-Child Chunking** (Local, non-LLM) |
| **Parent Size** | 1200 - 1500 characters |
| **Child Size** | 256 - 400 characters (embedded) |
| **Contextual Strategy**| Search is performed on **Child Chunks**; context for LLM is retrieved from the corresponding **Parent Chunk**. |
| **Metadata Mapping** | A mapping between Child `parent_id` and Parent text is stored in **SQLite** (`parent_chunks.db`) for O(log n) lookup. |

> **Key Detail — Dual Encoder Setup**: MedCPT uses an **asymmetric** architecture. `MedCPT-Query-Encoder` encodes the user query, while `MedCPT-Article-Encoder` encodes document chunks. This is by design and yields better retrieval quality than using a single encoder for both.

> **Key Detail — Parent-Child Strategy**: Instead of LLM-based enrichment, we use a structural approach. Small "Child" chunks provide high precision for vector/keyword matching, while their larger "Parent" chunks provide the full semantic context needed by the LLM to generate accurate answers. |,StartLine:33,TargetContent:

#### Stream 2: Keyword Search (Lexical)
| Setting | Value |
|---|---|
| **Engine** | Weaviate (Built-in BM25) |
| **Algorithm** | BM25 |
| **Index** | Same **Child Chunks** as Vector stream |
| **Persistence** | Persisted on disk (Weaviate volumes) |
| **Purpose** | Captures exact medical terminology that semantic embeddings might miss. |

#### Stream 3: Medical Knowledge Graph
| Setting | Value |
|---|---|
| **Database** | Neo4j |
| **Query Depth** | 2-hop subgraph retrieval |
| **Graph ML** | HGT (Heterogeneous Graph Transformers) trained end-to-end for downstream tasks |
| **Entity Extraction** | Llama 3.3 70B via Groq API (LLM-based NER, no RE needed) |
| **Linearization** | Rule-based Python templates (no LLM needed) |

> **KG Query Flow**:
> 1. User query → **LLM-based NER** (Llama 70B via Groq) → Extract medical entities (disease, symptom, drug)
> 2. No Relation Extraction needed — KG already contains structured relationships
> 3. Use extracted entities to query Neo4j via **HGT semantic embeddings** (semantic similarity, not keyword matching)
> 4. Retrieve 2-hop subgraph around matched entities
> 5. **Rule-based linearization**: Python templates convert subgraph → natural language text

> **KG Schema** (core node types — filtered from PrimeKG):
> - `Disease`, `Symptom`, `Drug` (primary focus)
> - Other PrimeKG node types are excluded

> **KG Relationships** (examples from PrimeKG):
> - `(:Disease)-[:HAS_SYMPTOM]->(:Symptom)`
> - `(:Drug)-[:TREATS]->(:Disease)`
> - `(:Disease)-[:AFFECTS]->(:BodyPart)`
> - `(:Disease)-[:DIAGNOSED_BY]->(:LabTest)`
> - `(:Disease)-[:HAS_RISK_FACTOR]->(:RiskFactor)`

> **HGT Training Pipeline** (offline, one-time):
> - **Paper**: [Heterogeneous Graph Transformer (Hu et al., 2020)](https://arxiv.org/abs/2003.01332)
> - **Architecture**: Node- and edge-type dependent attention parameters for heterogeneous graphs
> - **Training approach**: **End-to-end** for downstream tasks — NOT self-supervised
> - HGT layers learn graph representations that are directly optimized for the specific downstream task (e.g., node classification, link prediction)
> - The learned embeddings capture semantic relationships between different entity types (Disease, Symptom, Drug)
> - **Pipeline**:
>   1. Export Neo4j graph → PyTorch Geometric `HeteroData`
>   2. Train HGT model end-to-end for downstream task
>   3. Save checkpoint to `models/hgt/`
>   4. At inference: Use trained HGT embeddings for **semantic similarity** querying on graph (instead of keyword-based Cypher queries)

> **Subgraph Linearization** (Rule-Based):
> After retrieving a 2-hop subgraph from Neo4j, **Python rule-based templates** convert the graph structure into natural language text. Example template:
> - `"Drug X treats Disease Y"` from `(:Drug {name: X})-[:TREATS]->(:Disease {name: Y})`
> - `"Disease Y has symptom Z"` from `(:Disease {name: Y})-[:HAS_SYMPTOM]->(:Symptom {name: Z})`
> This linearized text is then pooled with text retrieval results for cross-encoder reranking.

### 2.2 Reranking Phase — 2 Stages

#### Stage 1: Reciprocal Rank Fusion (RRF) — Text Retrieval Only
- Merges results from **Vector Search + BM25** ONLY (NOT KG)
- Formula: `RRF(d) = Σ 1 / (k + rank_i(d))` where `k = 60` (standard)
- Produces a unified **Text Retrieval** ranked list

#### Stage 2: Cross-Encoder Reranking — Merge All Streams
| Setting | Value |
|---|---|
| **Model** | `ncbi/MedCPT-Cross-Encoder` |
| **Deployment** | Modal (cloud GPU) |
| **Input** | (query, passage) pairs from BOTH Text Retrieval (RRF output) AND KG Retrieval (linearized subgraph) |
| **Function** | Reranks the **combined pool** of text + KG results into a single final ranked list |
| **Output** | Top-K re-scored passages (unified from both streams) |

### 2.3 Generation Phase

#### Query Rewriting (Pre-Retrieval)
- **Model**: Llama 3.3 70B via **Groq API**
- **Purpose**:
  - Fix spelling errors in user query
  - Rewrite query to be more specific/retrievable
  - Connect with conversation history (multi-turn context)
- Runs BEFORE the 3 parallel retrieval streams

#### Prompt Construction: Head-Tail Placement
- To avoid the **Lost-in-the-Middle** problem:
  - Most relevant passages → placed at the **HEAD** of the context
  - Second-most relevant passages → placed at the **TAIL** of the context
  - Least relevant passages → placed in the **MIDDLE**
- This leverages LLM attention patterns that favor beginning and end of context

#### LLM Generation
| Setting | Value |
|---|---|
| **Model** | `meta-llama/Llama-3.3-70B-Versatile` |
| **API** | **Groq API** |
| **Temperature** | 0.3 (factual, low creativity) |
| **Max Tokens** | 2048 |
| **System Prompt** | Medical assistant, cite sources, refuse if context insufficient |

---

## 3. Data Sources & Evaluation

### 3.1 Document Corpus (for FAISS Vector DB + Elasticsearch BM25)
| Setting | Value |
|---|---|
| **Source** | BioASQ PubMed Annual Baseline Corpus (`jmhb/pubmed_bioasq_2022`) |
| **Size** | 300,000 articles |
| **Language** | English |
| **Selection Criteria** | Filtered by MeSH Tree Numbers to focus on the **Disease-Drug-Target** domain. Includes all PMIDs from the Test set. |

### 3.2 Knowledge Graph Data
| Setting | Value |
|---|---|
| **Source** | PrimeKG (existing structured dataset) |
| **Filtering** | Keep only nodes/relationships related to Disease, Symptom, Drug. Discard the rest. |

### 3.3 Evaluation Datasets (BioASQ Task B)

#### Q&A Dataset Split
| Dataset | Split | Size |
|---|---|---|
| **BioASQ Task B** | **Validation** (used for hyperparameter tuning) | 500 questions |
| **BioASQ Task B** | **Test** (final evaluation) | 500 questions |

#### External Evaluation
| Dataset | Split | Size | Phase |
|---|---|---|---|
| MedQA | Validation | Full set | Generation |
| MedQA | Test | Full set | Generation |

### 3.4 Evaluation Metrics

#### Text Retrieval (after merging vector + BM25 via RRF)
- **Recall@K**
- **Evaluation Method**: For a given question from BioASQ Task B:
  1. Retrieve top-$k$ chunks from the parallel pipeline.
  2. Map retrieved FAISS indices to their original PMIDs using `faiss_metadata.jsonl`.
  3. Compare retrieved PMIDs with the "golden" relevant document PMIDs.
  4. (Optional) For exact matches, compare text snippets in the query with the content of the retrieved chunk.

#### Generation
- **Exact Match / F1** (standard QA metrics)
- **Answer Relevance** (RAGAS)
- **Answer Faithfulness** (RAGAS)
- **Context Precision** (RAGAS)

---

## 4. LLM/LM Strategy Summary

| Task | Model | Platform | When |
|---|---|---|---|
| **Parent-Child Chunking** | Rule-based (LangChain) | Local CPU | Offline (batch, one-time) |
| **Entity Extraction (NER)** | Llama 3.3 70B | Groq API | Inference (per query) |
| **HGT Training** | HGT (PyG) | Local / Kaggle GPU | Offline (one-time, end-to-end) |
| **Cross-Encoder Reranking** | MedCPT-Cross-Encoder | Modal (cloud GPU) | Inference (per query) |
| **Query Rewriting** | Llama 3.3 70B Versatile | Groq API | Inference (per query) |
| **KG Linearization** | Rule-based Python templates | Local (no LLM) | Inference (per query) |
| **Answer Generation** | Llama 3.3 70B Versatile | Groq API | Inference (per query) |

---

## 5. External Services & Dependencies

| Service | Purpose | Deployment |
|---|---|---|
| **Neo4j** | Medical Knowledge Graph (PrimeKG) | Docker container (local) |
| **Weaviate** | Vector + BM25 hybrid search | Docker container (local) |
| **PostgreSQL** | Conversation history (multi-turn) | Docker container (local) |
| **SQLite** | Parent chunk storage & lookup | Local file (`parent_chunks.db`) |
| **Modal** | GPU inference (Cross-Encoder only) | Cloud (modal.com) |
| **Groq API** | LLM inference (Llama 70B) | Cloud (groq.com) |

### Key Python Packages
```
uv add weaviate-client             — Weaviate Python V4 driver
langchain, langchain-community — Document loading, text splitting
neo4j                     — Neo4j Python driver
torch, transformers       — MedCPT models
torch-geometric           — HGT model (PyG)
modal                     — Cloud GPU deployment (Cross-Encoder)
groq                      — Groq API client (Llama 70B)
sqlite3                   — Built-in Python DB (Parent storage)
pydantic, pydantic-settings — Config & validation
ragas                     — RAG evaluation framework
```

---

## 6. Key File Mapping

| File / Directory | Purpose |
|---|---|
| `config/settings.py` | All env vars, constants, paths via Pydantic Settings |
| `src/query/query_rewriter.py` | Query rewriting via Groq (spell fix, specificity, history) |
| `src/query/query_extractor.py` | LLM-based medical NER (Llama 70B via Groq, no RE) |
| `src/embeddings/medcpt_embedder.py` | MedCPT dual encoder (Query-Encoder + Article-Encoder) |
| `src/dataset_builder/preprocess_bioasq_taskA.py` | Load BioASQ PubMed articles (Task A) |
| `src/dataset_builder/preprocess_bioasq_taskB.py` | Preprocess Q&A for Task B (test, val split) |
| `src/dataset_builder/parent_child_chunker.py` | Parent-Child structural chunking (Local CPU) |
| `src/storage/parent_store.py` | SQLite lookup for Parent chunks (by parent_id) |
| `src/storage/weaviate_client.py` | Weaviate Client (Vector + BM25 on Children) |
| `src/retrieval/vector_search.py` | Weaviate vector search (Child -> Parent mapping) |
| `src/retrieval/keyword_search.py` | Weaviate BM25 search (Child -> Parent mapping) |
| `src/retrieval/kg_search.py` | Neo4j 2-hop subgraph retrieval (with HGT embeddings) |
| `src/retrieval/kg_linearization.py` | Rule-based subgraph → text linearization templates |
| `src/retrieval/parallel_retriever.py` | Orchestrates 3 parallel streams |
| `src/reranking/rrf.py` | Reciprocal Rank Fusion (Vector + BM25 only) |
| `src/reranking/cross_encoder.py` | MedCPT-Cross-Encoder via Modal (merges text + KG) |
| `src/generation/prompt_builder.py` | Head-tail context placement |
| `src/generation/llm_generator.py` | Llama 70B generation via Groq API |
| `src/kg/neo4j_client.py` | Neo4j connection & Cypher queries |
| `src/kg/hgt/model.py` | HGT model definition (arxiv:2003.01332) |
| `src/kg/hgt/trainer.py` | HGT end-to-end training for downstream tasks |
| `src/pipeline/rag_pipeline.py` | End-to-end: query → retrieval → rerank → generate |
| `api/main.py` | FastAPI app entry point (includes CORS config) |
| `api/routes/chat.py` | POST /chat endpoint |
| `modal_deployments/cross_encoder_service.py` | Modal Cross-Encoder endpoint |
| `frontend/index.html` | Chat UI |
| `scripts/ingest_documents.py` | Document ingestion pipeline |
| `scripts/build_kg.py` | PrimeKG → Neo4j construction script |
| `scripts/train_hgt.py` | HGT training script |
| `scripts/evaluate_retrieval.py` | BioASQ Phase A retrieval evaluation |
| `scripts/evaluate_generation.py` | BioASQ Phase B + MedQA generation evaluation |

---

## 7. API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Send user question, receive AI answer |
| `GET` | `/api/health` | Health check |
| `GET` | `/` | Serve frontend (static files) |

### POST /api/chat — Request
```json
{
  "question": "What are the symptoms of diabetes?",
  "conversation_id": "optional-uuid",
  "top_k": 5
}
```

### POST /api/chat — Response
```json
{
  "answer": "Diabetes presents with several key symptoms...",
  "sources": [
    {"type": "text_retrieval", "content": "...", "score": 0.92},
    {"type": "kg", "content": "...", "subgraph": "..."}
  ],
  "conversation_id": "uuid"
}
```

---

## 8. Frontend Design

- **Type**: Single Page Application (Vanilla HTML/CSS/JS)
- **Theme**: Dark/Light toggle, medical-themed color palette
- **Layout**: Chat-centric with message bubbles (user vs. bot)
- **Features**:
  - Markdown rendering for bot responses
  - Source citation display (expandable)
  - Loading animation during generation
  - Responsive design (mobile-friendly)
  - Suggested starter questions

---

## 9. Data Flow Summary

```
User Question
    │
    ├─── Query Rewriting (Llama 70B via Groq)
    │    ├── Fix spelling errors
    │    ├── Make query more specific/retrievable
    │    └── Connect with conversation history
    │
    ├─── LLM Entity Extraction (Llama 70B via Groq)
    │    └── Extract: disease, symptom, drug entities (NER only, no RE)
    │
    ├─── [Parallel Retrieval Streams]
    │    │
    │    ├── Weaviate Vector (Children) ──┐
    │    │             │                  ├─── Parent-level aggregation
    │    │             └── Mapping ───────┤    (max score per parent)
    │    │                                ├─── RRF Fusion (Text Search)
    │    ├── Weaviate BM25 (Children) ────┤           │
    │    │             │                  │           │
    │    │             └── Mapping ───────┘           │
    │    │                                            │
    │    └── KG Search (Neo4j + HGT embeddings)       │
    │                   │                             │
    │                   └── Rule-based Linearization  │
    │                        (Python templates → Text)│
    │                               │                 │
    │                               └────────┬────────┘
    │                                        ▼
    ├─── Cross-Encoder Reranking (MedCPT, Modal GPU)
    │    └── Rerank combined pool: Text Search + KG Search
    │        → Single unified ranked list
    │
    ├─── Head-Tail Placement ──► Build context prompt
    │
    ├─── Llama 3.3 70B (Groq API) ──► Generate answer
    │
    └─── Response ──► Return to user with sources
```

---

## 10. Environment Variables Reference

```env
# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=

# Weaviate
WEAVIATE_URL=http://localhost:8080
WEAVIATE_GRPC_PORT=50051

# Paths
SQLITE_PARENT_DB_PATH=./vectorstore/parent_chunks.db
HGT_MODEL_PATH=./models/hgt/
RAW_DATA_PATH=./data/raw/

# Pipeline Params
RETRIEVAL_TOP_K=20
RERANK_TOP_K=10
PARENT_CHUNK_SIZE=1200
PARENT_CHUNK_OVERLAP=200
CHILD_CHUNK_SIZE=256
CHILD_CHUNK_OVERLAP=64
KG_HOP_DEPTH=2
RRF_K=60

# LLM Config
LLM_MODEL=meta-llama/Llama-3.3-70B-Versatile
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.3

# Embedding
EMBEDDING_MODEL=ncbi/MedCPT-Article-Encoder
QUERY_MODEL=ncbi/MedCPT-Query-Encoder
CROSS_ENCODER_MODEL=ncbi/MedCPT-Cross-Encoder
```

---

## 11. Development Notes

- **Existing code**: `chatbot.py` contains early prototype code with MedCPT embedding + FAISS setup. This will be refactored into the modular structure.
- **Papers reference**: `papers/` directory contains reference papers (GraphRAG - Microsoft, MedRAG - Reasoning with KG).
- **HGT reference**: [Heterogeneous Graph Transformer (Hu et al., 2020)](https://arxiv.org/abs/2003.01332)
- **Offline vs Online**: HGT training + Contextual chunking are done OFFLINE (one-time). All inference components run ONLINE.
- **Cloud GPU Strategy**:
  - **Modal**: Used ONLY for Cross-Encoder reranking
  - **Groq API**: Used for Llama 70B (NER, query rewriting, answer generation)
  - **Kaggle GPU**: Used for offline contextual chunking with Gemini
- **Language**: English only for all data and queries.
- **Conversation**: Multi-turn support with PostgreSQL-backed conversation history.
- **KG Linearization**: Rule-based Python templates — no LLM overhead at inference time.
- **Entity Extraction**: LLM-based (Llama 70B via Groq) instead of BioBERT. Simpler pipeline, no separate NER model to maintain. No relation extraction needed since KG already has structured relationships.
