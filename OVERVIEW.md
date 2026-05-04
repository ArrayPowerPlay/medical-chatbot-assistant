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
- **Package Manager**: conda
- **Backend Framework**: FastAPI

---

## 2. Pipeline Architecture

The system has **two main phases**: **Retrieval** and **Generation**.

### 2.1 Retrieval Phase — 3 Parallel Streams + Stage-based Reranking

> **IMPORTANT: Retrieval Flow**
> 1. Run 3 retrieval streams in **parallel**: Vector Search, Keyword Search, and KG Search.
> 2. **RRF (Reciprocal Rank Fusion)** merges ONLY **Vector Search + BM25** → produces **Text Search** results.
> 3. **Cross-Encoder** (MedCPT-Cross-Encoder) reranks **Text Search** (from RRF) and **KG Search** (linearized subgraphs) independently to prevent out-of-distribution (OOD) score bias against structured KG paths.
> 4. **Post-Rerank Interleaving**: The Top-M text and Top-N KG results are merged via 1-by-1 interleaving before final Head-Tail placement.

> **Implementation Note**: The core RAG orchestration remains synchronous. `ParallelRetriever` uses a thread pool to run the three retrieval streams in parallel, and FastAPI can wrap the pipeline in a threadpool later without changing the retrieval or generation clients.

#### Stream 1: Vector Search (Semantic)
| Setting | Value |
|---|---|
| **Vector DB** | Weaviate (Local Docker, Cosine Similarity) |
| **Embedding Model** | `ncbi/MedCPT-Article-Encoder` (for documents) |
| **Query Model** | `ncbi/MedCPT-Query-Encoder` (for queries) |
| **Chunking** | **Adaptive 3-Tier Chunking** (SciSpaCy, no LLM) |
| **Parent Size** | Tier 1/2: Full abstract. Tier 3: 1500 chars (256 overlap) |
| **Child Size** | Tier 1: Full. Tier 2/3: ~500 chars (with Title Injection) |
| **Contextual Strategy**| Search is performed on **Child Chunks**; context for LLM is retrieved from the corresponding **Parent Chunk**. |
| **Metadata Mapping** | A mapping between Child `parent_id` and Parent text is stored in **SQLite** (`parent_chunks.db`) for O(log n) lookup. |

> **Key Detail — Adaptive Chunking**: Uses `SciSpaCy` to safely split medical sentences without breaking acronyms.
> - **Tier 1** (<=500 chars): Parent = Child = Full Article.
> - **Tier 2** (<=2000 chars): Parent = Full Article. Child = ~500 chars (Title Injected).
> - **Tier 3** (>2000 chars): Parent = 1500 chars (overlap 256). Child = ~500 chars (Title Injected).

> **Key Detail — Dual Encoder Setup**: MedCPT uses an **asymmetric** architecture. `MedCPT-Query-Encoder` encodes the user query, while `MedCPT-Article-Encoder` encodes document chunks.

> **Key Detail — Parent-Child Chunking Strategy**: Small "Child" chunks provide high precision for vector/keyword matching, while their larger "Parent" chunks provide the full semantic context needed by the LLM to generate accurate answers.

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
| **Node Embedding** | `ncbi/MedCPT-Article-Encoder` on enriched text `"{NodeType}: {name}"` for every Disease, Drug, GeneProtein, EffectPhenotype node |
| **Graph ML** | None |
| **Entity Extraction** | Llama 3.3 70B via Groq API |
| **Linearization** | Rule-based Python templates |

> **KG Query Flow (Inference)**:
> 1. User query → **LLM-based NER + Intent Classification** (Llama 70B via Groq, single API call)
>    - Extracts entity strings: `entities = ["Type 2 Diabetes", "Metformin", ...]`
>    - Classifies query **Intent** into one of 8 categories: `symptom_lookup`, `treatment_lookup`,
>      `mechanism_lookup`, `side_effect_lookup`, `contraindication_lookup`, `disease_relation`,
>      `genetic_association`, `drug_target_lookup`, `general`
> 2. Encode **rewritten query** with **MedCPT-Query-Encoder** → `rewritten_query_vec`
>    - Shared with Weaviate vector search stream
>    - Used for **Stage 2** neighbour ranking in KG (Q-E vs A-E cross-space, MedCPT asymmetric design)
> 3. Encode **each entity string** with **MedCPT-Article-Encoder** → `entity_article_embeddings`
>    - Same encoder space as stored node embeddings → reliable same-space cosine comparison
>    - Used for **Stage 1** anchor search only
> 4. **Stage 1 — Anchor search** (same-space: A-E vs A-E):
>    - For each `entity_article_emb`, query Neo4j vector index `medcpt_node_embeddings` → top-k=3 anchors
>    - Union and deduplicate anchors across all entities
> 5. **Stage 2a — 1-hop traversal** from anchors (cross-space: Q-E vs A-E):
>    - Filter: only edge types allowed by Intent (from `INTENT_EDGE_FILTER` in schema.py)
>    - Rank: `cosine_sim(rewritten_query_vec, neighbour.embedding_medcpt)`
>    - Keep: top-M=10 per anchor node
> 6. **Stage 2b — 2-hop traversal** from filtered 1-hop nodes (cross-space: Q-E vs A-E):
>    - Filter: intent-allowed edge types
>    - Rank: `cosine_sim(rewritten_query_vec, neighbour.embedding_medcpt)`
>    - Keep: top-N=5 per 1-hop node — hard cap 50 triples total
> 7. **Rule-based linearization**: Python templates convert triples → natural language text.
>    - *Dead-end Optimization*: 1-hop paths are only emitted if they do not extend into 2-hop paths, preventing duplicate prefixes from consuming Cross-Encoder top-N slots.

> **One embedding vector per node stored in Neo4j**:
> - `embedding_medcpt` — `Article-Encoder("{NodeType}: {name}")`, stored offline by `build_kg.py`.
>   Used for **both** Stage 1 index (`medcpt_node_embeddings`) and Stage 2 neighbour ranking.

> **Two-Encoder Inference Strategy**:
> - **Stage 1**: `Article-Encoder(entity)` enables reliable same-space lookup against the node index.
> - **Stage 2**: `Query-Encoder(rewritten_query)` ranks neighbours by relevance to full question intent (cross-space comparison).

> **Node text enrichment** (applied in `build_kg.py`):
> Bare entity names produce underspecified embeddings. Prepending the node type grounds the
> embedding in the right semantic region:
> - L1 (current): `"Disease: Type 2 Diabetes"`, `"Drug: Metformin"`, etc.
> - L2 (future): `"Disease: Type 2 Diabetes. {node_definition}"` if PrimeKG description is available

> **KG Schema** (node types — filtered from PrimeKG):
> - `Disease`, `Drug`, `GeneProtein`, `EffectPhenotype` (covers Symptom/Side-effect)
> - All nodes also carry the secondary label `:KGNode` (enables a single vector index spanning all types)
> - Other PrimeKG node types (anatomy, pathway, biological_process, exposure, etc.) are excluded

> **KG Relationships** (final subset from PrimeKG, mapped to Neo4j labels):
>
> | PrimeKG `display_relation` | Neo4j label | Edge | Count in kg.csv |
> |---|---|---|---|
> | `indication` | `TREATS` | Drug → Disease | 18,776 |
> | `contraindication` | `CONTRAINDICATES` | Drug → Disease | 61,350 |
> | `off-label use` | `OFF_LABEL_USE` | Drug → Disease | 5,136 |
> | `target` | `TARGETS` | Drug → GeneProtein | 32,760 |
> | `enzyme` | `METABOLIZED_BY` | Drug → GeneProtein | 10,634 |
> | `transporter` | `TRANSPORTED_BY` | Drug → GeneProtein | 6,184 |
> | `carrier` | `CARRIED_BY` | Drug → GeneProtein | 1,728 |
> | `side effect` | `HAS_SIDE_EFFECT` | Drug → EffectPhenotype | 129,568 |
> | `phenotype present` | `PRESENTS` | Disease → EffectPhenotype | 300,634 |
> | `phenotype absent` | `PHENOTYPE_ABSENT` | Disease → EffectPhenotype | 2,386 |
> | `associated with` | `ASSOCIATED_WITH` | GeneProtein → Disease | 167,482 |
> | `parent-child` | `SUBTYPE_OF` | Disease → Disease | ~subset of 281,744 |
>
> **Excluded relations**: `ppi`, `synergistic interaction`, `expression present/absent`, `comorbidity`, `interacts with`, `linked to` (Out of scope).


> **Node Embedding Pipeline** (offline, one-time, `build_kg.py`):
> 1. For each node, build enriched text: `"{NodeType}: {name}"` (L1 enrichment)
> 2. Batch encode with **MedCPT-Article-Encoder** → 768-dim L2-normalised vectors
> 3. Store as `embedding_medcpt` property on each Neo4j node
> 4. Create vector index `medcpt_node_embeddings` (cosine, 768 dims) on `:KGNode` label
> Fully inductive — new nodes can be embedded and added without rebuilding the whole graph.


> **Subgraph Linearization (Path-based with Node Types)**:
> After retrieving a 2-hop subgraph from Neo4j, **Python templates** in `src/kg/kg_linearization.py`
> convert graph triples into independent path sentences. Examples:
> - 1-hop: `"[Drug] [X] TREATS [Disease] [Y]"`
> - 2-hop: `"[Drug] [X] TARGETS [GeneProtein] [P] which is ASSOCIATED_WITH [Disease] [Y]"`
> This Path-based approach explicitly preserves multi-hop reasoning. The paths are pooled with Text Retrieval and jointly reranked by the Cross-Encoder.



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
| **Input** | (query, passage) pairs from BOTH Text Retrieval (RRF output) AND KG Retrieval (independent path sentences) |
| **Function** | Reranks Text and KG passages in a single API batch for efficiency, but **separates and sorts them independently** to prevent Text from unfairly dominating KG paths (OOD bias). |
| **Output** | Two separate lists: Top-M ranked Text and Top-N ranked KG paths. |

> **Normalization Detail**: Text passages are normalized with `source_type="text_retrieval"`, `parent_id`, `pmid`, etc. KG passages retain their `metadata` dictionary and are tagged with `source_type="kg_retrieval"`.

### 2.3 Generation Phase

#### Query Analyzer (Pre-Retrieval)
- **Model**: Llama 3.3 70B via **Groq API**
- **Purpose**:
  - Fix spelling errors and rewrite query for retrieval
  - Connect with conversation history (multi-turn context)
  - Extract entities (Disease, EffectPhenotype, Drug, GeneProtein) and intent
- Runs as a single LLM call BEFORE the 3 parallel retrieval streams

> **History Handling**: Conversation history is persisted separately from the prompt window. Only the latest turns are injected into query analysis and answer generation, which keeps prompts bounded while preserving long-term conversation state in storage.

#### Post-Rerank KG Merging & Interleaving (Prompt Prep)
- **KG Merging**: Groups paths by `(prefix, rel2)` metadata. Merges `A targets B associated with C` and `D` into `A targets B which is associated with C, and D.`
- **Density Bonus**: `Agg_Score = MAX(scores) + 0.05 * (N - 1)`.
- **Manual Interleaving**: Since Text and KG are sorted independently, they are interleaved 1-by-1 (`Text 1, KG 1, Text 2, KG 2`) to ensure both modalities get equal placement opportunity in the final prompt.

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
| **Parent-Child Chunking** | Adaptive 3-Tier (SciSpaCy) | Local CPU | Offline (batch, one-time) |
| **KG Node Embedding** | MedCPT-Article-Encoder | Local CPU/GPU | Offline (one-time, inductive) |
| **Query Analysis (Rewrite + NER)** | Llama 3.3 70B Versatile | Groq API | Inference (per query) |
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
conda install weaviate-client             — Weaviate Python V4 driver
langchain, langchain-community — Document loading, text splitting
neo4j                     — Neo4j Python driver
torch, transformers       — MedCPT models (Article-Encoder, Query-Encoder, Cross-Encoder)
modal                     — Cloud GPU deployment (Cross-Encoder)
groq                      — Groq API client (Llama 70B)
sqlite3                   — Built-in Python DB (Parent storage)
pydantic, pydantic-settings — Config & validation
ragas                     — RAG evaluation framework
```

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
    ├─── Query Analyzer (Llama 70B via Groq)
    │    ├── Rewrite query for retrieval
    │    └── Extract: disease, symptom, drug entities & intent
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
    │    └── KG Search (Neo4j)                        │
    │                   │                             │
    │                   └── Path-based Linearization  │
    │                        (A -> B -> C paths)      │
    │                               │                 │
    │                               └────────┬────────┘
    │                                        ▼
    ├─── Cross-Encoder Reranking (MedCPT, Modal GPU)
    │    ├── Batch inference for all Text + KG paths
    │    └── Separate into 2 independently sorted lists (Top-M Text, Top-N KG)
    │
    ├─── Post-Rerank KG Merging ──► Condense Top-N KG prefixes + Density Bonus
    │
    ├─── Manual Interleaving ──► Trộn 1-by-1 (Text 1, KG 1, Text 2, KG 2)
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

- **Data Ingestion Pipeline**: Uses Producer-Consumer architecture (Streaming Batch Processing) via `threading.Thread` and `queue.Queue` to separate Chunking, Embedding, and DB Storage into independent streams.
- **Papers reference**: `papers/` directory contains reference papers.
- **Two-vector KG inference**: `entity_article_embeddings` for Stage 1 anchor lookup; `rewritten_query_vec` for Stage 2 neighbour ranking.
- **Offline vs Online**: Node embedding (`build_kg.py`) is done OFFLINE. Inference components run ONLINE.
- **Cloud GPU Strategy**: Modal for Cross-Encoder reranking, Groq API for Llama 70B inference.
- **Language**: English only for all data and queries.
- **Conversation**: Multi-turn support with PostgreSQL.
- **KG Linearization**: Rule-based Python templates.
- **Entity Extraction**: Handled by `query_analyzer.py` via Llama 70B.
