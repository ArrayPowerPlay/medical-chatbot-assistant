# 🧠 OVERVIEW — MedKG-RAG Chatbot

> **MANDATORY RULE**: CẬP NHẬT FILE `CONTEXT.MD` NÀY MỖI KHI CÓ THAY ĐỔI VỀ KIẾN TRÚC/TÍNH NĂNG Ở CÁC FILE KHÁC (NẾU CẦN THIẾT). TUYỆT ĐỐI KHÔNG ĐƯỢC CODE NẾU CHƯA RÕ NỘI DUNG HOẶC YÊU CẦU. NẾU CÓ BẤT KỲ ĐIỀU GÌ CHƯA RÕ RÀNG HOẶC MƠ HỒ, PHẢI HỎI LẠI NGƯỜI DÙNG ĐỂ XÁC NHẬN TRƯỚC KHI BẮT ĐẦU CODE.
>
> **Purpose**: This file contains ALL essential context for the project. Reading this single file should be sufficient to fully understand the project's scope, architecture, and implementation details.

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
>
> The number of turns sent to the LLM is controlled by the server-side setting `HISTORY_TURNS_FOR_LLM` (default: **5 turns** = 10 messages). One **turn** = 1 user message + 1 assistant response. This setting is a global constant configured in `config/settings.py` and `.env`, applying uniformly to all users. It governs both **QueryAnalyzer** (query rewriting / entity extraction) and **LLMGenerator** (answer generation).

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

### 3.1 Document Corpus (for Weaviate Vector DB + BM25)
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
| Dataset | Split | File | Size | Notes |
|---|---|---|---|---|
| **BioASQ Task B** | **Validation** | `data/val/val_bioasq.jsonl` | ~277 questions | Hyperparameter tuning |
| **BioASQ Task B** | **Test** | `data/test/test_bioasq.jsonl` | ~283 questions | Final evaluation |

> **Snippet Coverage Filter**: Only questions where **every** relevant PMID has at least one corresponding gold snippet are retained. This ensures snippet-level evaluation is valid. Filtered by `_has_full_snippet_coverage()` in `preprocess_bioasq_taskB.py`.

#### Gold Data Schema (per question in JSONL)
```json
{
  "id": "question_id",
  "body": "Original question text",
  "relevant_pmid": ["12345", "67890"],
  "snippets": [{"text": "relevant passage...", "pmid": "12345"}],
  "ideal_answer": ["Gold reference answer"]
}
```

#### External Evaluation
| Dataset | Split | Size | Phase |
|---|---|---|---|
| MedAESQA | Validation | 12 questions | Generation |
| MedAESQA | Test | 28 questions | Generation |
| PubMedQA (`PQA-L`) | Optional | Labeled subset | Oracle-context generation |

> **MedAESQA policy**: used as a small external **generation benchmark**, not as the main retrieval benchmark.
> **PubMedQA policy**: if used, feed only the dataset-provided abstract context and disable retrieval to avoid source-article leakage.
> **MedAESQA corpus coverage policy**: it is acceptable to append all MedAESQA-referenced PMIDs into `data/corpus/corpus.jsonl` once for full-dataset coverage before evaluation. This is not treated as leakage because MedAESQA is not the main retrieval benchmark and the augmentation is done corpus-wide, not per-question at inference time.

### 3.4 Evaluation Metrics

#### Text Retrieval (`scripts/evaluation/bioasq/val_retrieval.py`, `scripts/evaluation/bioasq/test_retrieval.py`)

Pipeline evaluated: QueryAnalyzer (temp=0) → Vector + BM25 → RRF → Cross-Encoder.
Evaluation is **deterministic** (single run, temperature=0 for QueryAnalyzer).

> **Code layout**:
> - `scripts/evaluation/shared/retrieval_common.py` holds the shared BioASQ retrieval evaluation logic.
> - `scripts/evaluation/bioasq/val_retrieval.py` writes validation outputs.
> - `scripts/evaluation/bioasq/test_retrieval.py` mirrors the same evaluator for the frozen test split.

> **Best observed text-only eval config (May 16, 2026)**:
> `VECTOR_TOP_K=40`, `KEYWORD_TOP_K=80`, `TOP_K_RRF=80`, `K_RRF=60`, `CHILD_FETCH_LIMIT=120`, `RERANK_TEXT_TOP_M=20`.
> This is the strongest configuration tested so far and is now the default retrieval baseline.
>
> **Observed full validation metrics with the best text-only config**:
> - Document: `P@5=0.4912`, `R@5=0.6560`, `F1@5=0.4605`, `MAP@5=0.7200`, `GMAP@5=0.3655`
> - Document: `P@10=0.3578`, `R@10=0.7905`, `F1@10=0.4139`, `MAP@10=0.7211`, `GMAP@10=0.4656`
> - Document: `P@20=0.2240`, `R@20=0.8735`, `F1@20=0.3101`, `MAP@20=0.7336`, `GMAP@20=0.5241`, `MRR=0.8777`
> - Snippet: `Snippet_Recall@5=0.5982`, `Snippet_Precision@5=0.4608`, `Snippet_F1@5=0.4315`
> - Snippet: `Snippet_Recall@10=0.7208`, `Snippet_Precision@10=0.3370`, `Snippet_F1@10=0.3889`
> - Snippet: `Snippet_Recall@20=0.7955`, `Snippet_Precision@20=0.2137`, `Snippet_F1@20=0.2936`
>
> **Important implementation note**:
> `CrossEncoderReranker` must preserve negative logits. MedCPT Cross-Encoder returns raw scores, so filtering out passages with `score <= 0` incorrectly lowers `Recall@10/@20`, `MAP@20`, and snippet coverage.

##### Document-Level Metrics (PMID matching)
| Metric | K Values | Description |
|---|---|---|
| **Precision@K** | 5, 10, 20 | Fraction of retrieved PMIDs that are relevant |
| **Recall@K** | 5, 10, 20 | Fraction of gold PMIDs retrieved in top-K |
| **F1@K** | 5, 10, 20 | Harmonic mean of P@K and R@K |
| **MAP@K** | 5, 10, 20 | Mean Average Precision truncated at K |
| **GMAP@K** | 5, 10, 20 | Geometric Mean Average Precision truncated at K |
| **MRR** | — | Mean Reciprocal Rank of first relevant document |

> **MAP@K**: `MAP@K = mean(AP@K)` across all queries.
> `AP@K = (1/|gold|) * Σᵢ₌₁ᴷ P(i) × rel(i)` — considers ranking quality up to position K.
>
> **GMAP@K**: `GMAP@K = exp(mean(log(AP@K + 1e-6)))` across all queries.

##### Snippet-Level Metrics (text containment proxy)
| Metric | K Values | Description |
|---|---|---|
| **Snippet Recall@K** | 5, 10, 20 | Fraction of gold snippets whose text appears as a substring in a retrieved parent chunk with the same PMID |
| **Snippet Precision@K** | 5, 10, 20 | Fraction of top-K retrieved items that contain at least one gold snippet |

> **Matching Logic**: A gold snippet is "covered" if `gold_snippet_text in retrieved_parent_text` for a parent chunk sharing the same PMID.
>
> **Evaluation Policy (paper direction, current decision)**:
> - `Document retrieval` is reported in a single summary block using:
>   - `Precision@K`, `Recall@K`, `F1@K`, `MAP@K`, `GMAP@K`
>   - `MRR`
> - `@10` is the closest slice to BioASQ-style official-like document evaluation.
> - `Snippet retrieval` remains a **proxy metric** based on parent-text containment.
> - The repo does **not** currently implement official BioASQ snippet overlap scoring with character offsets.
> - The repo does **not** currently use child chunks or sentence windows as snippet prediction units.

> **Dataset provenance note**:
> `preprocess_bioasq_taskB.py` preserves snippet-to-PMID alignment, but the eval JSONL currently keeps only:
> - `snippet.text`
> - `snippet.pmid`
>
> Article `abstractText` is refetched from PubMed XML and serialized into the corpus, so any exact-match mismatch should be treated as a text-normalization / serialization issue, not as a PMID mismatch.

##### Output
| File | Path | Content |
|---|---|---|
| **Validation Detail** | `results/eval_results/bioasq/retrieval/detail.jsonl` | Per-question validation results |
| **Validation Summary** | `results/eval_results/bioasq/retrieval/summary.json` | Aggregate validation metrics |
| **Test Detail** | `results/test_results/bioasq/retrieval/detail.jsonl` | Per-question frozen test results |
| **Test Summary** | `results/test_results/bioasq/retrieval/summary.json` | Aggregate frozen test metrics |
| **Grid Search** | `results/eval_results/bioasq/retrieval/grid_search_20q_<timestamp>.json` | Full config + all document/snippet metrics for each retrieval hyperparameter run on a fixed 20-question subset |

> **Grid Search Utility**: `scripts/evaluation/grid_search_retrieval.py` runs a fixed 20-question retrieval benchmark over a preset grid of `(VECTOR_TOP_K, KEYWORD_TOP_K)` configurations while keeping the stronger baseline values for `CHILD_FETCH_LIMIT`, `TOP_K_RRF`, `K_RRF`, and `RERANK_TEXT_TOP_M` unless overridden via CLI.

#### Generation
- **ROUGE-SU4 F1** — primary metric for BioASQ (ideal answer quality)
- **RAGAS Context Precision**
- **RAGAS Context Recall**
- **RAGAS Faithfulness**
- **RAGAS Answer Correctness**
- **RAGAS Answer Relevancy** — validation/debug only

> **Generation improvements (June 2026)**:
> 1. **Type-conditional prompting**: `QueryAnalyzer` now classifies `question_type` (summary | list | yesno | factoid) as **TASK 4** in its existing single LLM call (no extra cost). `prompt_builder.py` uses this to inject a type-specific instruction into the system prompt.
> 2. **Preamble stripping**: Before computing ROUGE-SU4, `strip_preamble()` removes common LLM openers ("Based on the provided context...") that don't appear in gold references and hurt lexical overlap.
> 3. **Anti-preamble instruction**: System prompt now explicitly instructs the LLM to start answers directly without preamble phrases.

> **MedAESQA metric set (secondary dataset)**:
> `ROUGE-SU4-F1` + `Citation-Precision` + `Citation-Recall` + `Citation-F1`. RAGAS dropped (redundant for secondary dataset). `Citation-Coverage` and `Citation-Count` are also dropped as they are not informative/necessary.
> `Citation-F1` = harmonic mean of Citation-P and Citation-R (added June 2026).
> `use_citations=True` is hardcoded default for MedAESQA test entrypoint (citation quality is its primary signal).

> **MedAESQA evaluator policy**: use a **custom evaluator** for the project pipeline. `medaesqa_eval.py` is kept only as a dataset-reference script and is not the main evaluator for project results.

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
| **PostgreSQL** | Conversation history (multi-turn). DB: `chat_history`, persisted via Docker named volume `postgres_data`. | Docker container (local, `postgres:15`) |
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
psycopg2-binary           — PostgreSQL driver (conversation history)
ragas                     — RAG evaluation framework
```

### Generation Baselines / Settings (BioASQ End-to-End)
- **LLM-only** — no retrieval, direct answer generation
- **BM25-only + generator** — lexical retrieval baseline
- **Vector-only + generator** — dense retrieval baseline
- **Text-only hybrid RAG** — Vector + BM25 + RRF + Cross-Encoder, no KG
- **KG-only + generator** — graph evidence only
- **Full system** — Text retrieval + KG retrieval + reranking + interleaving

### Evaluation Code Structure
- **Runtime code** lives under `src/`
- **Evaluation code** lives under `scripts/evaluation/`
  - `shared/` — common code used by both validation and test evaluators
  - `bioasq/` — BioASQ split-specific entrypoints (`val_*`, `test_*`)
  - `medaesqa/` — MedAESQA split-specific entrypoints (`val_generation.py`, `test_generation.py`)
- **Dataset augmentation scripts** live under `src/dataset_builder/`
- **Outputs** are split between `results/eval_results/` and `results/test_results/`

---



## 7. API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Send user question, receive AI answer |
| `GET` | `/api/conversations` | List all conversations (most recent first) |
| `GET` | `/api/conversations/{id}/messages` | Paginated message history (cursor-based) |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/auth/register` | Register new user |
| `POST` | `/api/auth/login` | Login and receive JWT |
| `POST` | `/api/auth/guest` | Receive guest JWT |
| `PUT` | `/api/conversations/{id}` | Rename conversation |
| `DELETE` | `/api/conversations/{id}` | Delete conversation |
| `PUT` | `/api/conversations/{id}/pin` | Pin/Unpin conversation |
| `GET` | `/api/conversations/search` | Search conversations by title or message content |
| `POST` | `/api/conversations/{id}/messages/{msg_id}/feedback` | Submit like/dislike feedback and optional comment |
| `GET` | `/api/admin/stats` | Get system statistics (Admin only) |
| `GET` | `/api/admin/users` | List all users (Admin only) |
| `DELETE` | `/api/admin/users/{id}` | Delete a user (Admin only) |
| `PUT` | `/api/admin/users/{id}/password` | Reset user password (Admin only) |
| `GET` | `/api/admin/feedback/bad` | List disliked messages with comments (Admin only) |
| `GET` | `/api/admin/feedback/good` | List liked messages with comments (Admin only) |
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
> **Note**: This endpoint will be updated to support **Server-Sent Events (SSE)** via `StreamingResponse` to stream the answer progressively.

### GET /api/conversations/{id}/messages — Paginated Messages
> **Cursor-based pagination** for loading chat history. The frontend first loads the newest messages, then fetches older pages as the user scrolls up (reverse-chronological infinite scroll, like ChatGPT/Gemini).

**Query Parameters**:
| Param | Type | Default | Description |
|---|---|---|---|
| `limit` | int | `MESSAGE_PAGE_SIZE` (20) | Number of messages per page |
| `before_id` | int | `None` | Cursor: return messages older than this message ID |

**Response**:
```json
{
  "messages": [
    {"id": 42, "role": "user", "content": "What is metformin?", "created_at": "..."},
    {"id": 43, "role": "assistant", "content": "Metformin is...", "created_at": "..."}
  ],
  "has_more": true
}
```

### GET /api/conversations — Response
```json
{
  "conversations": [
    {"id": "uuid-1", "title": "What are symptoms of diabetes?", "created_at": "...", "updated_at": "..."},
    {"id": "uuid-2", "title": "Treatment for hypertension", "created_at": "...", "updated_at": "..."}
  ]
}
```

---

## 8. Frontend Stack & Architecture

- **Core**: React.js (Vite) + TypeScript
- **Styling**: Tailwind CSS v4 (Dark/Light mode via `dark:` classes)
- **State & API**: Zustand (Auth/Theme), Axios (JWT Interceptors)
- **Routing**: React Router (Protected routes for user/admin)
- **Key Features**: JWT Auth, Admin Dashboard, SSE Streaming (Typing effect), Reverse-scroll infinite chat history, Markdown rendering.

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

## 10. Environment Variables

> **Note**: For the full list of configuration parameters, refer to `config/settings.py` or `.env.example`. 

The system relies on `.env` for:
- Database connections (Neo4j, Weaviate, PostgreSQL)
- LLM and Embedding config (Model names, API keys)
- RAG Hyperparameters (Top-K limits, Chunk sizes, Context window sizes)
- JWT Auth (Algorithm, Expiration)

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
- **Generation Phase**: Completed — `kg_merger.py`, `prompt_builder.py`, `llm_generator.py` are implemented.
- **End-to-End Pipeline**: `rag_pipeline.py` orchestrates the full flow from query analysis to answer generation.
- **Pipeline Config Alignment**: As of May 16, 2026, `RAGPipeline.run()` forwards both `top_k` and `child_fetch_limit` into `ParallelRetriever`, so online inference can use the same retrieval-depth settings as `scripts/evaluation/bioasq/val_retrieval.py`.
- **Generation Tuning Scope**: once retrieval is frozen, generation tuning only touches prompt design, text/KG context budget, text/KG ratio, KG merger/interleaving, head-tail placement, abstain policy, citation style, `temperature`, and `max_tokens`. Retrieval knobs such as `VECTOR_TOP_K`, `KEYWORD_TOP_K`, `TOP_K_RRF`, `RERANK_TEXT_TOP_M`, and `RERANK_KG_TOP_N` must stay frozen during generation tuning.
- **Conversation History**: `ConversationStore` (PostgreSQL, `psycopg2`) persists multi-turn sessions. Data stored in Docker named volume `postgres_data` for durability. Auto-titles conversations with the first user question.
- **API Layer**: FastAPI app (`api/main.py`) with `/api/chat`, `/api/conversations`, `/api/conversations/{id}/messages`, and `/api/health` endpoints. Static frontend served at `/`.
- **LLM History Window**: Controlled by `HISTORY_TURNS_FOR_LLM` (default 5 turns = 10 messages). This is a global server-side constant, not per-user. Applied uniformly in `QueryAnalyzer` and `LLMGenerator`.
- **Chat Pagination**: `MESSAGE_PAGE_SIZE` (default 20) controls the number of messages loaded per scroll batch in the frontend. Cursor-based pagination using message `id` as cursor.
- **Frontend Status**: Frontend foundation is built (React + Vite + Tailwind v4 + Zustand). Authentication UI (Login/Register) is implemented.
- **User Authentication**: Fully implemented with JWT. Supports three roles: `user`, `guest` (limited questions), and `admin`.
- **Admin Dashboard**: Backend APIs are implemented for managing users, viewing system stats, and auditing good/bad feedback.

### Retrieval Implementation Notes

- `K_RRF` and `TOP_K_RRF` are separate knobs: `K_RRF` is the damping constant in the RRF formula, while `TOP_K_RRF` is the number of fused text candidates kept before cross-encoder reranking.
- `CrossEncoderReranker` keeps negative MedCPT logits and ranks all passages by score instead of dropping `score <= 0`. This preserves valid relative ordering from the classifier output.
- `RAGPipeline.run()` keeps the rewritten query embedding as a NumPy-like vector for Weaviate vector search and separately converts it to a Python list only where KG retrieval needs JSON-serializable data.
- `RAGPipeline._normalize_history()` now truncates the already-cleaned history, so invalid or empty messages are not reintroduced by slicing.
- `src.pipeline.rag_pipeline` imports `KGSearch` from `src.kg.kg_search`, so the module imports cleanly from the project root.

## 12. Clean Architecture & Ablation Study (July 2026)

**Layered Architecture (Dependency Inversion)**:
- **Presentation (API)**: FastAPI routes parsing requests.
- **Use Case**: `RAGPipeline` (Orchestrator).
- **Domain**: `src/interfaces/` defining contracts (`ISearchEngine`, `IKGSearcher`, `ILLMGenerator`, `IQueryAnalyzer`). NO data access logic exists here.
- **Infrastructure**: Data access classes (`AsyncWeaviateChildStore`, `Neo4jClient`, `GroqGenerator`) implementing the domain interfaces.
- **Async Execution**: The pipeline leverages `asyncio.gather()` for fully non-blocking I/O during retrieval.

**Ablation Study Strategy (Single Branch)**:
- **Code Separation**: Dedicated entrypoint scripts in `scripts/evaluation/bioasq/<version>/test_*.py` configure `RunConfig` statically for each variant.
- **Result Routing**: The output of each ablation variant is routed to `results/test_results/bioasq/<version>/` instead of the root folder, keeping metrics separate.
- **Baselines**: The primary baseline is **Vector Search Only + LLM Generator**. The intermediate baseline is **Text-only Hybrid RAG (No KG)**.
