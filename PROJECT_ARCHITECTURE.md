# 📁 Project Architecture - Medical KG-RAG Chatbot

```
rag-project/
│
├── .env                              # API keys, DB URIs, Groq/Modal tokens
├── .env.example                      # Template env file
├── .gitignore
├── pyproject.toml                    # Project dependencies
├── README.md
├── OVERVIEW.md                       # Full project context for AI assistants
├── PROJECT_ARCHITECTURE.md           # This file
│
├── config/
│   ├── __init__.py
│   ├── settings.py                   # Pydantic Settings: load .env, constants
│   └── logging_config.py            # Logging configuration
│
├── data/
│   ├── raw/                          # BioASQ PubMed articles (200-300K)
│   ├── processed/                    # Cleaned & chunked documents
│   └── kg/                           # PrimeKG data (filtered: Disease, Symptom, Drug)
│       ├── nodes.csv                 # PrimeKG nodes (filtered)
│       └── relationships.csv         # PrimeKG edges (filtered)
│
├── results/
│   ├── eval_results/                 # Validation/dev evaluation outputs
│   │   ├── bioasq/
│   │   │   ├── retrieval/
│   │   │   └── generation/
│   │   └── medaesqa/
│   └── test_results/                 # Final frozen test outputs
│       ├── bioasq/
│       └── medaesqa/
│
├── scripts/
│   ├── ingest_documents.py           # Load BioASQ docs → parent-child chunk → MedCPT → Weaviate
│   ├── build_kg.py                   # PrimeKG → filter → Neo4j import + MedCPT node embeddings
│   ├── evaluation/
│   │   ├── bioasq/
│   │   │   ├── val_retrieval.py      # BioASQ validation retrieval evaluation
│   │   │   ├── test_retrieval.py     # BioASQ frozen test retrieval evaluation
│   │   │   ├── val_generation.py     # BioASQ validation generation evaluation
│   │   │   └── test_generation.py    # BioASQ frozen test generation evaluation
│   │   ├── medaesqa/                 # Future MedAESQA evaluation entrypoints
│   │   ├── shared/                   # Shared evaluation helpers for val/test
│   │   └── grid_search_retrieval.py  # Retrieval hyperparameter sweep
│   └── seed_demo_data.py            # Optional: seed sample data for dev
│
├── src/
│   ├── __init__.py
│   │
│   ├── storage/                      # NEW: Persistent storage layer
│   │   ├── __init__.py
│   │   ├── parent_store.py           # SQLite manager for parent chunks
│   │   ├── conversation_store.py     # PostgreSQL manager for chat history (multi-turn)
│   │   └── weaviate_client.py        # Weaviate client for child chunks (vector + BM25)
│   │
│   ├── query/                        # Pre-retrieval query processing
│   │   ├── __init__.py
│   │   └── query_analyzer.py         # Query rewriting + Medical NER + Intent extraction via Groq
│   │
│   ├── embeddings/                   # Embedding model wrappers (runtime)
│   │   ├── __init__.py
│   │   └── medcpt_embedder.py        # MedCPT dual encoder (Query-Encoder + Article-Encoder)
│   │
│   ├── dataset_builder/              # Offline data processing (batch, one-time)
│   │   ├── __init__.py
│   │   ├── preprocess_bioasq_taskA.py # Load BioASQ PubMed articles
│   │   ├── preprocess_bioasq_taskB.py # Preprocess Q&A for task B (test, val split)
│   │   ├── parent_child_chunker.py    # Adaptive 3-tier chunking using SciSpaCy
│   │
│   ├── retrieval/                    # 3 parallel retrieval streams
│   │   ├── __init__.py
│   │   ├── vector_search.py          # Weaviate vector search on Children -> map to Parents
│   │   ├── keyword_search.py         # Weaviate BM25 search on Children -> map to Parents
│   │   └── parallel_retriever.py     # Orchestrate 3 parallel streams (wire entities → Article-Encoder → KGSearch)
│   │
│   ├── reranking/                    # Fusion + reranking
│   │   ├── __init__.py
│   │   ├── rrf.py                    # Reciprocal Rank Fusion (Vector + BM25 only → Text Retrieval)
│   │   └── cross_encoder.py          # MedCPT-Cross-Encoder (Modal GPU) — Dynamic Quota Top-M/N, filters score < 0
│   │
│   ├── generation/                   # Post-retrieval: prompt building + LLM generation
│   │   ├── __init__.py
│   │   ├── kg_merger.py              # Post-rerank KG prefix merging (A->B->C, A->B->D)
│   │   ├── prompt_builder.py         # Head-tail placement prompt construction
│   │   └── llm_generator.py          # Llama 70B answer generation via Groq API
│   │
│   ├── kg/                           # Knowledge Graph infrastructure
│   │   ├── __init__.py
│   │   ├── neo4j_client.py           # Stage 1: medcpt_node_embeddings anchor search (A-E) + Stage 2: 2-hop Cypher (Q-E ranking)
│   │   ├── kg_search.py              # KG retrieval module (called by parallel_retriever)
│   │   ├── kg_linearization.py       # Path-based Linearization with Node Types (A -> B -> C)
│   │   └── schema.py                 # KG node/relationship type definitions (PrimeKG subset)
│   │
│   ├── pipeline/                     # End-to-end orchestration
│   │   ├── __init__.py
│   │   └── rag_pipeline.py           # Full pipeline: query → answer
│   │
│   └── utils/
│       ├── __init__.py
│       └── text_processing.py        # Text cleaning, normalization
│
├── modal_deployments/
│   └── cross_encoder_service.py      # Modal app: MedCPT-Cross-Encoder endpoint (only)
│
├── api/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app entry point (includes CORS config)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py                   # POST /chat - main Q&A endpoint
│   │   └── health.py                 # GET /health - health check
│   │   └── conversation.py           # POST /conversation - save conversation
│   └── schemas/
│       ├── __init__.py
│       ├── request.py                # Pydantic request models
│       └── response.py               # Pydantic response models
│
├── frontend/
│   ├── index.html                    # Main SPA entry
│   ├── css/
│   │   └── style.css                 # Global styles, design system
│   ├── js/
│   │   ├── app.js                    # Main application logic
│   │   ├── chat.js                   # Chat UI interactions & API calls
│   │   ├── markdown.js               # Markdown rendering for responses
│   │   └── theme.js                  # Dark/light theme toggle
│   └── assets/
│       ├── icons/                    # SVG icons
│       └── images/                   # Logo, illustrations
│
├── models/
│   └── .gitkeep
│
├── vectorstore/
│   └── parent_chunks.db              # SQLite database for original parent texts
│
├── tests/
│   ├── __init__.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   ├── test_reranking.py
│   ├── test_generation.py
│   ├── test_kg.py
│   └── test_api.py
│
├── notebooks/                        # Use for development testing
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   ├── 03_kg_visualization.ipynb
│   └── 04_pipeline_evaluation.ipynb
│
└── docker/
    ├── Dockerfile                    # Backend + Frontend container
    ├── docker-compose.yml            # App + Neo4j + Weaviate + PostgreSQL
    └── .dockerignore
```
