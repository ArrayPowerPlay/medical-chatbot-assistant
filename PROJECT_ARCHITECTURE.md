# 📁 Project Architecture - Medical KG-RAG Chatbot

```
rag-project/
│
├── .env                              # API keys, DB URIs, Groq/Modal tokens
├── .env.example                      # Template env file
├── .gitignore
├── pyproject.toml                    # Project dependencies (uv)
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
├── scripts/
│   ├── ingest_documents.py           # Load BioASQ docs → chunk → contextual embedding → FAISS
│   ├── build_kg.py                   # PrimeKG → filter → populate Neo4j KG
│   ├── train_hgt.py                  # Train HGT model end-to-end (offline, one-time)
│   ├── evaluate_retrieval.py         # BioASQ Phase A retrieval evaluation (Recall@K)
│   ├── evaluate_generation.py        # BioASQ Phase B + MedQA generation eval (EM/F1, RAGAS)
│   └── seed_demo_data.py            # Optional: seed sample data for dev
│
├── src/
│   ├── __init__.py
│   │
│   ├── query/                        # Pre-retrieval query processing
│   │   ├── __init__.py
│   │   ├── query_rewriter.py         # Query rewriting via Groq (spell fix, specificity, history)
│   │   └── query_extractor.py        # LLM-based medical NER (Llama 70B via Groq, no RE)
│   │
│   ├── embeddings/                   # Embedding model wrappers (runtime)
│   │   ├── __init__.py
│   │   └── medcpt_embedder.py        # MedCPT dual encoder (Query-Encoder + Article-Encoder)
│   │
│   ├── dataset_builder/              # Offline data processing (batch, one-time)
│   │   ├── __init__.py
│   │   ├── preprocess_bioasq_taskA.py # Load BioASQ PubMed articles
│   │   ├── preprocess_bioasq_taskB.py # Preprocess Q&A for task B (test, val split)
│   │   ├── contextual_chunker.py     # Gemini 2.5 Flash/Pro contextual chunk enrichment
│   │   └── index_builder.py          # Build FAISS + Elasticsearch indexes
│   │
│   ├── retrieval/                    # 3 parallel retrieval streams
│   │   ├── __init__.py
│   │   ├── vector_search.py          # FAISS vector similarity search
│   │   ├── keyword_search.py         # Elasticsearch BM25 keyword search
│   │   ├── kg_search.py              # Neo4j 2-hop subgraph retrieval (HGT semantic embeddings)
│   │   ├── kg_linearization.py       # Rule-based subgraph → text (Python templates, no LLM)
│   │   └── parallel_retriever.py     # Orchestrate 3 parallel retrieval streams
│   │
│   ├── reranking/                    # Fusion + reranking
│   │   ├── __init__.py
│   │   ├── rrf.py                    # Reciprocal Rank Fusion (Vector + BM25 only → Text Retrieval)
│   │   └── cross_encoder.py          # MedCPT-Cross-Encoder (Modal GPU) — merges Text + KG
│   │
│   ├── generation/                   # Post-retrieval: prompt building + LLM generation
│   │   ├── __init__.py
│   │   ├── prompt_builder.py         # Head-tail placement prompt construction
│   │   └── llm_generator.py          # Llama 70B answer generation via Groq API
│   │
│   ├── kg/                           # Knowledge Graph infrastructure
│   │   ├── __init__.py
│   │   ├── neo4j_client.py           # Neo4j driver & query helpers
│   │   ├── schema.py                 # KG node/relationship type definitions (PrimeKG subset)
│   │   └── hgt/
│   │       ├── __init__.py
│   │       ├── model.py              # HGT model definition (arxiv:2003.01332)
│   │       ├── dataset.py            # PyG HeteroData loader from Neo4j
│   │       └── trainer.py            # End-to-end training for downstream tasks
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
│   │   └── health.py                # GET /health - health check
│   └── schemas/
│       ├── __init__.py
│       ├── request.py                # Pydantic request models
│       └── response.py              # Pydantic response models
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
│   └── hgt/                          # Saved HGT model checkpoints
│       └── .gitkeep
│
├── vectorstore/
│   └── faiss_index/                  # Persisted FAISS index files
│       └── .gitkeep
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
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   ├── 03_kg_visualization.ipynb
│   └── 04_pipeline_evaluation.ipynb
│
└── docker/
    ├── Dockerfile                    # Backend + Frontend container
    ├── docker-compose.yml            # App + Neo4j + Elasticsearch + PostgreSQL
    └── .dockerignore
```
