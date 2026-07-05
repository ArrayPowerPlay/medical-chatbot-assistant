from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import List

class Settings(BaseSettings):
    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent   # .resolve: absolute path
    DATA_PATH: Path = BASE_DIR / "data"
    RESULTS_PATH: Path = BASE_DIR / "results"
    EVAL_RESULTS_PATH: Path = RESULTS_PATH / "eval_results"
    TEST_RESULTS_PATH: Path = RESULTS_PATH / "test_results"

    # API Keys
    GROQ_API_KEY: str = Field(default="")
    MODAL_TOKEN_ID: str | None = None
    MODAL_TOKEN_SECRET: str | None = None
    HF_TOKEN: str | None = Field(default=None)
    GOOGLE_API_KEY: str = Field(default="")

    # Database URLs
    NEO4J_URL: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""

    # Weaviate
    WEAVIATE_URL: str = "http://localhost:8081" # RESTful API port 8081
    WEAVIATE_GRPC_PORT: int = 50051             # Use GRPC protocol for big data processing     

    # RAG hyperparameters
    VECTOR_TOP_K: int = 40
    KEYWORD_TOP_K: int = 80
    CHILD_FETCH_LIMIT: int = 120        # Number of child chunks to be fetched

    # Parent-Child chunking
    TIER1_MAX_LEN: int = 500            # Threshold of article's title + abstract length to be chunked or not
    TIER2_MAX_LEN: int = 2000           # Threshold of article's title + abstract length to be chunked or not
    PARENT_CHUNK_SIZE: int = 1500
    PARENT_CHUNK_OVERLAP: int = 256
    CHILD_CHUNK_SIZE: int = 500

    # Reranking configuration
    RERANK_TEXT_TOP_M: int = 20
    RERANK_KG_TOP_N: int = 20

    # RRF configuration
    TOP_K_RRF: int = 80
    K_RRF: int = 60

    # K values used for evaluation
    K_VALUES: List[int] = [5, 10, 20]

    # PostgreSQL configuraion
    POSTGRE_HOST: str = "localhost"
    POSTGRE_PORT: int = 5432
    POSTGRE_USER: str = ""
    POSTGRE_PASSWORD: str = ""
    POSTGRE_DB: str = ""

    # Model names
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL: str = "ncbi/MedCPT-Article-Encoder"
    QUERY_MODEL: str = "ncbi/MedCPT-Query-Encoder"
    RAGAS_EVALUATOR_LLM_MODEL: str = "gpt-4o-mini"        # Judge model used by RAGAS to score answer quality and context usage
    RAGAS_EVALUATOR_EMBEDDING_MODEL: str = "text-embedding-3-small"  # Embedding model used by RAGAS for semantic evaluation steps

    # Generation defaults
    GENERATION_TEMPERATURE: float = 0.0   # Default sampling temperature for answer generation; keep low for factual QA
    GENERATION_MAX_TOKENS: int = 512      # Default maximum number of tokens the generator may produce for one answer
    USE_KG_MERGER: bool = True            # Whether KG paths with the same prefix should be merged before prompt construction
    USE_HEAD_TAIL_PLACEMENT: bool = True  # Whether retrieved context should be reordered with head-tail placement before generation
    USE_CITATIONS: bool = False           # Whether the generator should be instructed to cite sources using PMIDs
    KG_TOP_K: int = 2                     # Maximum number of anchor nodes retrieved per extracted entity during Stage 1 anchor search
    KG_HOP1_M: int = 3                    # Maximum number of 1-hop neighbours kept for each anchor node during Stage 2a expansion
    KG_HOP2_N: int = 3                    # Maximum number of 2-hop neighbours kept for each 1-hop node during Stage 2b expansion
    KG_HOP2_CAP: int = 30                 # Global hard cap on the total number of 2-hop triples returned after KG expansion
    RERANK_KG_TOP_N: int = 10             # Number of KG paths kept after cross-encoder reranking

    # LLM History Configuration
    HISTORY_TURNS_FOR_LLM: int = 5    # 1 turn = user + assistant. 5 turns = 10 messages
    # Chat History Pagination
    MESSAGE_PAGE_SIZE: int = 20        # Messages per page for cursor-based pagination

    # Persistence
    SQLITE_PARENT_DB_PATH: Path = BASE_DIR / "vectorstore" / "parent_chunks.db"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"  # Allow extra variables like PYTHONPATH in .env
    )


settings = Settings()
