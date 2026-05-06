from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent   # .resolve: absolute path
    DATA_PATH: Path = BASE_DIR / "data"

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
    WEAVIATE_URL : str = "http://localhost:8080" # RESTful API port 8080
    WEAVIATE_GRPC_PORT : int = 50051             # Use GRPC protocol for big data processing     

    # RAG hyperparameters
    RETREVAL_TOP_K: int = 20
    RERANK_TOP_K: int = 10

    # Parent-Child chunking
    TIER1_MAX_LEN: int = 500            # Threshold of article's title + abstract length to be chunked or not
    TIER2_MAX_LEN: int = 2000           # Threshold of article's title + abstract length to be chunked or not
    PARENT_CHUNK_SIZE : int = 1500
    PARENT_CHUNK_OVERLAP : int = 256
    CHILD_CHUNK_SIZE : int = 500

    # Reranking configuration
    RERANK_TEXT_TOP_M: int = 5
    RERANK_KG_TOP_N: int = 20

    # RRF top_k configuration
    TOP_K_RRF: int = 60

    # PostgreSQL configuraion
    POSTGRE_HOST: str = "localhost"
    POSTGRE_PORT: int = 5432
    POSTGRE_USER: str = ""
    POSTGRE_PASSWORD: str = ""
    POSTGRE_DB: str = ""

    # Model names
    LLM_MODEL: str = "llama-3.1-8b-instant"
    EMBEDDING_MODEL: str = "ncbi/MedCPT-Article-Encoder"
    QUERY_MODEL: str = "ncbi/MedCPT-Query-Encoder"

    # Persistence
    SQLITE_PARENT_DB_PATH: Path = BASE_DIR / "vectorstore" / "parent_chunks.db"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"  # Allow extra variables like PYTHONPATH in .env
    )


settings = Settings()