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
    PARENT_CHUNK_SIZE : int = 1200
    PARENT_CHUNK_OVERLAP : int = 200
    CHILD_CHUNK_SIZE : int = 256
    CHILD_CHUNK_OVERLAP : int = 64

    # Model names
    LLM_MODEL: str = "meta-llama/Llama-3.3-70B-Versatile"
    EMBEDDING_MODEL: str = "ncbi/MedCPT-Article-Encoder"
    QUERY_MODEL: str = "ncbi/MedCPT-Query-Encoder"

    # Persistence
    SQLITE_PARENT_DB_PATH: str = "./vectorstore/parent_chunks.db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()