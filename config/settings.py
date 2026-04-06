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
    ELASTICSEARCH_URL: str = "http://localhost:9200"

    # RAG hyperparameters
    RETREVAL_TOP_K: int = 20
    RERANK_TOP_K: int = 10
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200

    # Model names
    LLM_MODEL: str = "meta-llama/Llama-3.3-70B-Versatile"
    EMBEDDING_MODEL: str = "ncbi/MedCPT-Article-Encoder"
    QUERY_MODEL: str = "ncbi/MedCPT-Query-Encoder"

    # Persistence
    @property
    def FAISS_INDEX_DIR(self) -> Path:
        return self.BASE_DIR / "vectorstore"

    @property
    def FAISS_INDEX_PATH(self) -> str:
        return str(self.FAISS_INDEX_DIR / "faiss_index")

    @property
    def FAISS_METADATA_PATH(self) -> str:
        return str(self.FAISS_INDEX_DIR / "faiss_metadata.jsonl")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()