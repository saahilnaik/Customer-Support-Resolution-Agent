from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    llm_retry_attempts: int = 4
    llm_retry_max_wait: float = 20.0

    database_url: str = "sqlite:///./data/support.db"
    checkpoint_db: str = "./data/checkpoints.db"

    external_api_base_url: str = "http://127.0.0.1:8000/mock"
    external_retry_attempts: int = 3
    external_backoff: float = 0.3  # seconds, multiplied exponentially

    refund_approval_threshold: float = 50.0

    embedding_backend: str = "fastembed"  # fastembed | tfidf
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    kb_dir: str = "data/kb"
    rag_top_k: int = 3

    graph_recursion_limit: int = 30
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
