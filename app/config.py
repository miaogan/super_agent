import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-ada-002"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # RAG
    chunk_size: int = 1024
    chunk_overlap: int = 200
    similarity_top_k: int = 3

    # Paths
    data_dir: str = "data"
    index_dir: str = "data/index"

    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache
def get_settings() -> Settings:
    return Settings()
