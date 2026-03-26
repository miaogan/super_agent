import os
from pydantic_settings import BaseSettings
from functools import lru_cache


def get_project_root() -> str:
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    # Paths - 使用绝对路径
    data_dir: str = ""
    index_dir: str = ""

    class Config:
        env_file = ".env"
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 设置默认绝对路径
        project_root = get_project_root()
        if not self.data_dir:
            self.data_dir = os.path.join(project_root, "data")
        if not self.index_dir:
            self.index_dir = os.path.join(project_root, "data", "index")


@lru_cache
def get_settings() -> Settings:
    return Settings()
