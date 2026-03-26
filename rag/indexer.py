import os
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

from app.config import get_settings


class Indexer:
    """索引构建器"""

    def __init__(self):
        self.settings = get_settings()
        self.embed_model = OpenAIEmbedding(
            model=self.settings.embedding_model,
            api_key=self.settings.openai_api_key,
            api_base=self.settings.openai_base_url,
        )
        self.node_parser = SentenceSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def build_index(self, documents: list[Document]) -> VectorStoreIndex:
        """构建向量索引"""
        # 解析文档为节点
        nodes = self.node_parser.get_nodes_from_documents(documents)

        # 创建索引
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=self.embed_model,
        )
        return index

    def save_index(self, index: VectorStoreIndex, persist_dir: str = None):
        """保存索引到磁盘"""
        persist_dir = persist_dir or self.settings.index_dir
        os.makedirs(persist_dir, exist_ok=True)

        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir
        )
        index.storage_context.json persist_dir

    def load_index(self, persist_dir: str = None) -> VectorStoreIndex:
        """从磁盘加载索引"""
        persist_dir = persist_dir or self.settings.index_dir

        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Index not found at {persist_dir}")

        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir
        )
        return load_index_from_storage(storage_context)


indexer = Indexer()
