from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from app.config import get_settings


class Retriever:
    """检索器"""

    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.settings = get_settings()

    def get_retriever(self):
        """获取检索器实例"""
        return VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.settings.similarity_top_k,
        )

    def get_query_engine(self):
        """获取查询引擎"""
        retriever = self.get_retriever()
        return RetrieverQueryEngine.from_args(retriever)

    def retrieve(self, query: str):
        """检索相关文档"""
        retriever = self.get_retriever()
        return retriever.retrieve(query)
