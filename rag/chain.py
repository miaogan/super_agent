from typing import List, Dict, Any
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager

from app.config import get_settings


# 自定义 QA 提示模板
QA_TEMPLATE = """<context>
已知信息如下：
{context_str}

</context>
问题: {query_str}

根据已知信息回答问题。如果无法从已知信息中找到答案，请如实说明。"""


class RAGChain:
    """RAG 对话链 - 使用 LlamaIndex 原生查询引擎"""

    def __init__(self, retriever):
        self.settings = get_settings()
        self.retriever = retriever
        self.query_engine = retriever.get_query_engine()

        # 设置提示模板
        self.qa_template = PromptTemplate(QA_TEMPLATE)

    def chat(self, query: str) -> Dict[str, Any]:
        """对话"""
        # 使用查询引擎
        response = self.query_engine.query(query)
        
        # 获取源文档
        source_docs = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                source_docs.append({
                    "content": node.node.text if hasattr(node.node, 'text') else str(node.node),
                    "metadata": node.node.metadata if hasattr(node.node, 'metadata') else {}
                })

        return {
            "answer": str(response),
            "source_documents": source_docs,
        }

    def reset_memory(self):
        """重置对话（当前实现使用原生引擎，无需重置）"""
        pass

    def get_chat_history(self) -> List[Dict[str, str]]:
        """获取对话历史（当前实现不支持）"""
        return []
