from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

from rag.retriever import Retriever
from app.config import get_settings


class RAGChain:
    """RAG 对话链 - 结合 LangChain 和 LlamaIndex"""

    def __init__(self, retriever: Retriever):
        self.settings = get_settings()
        self.retriever = retriever

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model_name=self.settings.openai_model,
            openai_api_key=self.settings.openai_api_key,
            openai_api_base=self.settings.openai_base_url,
            temperature=0.7,
        )

        # 对话记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
        )

        # 构建 RAG 链
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever.get_retriever(),
            memory=self.memory,
            return_source_documents=True,
        )

    def chat(self, query: str) -> Dict[str, Any]:
        """对话"""
        result = self.chain.invoke({"question": query})
        return {
            "answer": result["answer"],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in result.get("source_documents", [])
            ],
        }

    def reset_memory(self):
        """重置对话记忆"""
        self.memory.clear()

    def get_chat_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        messages = self.memory.chat_memory.messages
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "ai", "content": msg.content})
        return history
