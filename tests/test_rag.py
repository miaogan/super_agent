import pytest
from unittest.mock import Mock, patch


def test_config_loads():
    """测试配置加载"""
    from app.config import Settings

    settings = Settings(
        openai_api_key="test-key",
        chunk_size=512,
    )
    assert settings.openai_api_key == "test-key"
    assert settings.chunk_size == 512


def test_document_loader():
    """测试文档加载器"""
    from rag.loader import DocumentLoader
    from llama_index.core import Document

    # 测试从文本创建文档
    doc = DocumentLoader.load_text("Hello world", {"source": "test"})
    assert doc.text == "Hello world"
    assert doc.metadata["source"] == "test"


def test_retriever():
    """测试检索器"""
    from rag.retriever import Retriever

    mock_index = Mock()
    retriever = Retriever(mock_index)

    assert retriever.index == mock_index


def test_rag_chain_initialization():
    """测试 RAG 链初始化"""
    from rag.chain import RAGChain
    from rag.retriever import Retriever

    mock_retriever = Mock(spec=Retriever)
    mock_retriever.get_retriever.return_value = Mock()

    with patch('rag.chain.ChatOpenAI'):
        chain = RAGChain(mock_retriever)
        assert chain.retriever == mock_retriever
