import os
import json
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel

from rag.loader import loader
from rag.indexer import indexer
from rag.retriever import Retriever
from rag.chain import RAGChain
from app.config import get_settings


router = APIRouter()

# 全局变量存储索引和链
_index = None
_rag_chain = None


def get_index():
    """获取或初始化索引"""
    global _index
    settings = get_settings()

    try:
        _index = indexer.load_index()
    except FileNotFoundError:
        # 如果索引不存在，加载文档后构建
        documents = loader.load_directory(settings.data_dir)
        if documents:
            _index = indexer.build_index(documents)
            indexer.save_index(_index)
        else:
            _index = None

    return _index


def get_rag_chain() -> RAGChain:
    """获取 RAG 链"""
    global _rag_chain

    idx = get_index()
    if idx is None:
        raise HTTPException(
            status_code=400,
            detail="No index available. Please add documents first."
        )

    if _rag_chain is None:
        retriever = Retriever(idx)
        _rag_chain = RAGChain(retriever)

    return _rag_chain


# ============ 请求/响应模型 ============

class ChatRequest(BaseModel):
    query: str
    reset_memory: bool = False


class ChatResponse(BaseModel):
    answer: str
    source_documents: List[dict]


class DocumentInfo(BaseModel):
    file_name: str
    doc_id: str


# ============ API 路由 ============

@router.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "message": "Super Agent RAG API"}


@router.post("/documents", response_model=List[DocumentInfo])
async def upload_documents(files: List[UploadFile] = File(...)):
    """上传文档到知识库"""
    settings = get_settings()
    data_dir = settings.data_dir
    os.makedirs(data_dir, exist_ok=True)

    saved_files = []

    for file in files:
        # 保存文件
        file_path = os.path.join(data_dir, file.filename)
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        saved_files.append(DocumentInfo(
            file_name=file.filename,
            doc_id=file.filename
        ))

    # 重新构建索引
    global _index, _rag_chain
    documents = loader.load_directory(data_dir)
    if documents:
        _index = indexer.build_index(documents)
        indexer.save_index(_index)
        _rag_chain = None  # 重置链

    return saved_files


@router.get("/documents")
async def list_documents():
    """列出知识库中的文档"""
    settings = get_settings()
    data_dir = settings.data_dir

    if not os.path.exists(data_dir):
        return []

    files = os.listdir(data_dir)
    return [
        {"file_name": f, "type": "file"}
        for f in files
        if os.path.isfile(os.path.join(data_dir, f))
    ]


@router.delete("/documents/{file_name}")
async def delete_document(file_name: str):
    """删除文档"""
    settings = get_settings()
    file_path = os.path.join(settings.data_dir, file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    os.remove(file_path)

    # 重新构建索引
    global _index, _rag_chain
    documents = loader.load_directory(settings.data_dir)
    if documents:
        _index = indexer.build_index(documents)
        indexer.save_index(_index)
        _rag_chain = None

    return {"message": f"Deleted {file_name}"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """问答"""
    rag_chain = get_rag_chain()

    if request.reset_memory:
        rag_chain.reset_memory()

    result = rag_chain.chat(request.query)

    return ChatResponse(
        answer=result["answer"],
        source_documents=result["source_documents"]
    )


@router.get("/chat/history")
async def get_chat_history():
    """获取对话历史"""
    rag_chain = get_rag_chain()
    return {"history": rag_chain.get_chat_history()}


@router.post("/chat/reset")
async def reset_chat():
    """重置对话"""
    rag_chain = get_rag_chain()
    rag_chain.reset_memory()
    return {"message": "Chat history cleared"}
