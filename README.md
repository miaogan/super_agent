# Super Agent - RAG Application

基于 LlamaIndex + FastAPI + LangChain 的 RAG 智能问答系统

## 功能特性

- 📄 文档加载与处理（支持 PDF、TXT、Markdown 等）
- 🔍 向量语义检索
- 🤖 LLM 对话问答
- 🌐 FastAPI REST API
- 📦 支持多种 Embedding 和 LLM 提供商

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置环境变量
export OPENAI_API_KEY="your-api-key"
# 或使用其他 LLM 提供商

# 3. 启动服务
uvicorn app.main:app --reload

# 4. 添加文档到知识库
curl -X POST "http://localhost:8000/documents" \
  -F "file=@your_document.pdf"

# 5. 问答
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "你的问题"}'
```

## API 文档

启动后访问: http://localhost:8000/docs

## 项目结构

```
super_agent/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 应用入口
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py    # API 路由
│   └── config.py       # 配置管理
├── rag/
│   ├── __init__.py
│   ├── loader.py       # 文档加载器
│   ├── indexer.py      # 索引构建
│   ├── retriever.py    # 检索器
│   └── chain.py        # RAG Chain
├── data/               # 文档存储
├── tests/              # 单元测试
├── requirements.txt    # 依赖
└── .env.example        # 环境变量示例
```
