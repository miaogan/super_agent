import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document


class DocumentLoader:
    """文档加载器 - 支持多种格式"""

    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.doc', '.html', '.xml'}

    @staticmethod
    def load_directory(directory_path: str) -> list[Document]:
        """加载目录下的所有文档"""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            return []

        try:
            # 使用 Path.glob 查找文件
            data_path = Path(directory_path)
            all_files = []
            
            for ext in DocumentLoader.SUPPORTED_EXTENSIONS:
                all_files.extend(data_path.glob(f'**/*{ext}'))
                all_files.extend(data_path.glob(f'*{ext}'))

            # 去重并过滤
            files = list(set(all_files))
            if not files:
                return []

            docs = []
            for f in files:
                try:
                    with open(f, 'r', encoding='utf-8') as fp:
                        content = fp.read()
                        doc = Document(
                            text=content,
                            metadata={
                                'file_name': str(f.name),
                                'file_path': str(f)
                            }
                        )
                        docs.append(doc)
                except Exception:
                    # 跳过无法读取的文件
                    continue

            return docs

        except Exception:
            return []

    @staticmethod
    def load_file(file_path: str) -> list[Document]:
        """加载单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return [Document(text=content, metadata={'file_name': os.path.basename(file_path)})]
        except Exception:
            return []

    @staticmethod
    def load_text(text: str, metadata: dict = None) -> Document:
        """从文本创建文档"""
        return Document(text=text, metadata=metadata or {})


loader = DocumentLoader()
