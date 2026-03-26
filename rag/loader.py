import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document


class DocumentLoader:
    """文档加载器 - 支持多种格式"""

    @staticmethod
    def load_directory(directory_path: str) -> list[Document]:
        """加载目录下的所有文档"""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            return []

        reader = SimpleDirectoryReader(
            input_dir=directory_path,
            recursive=True,
            exclude_hidden=True,
        )
        return reader.load_data()

    @staticmethod
    def load_file(file_path: str) -> list[Document]:
        """加载单个文件"""
        reader = SimpleDirectoryReader(input_files=[file_path])
        return reader.load_data()

    @staticmethod
    def load_text(text: str, metadata: dict = None) -> Document:
        """从文本创建文档"""
        return Document(text=text, metadata=metadata or {})


loader = DocumentLoader()
