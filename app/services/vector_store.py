from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.services.embedding import get_embedding


class VectorStoreService:
    """ChromaDB 向量库服务"""

    def __init__(self, kb_name: str):
        self.kb_name = kb_name
        self.store_path = Path("data") / "chroma_db"
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.embedding = get_embedding()
        self._client: Chroma | None = None

    def _get_client(self) -> Chroma:
        """获取或创建 Chroma 客户端"""
        if self._client is None:
            self._client = Chroma(
                collection_name=self.kb_name,
                embedding_function=self.embedding,
                persist_directory=str(self.store_path),
            )
        return self._client

    def build(self, docs: List[Document]) -> int:
        """全量重建向量库（会清空现有数据）"""
        client = self._get_client()
        client.delete_collection()
        self._client = Chroma(
            collection_name=self.kb_name,
            embedding_function=self.embedding,
            persist_directory=str(self.store_path),
        )
        self._client.add_documents(docs)
        return len(docs)

    def add_documents(self, docs: List[Document]) -> int:
        """增量添加文档"""
        client = self._get_client()
        client.add_documents(docs)
        return len(docs)

    def delete_by_ids(self, ids: List[str]) -> None:
        """根据 ID 删除文档"""
        client = self._get_client()
        client.delete(ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """向量检索，支持元数据过滤"""
        client = self._get_client()
        return client.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """向量检索，返回相关性分数，支持元数据过滤"""
        client = self._get_client()
        return client.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=filter,
        )

    def get_index(self) -> Chroma:
        """获取底层 Chroma 索引"""
        return self._get_client()

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        client = self._get_client()
        return {
            "total_documents": client._collection.count(),
            "collection_name": self.kb_name,
        }