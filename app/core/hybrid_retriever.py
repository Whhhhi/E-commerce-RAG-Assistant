import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from app.services.embedding import get_embedding


class HybridRetriever(BaseRetriever):
    """ChromaDB + BM25 混合检索器，使用 RRF 融合排序"""

    chroma_index: Optional[Chroma] = None
    bm25_index: Optional[BM25Retriever] = None
    top_k: int = 5
    rrf_k: int = 60
    chroma_weight: float = 1.0
    bm25_weight: float = 1.0
    filter: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.chroma_index is None and self.bm25_index is None:
            raise ValueError("chroma_index 和 bm25_index 不能同时为 None")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        chroma_results: List[Document] = []
        bm25_results: List[Document] = []

        if self.chroma_index is not None:
            results = self.chroma_index.similarity_search_with_relevance_scores(
                query,
                k=self.top_k * 2,
                filter=self.filter,
            )
            chroma_results = [
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "__chroma_score": float(score)},
                )
                for doc, score in results
            ]

        if self.bm25_index is not None:
            bm25_results = self.bm25_index.get_relevant_documents(query)
            bm25_results = bm25_results[: self.top_k * 2]

        doc_scores: dict[str, float] = {}

        for rank, doc in enumerate(chroma_results):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + self.chroma_weight / (
                self.rrf_k + rank + 1
            )

        for rank, doc in enumerate(bm25_results):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + self.bm25_weight / (
                self.rrf_k + rank + 1
            )

        ranked_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)[:self.top_k]

        seen_ids = set()
        final_docs = []
        for doc in chroma_results + bm25_results:
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            if doc_id in ranked_ids and doc_id not in seen_ids:
                doc.metadata["rrf_score"] = round(doc_scores[doc_id], 4)
                final_docs.append(doc)
                seen_ids.add(doc_id)

        return final_docs[: self.top_k]

    @classmethod
    def from_knowledge_base(
        cls,
        kb_name: str,
        top_k: int = 5,
        rrf_k: int = 60,
        filter: Optional[Dict[str, Any]] = None,
    ) -> "HybridRetriever":
        """从持久化目录加载指定知识库的检索器"""
        base_path = Path("data")

        chroma_path = base_path / "chroma_db"
        chroma_index = None
        if chroma_path.exists():
            embedding = get_embedding()
            chroma_index = Chroma(
                collection_name=kb_name,
                embedding_function=embedding,
                persist_directory=str(chroma_path),
            )

        bm25_path = base_path / "bm25_index" / f"{kb_name}.pkl"
        bm25_index = None
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                bm25_index = pickle.load(f)

        return cls(
            chroma_index=chroma_index,
            bm25_index=bm25_index,
            top_k=top_k,
            rrf_k=rrf_k,
            filter=filter,
        )