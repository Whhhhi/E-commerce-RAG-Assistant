import pickle
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


class BM25StoreService:
    def __init__(self, kb_name: str):
        self.kb_name = kb_name
        self.index_path = Path("data") / "bm25_index" / f"{kb_name}.pkl"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._retriever: BM25Retriever | None = None

    def build(self, docs: List[Document]) -> int:
        self._retriever = BM25Retriever.from_documents(docs)
        self._save()
        return len(docs)

    def add_documents(self, docs: List[Document]) -> int:
        if self._retriever is None:
            self._retriever = self._load()
        if self._retriever is None:
            return self.build(docs)

        self._retriever.add_documents(docs)
        self._save()
        return len(docs)

    def _save(self):
        if self._retriever is not None:
            with open(self.index_path, "wb") as f:
                pickle.dump(self._retriever, f)

    def _load(self) -> BM25Retriever | None:
        if self.index_path.exists():
            with open(self.index_path, "rb") as f:
                return pickle.load(f)
        return None

    def get_retriever(self) -> BM25Retriever | None:
        if self._retriever is None:
            self._retriever = self._load()
        return self._retriever