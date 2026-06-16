from pathlib import Path
from typing import List

from langchain_core.documents import Document


def load_document(file_path: Path, doc_id: str) -> List[Document]:
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(str(file_path))
    elif ext == ".md":
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
        loader = UnstructuredMarkdownLoader(str(file_path))
    elif ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_id"] = doc_id
        doc.metadata["source"] = file_path.name
    return docs