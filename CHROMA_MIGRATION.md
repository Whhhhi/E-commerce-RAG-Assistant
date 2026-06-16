# ChromaDB 迁移方案

> 版本：v1.0  
> 原方案：FAISS 向量库  
> 目标方案：ChromaDB（本地持久化）  
> 更新日期：2026-06-16

---

## 一、为什么选择 ChromaDB

### 1.1 电商场景需求分析

| 需求维度 | FAISS 现状 | ChromaDB 优势 |
|---------|-----------|--------------|
| **持久化** | 需要手动序列化/反序列化 | 内置持久化，自动保存 |
| **元数据过滤** | 不支持原生过滤 | 原生支持 metadata 过滤（category、brand 等） |
| **多租户/多知识库** | 需手动管理目录结构 | 支持 collection 概念，天然多库隔离 |
| **动态更新** | 增量更新需重新保存 | 支持增量添加、删除、更新 |
| **查询灵活性** | 仅支持向量检索 | 支持向量+元数据组合查询 |
| **生产就绪** | 学术工具，功能有限 | 专为生产环境设计 |

### 1.2 ChromaDB 核心优势

1. **元数据过滤支持**：电商场景经常需要按商品类别（category）、品牌（brand）、价格区间等过滤，ChromaDB 原生支持这些查询。

2. **本地持久化**：自动保存到磁盘，服务重启后数据不丢失，无需手动处理序列化。

3. **增量更新**：支持动态添加/删除文档，适合电商场景下商品信息的频繁更新。

4. **生产级 API**：提供 REST API 和 Python SDK，便于横向扩展。

---

## 二、目录结构调整

### 2.1 原有结构

```
data/
├── vector_store/     # FAISS 索引目录
│   ├── product/
│   └── policy/
├── bm25_index/       # BM25 索引
└── documents/        # 原始文档
```

### 2.2 迁移后结构

```
data/
├── chroma_db/        # ChromaDB 持久化目录（新增）
│   ├── collections/
│   └── chroma.sqlite3
├── bm25_index/       # 保留 BM25
└── documents/        # 保留原始文档
```

**清理说明**：
- 删除 `data/vector_store/` 目录（FAISS 旧数据）
- 创建 `data/chroma_db/` 目录（ChromaDB 持久化）

---

## 三、关键代码修改点

### 3.1 安装依赖

```bash
# 安装 ChromaDB 和 langchain-chroma
pip install chromadb langchain-chroma

# 移除不再需要的依赖（可选）
# pip uninstall faiss-cpu
```

### 3.2 修改 `app/services/vector_store.py`

将 FAISS VectorStoreService 替换为 ChromaDB 实现：

```python
from pathlib import Path
from typing import List, Dict, Any

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
```

### 3.3 修改 `app/core/hybrid_retriever.py`

更新 HybridRetriever 以支持 ChromaDB：

```python
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
    filter: Optional[Dict[str, Any]] = None  # 元数据过滤条件

    def __post_init__(self):
        if self.chroma_index is None and self.bm25_index is None:
            raise ValueError("chroma_index 和 bm25_index 不能同时为 None")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        chroma_results: List[Document] = []
        bm25_results: List[Document] = []

        # ChromaDB 检索
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

        # BM25 检索
        if self.bm25_index is not None:
            bm25_results = self.bm25_index.get_relevant_documents(query)
            bm25_results = bm25_results[: self.top_k * 2]

        # RRF 融合排序
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

        # ChromaDB
        chroma_path = base_path / "chroma_db"
        chroma_index = None
        if chroma_path.exists():
            embedding = get_embedding()
            chroma_index = Chroma(
                collection_name=kb_name,
                embedding_function=embedding,
                persist_directory=str(chroma_path),
            )

        # BM25
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
```

### 3.4 修改 `app/config.py`（可选）

添加 ChromaDB 相关配置：

```python
class Settings(BaseSettings):
    # ... 其他配置 ...

    # ChromaDB 配置
    chroma_persist_dir: str = "data/chroma_db"
    chroma_collection_product: str = "product"
    chroma_collection_policy: str = "policy"
```

---

## 四、建库与查询示例

### 4.1 初始化知识库

```python
from app.services.vector_store import VectorStoreService
from langchain_core.documents import Document

# 创建商品知识库
product_store = VectorStoreService(kb_name="product")

# 准备文档（带元数据）
docs = [
    Document(
        page_content="智能手机 X1 支持 5G 全网通，搭载骁龙8 Gen3处理器，配备6.7英寸AMOLED屏幕。",
        metadata={
            "doc_id": "doc_001",
            "source": "产品规格-智能手机X1.pdf",
            "category": "手机",
            "brand": "X品牌",
            "price_range": "3000-4000",
        }
    ),
    Document(
        page_content="无线蓝牙耳机 Pro 续航时间长达36小时，支持主动降噪功能。",
        metadata={
            "doc_id": "doc_002",
            "source": "产品规格-蓝牙耳机Pro.pdf",
            "category": "耳机",
            "brand": "X品牌",
            "price_range": "500-1000",
        }
    ),
]

# 构建向量库
product_store.add_documents(docs)

# 获取统计信息
stats = product_store.get_collection_stats()
print(f"文档总数: {stats['total_documents']}")
```

### 4.2 查询示例

```python
from app.core.hybrid_retriever import HybridRetriever

# 创建检索器
retriever = HybridRetriever.from_knowledge_base(
    kb_name="product",
    top_k=5,
    rrf_k=60,
)

# 基础检索
results = retriever.invoke("5G 手机")
for doc in results:
    print(f"内容: {doc.page_content}")
    print(f"来源: {doc.metadata.get('source')}")
    print(f"类别: {doc.metadata.get('category')}")
    print(f"品牌: {doc.metadata.get('brand')}")
    print(f"RRF分数: {doc.metadata.get('rrf_score')}")
    print("---")

# 带元数据过滤的检索
retriever_with_filter = HybridRetriever.from_knowledge_base(
    kb_name="product",
    top_k=5,
    filter={"category": "手机", "price_range": "3000-4000"},
)
filtered_results = retriever_with_filter.invoke("处理器")
```

### 4.3 更新/删除文档

```python
# 删除文档
product_store.delete_by_ids(["doc_001"])

# 增量添加新文档
new_docs = [
    Document(
        page_content="新款平板电脑 Tab X 配备11英寸LCD屏幕，支持手写笔。",
        metadata={
            "doc_id": "doc_003",
            "source": "产品规格-平板电脑TabX.pdf",
            "category": "平板",
            "brand": "X品牌",
            "price_range": "2000-3000",
        }
    ),
]
product_store.add_documents(new_docs)
```

---

## 五、迁移执行步骤

### 5.1 备份数据

```bash
# 备份原有 FAISS 数据（可选）
zip -r faiss_backup.zip data/vector_store/

# 备份 BM25 数据
zip -r bm25_backup.zip data/bm25_index/
```

### 5.2 安装依赖

```bash
pip install chromadb langchain-chroma
```

### 5.3 更新代码

1. 替换 `app/services/vector_store.py`（见 3.2 节）
2. 替换 `app/core/hybrid_retriever.py`（见 3.3 节）
3. 更新 `app/config.py`（可选，见 3.4 节）

### 5.4 清理旧数据

```bash
# 删除 FAISS 旧数据目录
rm -rf data/vector_store/

# 创建 ChromaDB 目录
mkdir -p data/chroma_db
```

### 5.5 重新上传文档

启动服务后，通过 `/upload` 接口重新上传知识库文档，或编写迁移脚本：

```python
# migration_script.py
import os
from pathlib import Path
from app.knowledge.loader import load_document
from app.knowledge.splitter import split_documents
from app.services.vector_store import VectorStoreService
from app.services.bm25_store import BM25StoreService

def migrate_from_faiss():
    # 遍历原有文档目录
    docs_dir = Path("data/documents")
    
    for kb_name in ["product", "policy"]:
        kb_dir = docs_dir / kb_name
        if not kb_dir.exists():
            continue
            
        vs = VectorStoreService(kb_name=kb_name)
        bm25 = BM25StoreService(kb_name=kb_name)
        
        for file_path in kb_dir.glob("*"):
            if file_path.suffix not in [".pdf", ".md", ".txt"]:
                continue
                
            doc_id = file_path.stem.replace("doc_", "")
            raw_docs = load_document(file_path, doc_id=doc_id)
            chunks = split_documents(raw_docs)
            
            vs.add_documents(chunks)
            bm25.add_documents(chunks)
            print(f"Migrated: {file_path.name} -> {len(chunks)} chunks")

if __name__ == "__main__":
    migrate_from_faiss()
```

### 5.6 验证迁移

```bash
# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 测试上传接口
curl -X POST http://localhost:8000/upload \
  -F "file=@test_document.pdf" \
  -F "knowledge_base=product"

# 测试聊天接口
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "这款手机支持5G吗？", "session_id": "test"}'
```

---

## 六、FAQ

### Q1: ChromaDB 如何保证数据一致性？

A: ChromaDB 使用 SQLite 作为元数据存储，向量数据存储在文件系统。每次写入操作都会自动持久化，服务重启后数据不丢失。

### Q2: 元数据过滤支持哪些操作？

A: ChromaDB 支持以下过滤操作：
- 精确匹配: `{"category": "手机"}`
- 数值比较: `{"price": {"$gt": 1000}}`
- 包含匹配: `{"tags": {"$contains": "新品"}}`

### Q3: 是否支持多线程/并发？

A: ChromaDB 支持线程安全操作。对于高并发场景，建议使用 ChromaDB 的远程模式（通过 REST API）。

### Q4: 迁移后性能有变化吗？

A: ChromaDB 在本地模式下性能与 FAISS 相当，但提供了更丰富的功能。对于大规模数据集（>10万文档），建议使用远程模式或考虑其他优化方案。

---

## 七、总结

| 维度 | FAISS | ChromaDB |
|------|-------|----------|
| 持久化 | 需手动处理 | 自动持久化 |
| 元数据过滤 | 不支持 | 原生支持 |
| 增量更新 | 需重新保存 | 支持 |
| 生产就绪 | 较低 | 高 |
| 功能丰富度 | 基础 | 丰富 |

**迁移收益**：
1. 支持按商品类别、品牌、价格等维度过滤
2. 自动持久化，无需担心数据丢失
3. 更好的生产环境支持
4. 保留原有 Hybrid Retrieval 能力（向量 + BM25）