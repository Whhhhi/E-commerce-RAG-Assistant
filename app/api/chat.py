import time
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.chat import ChatRequest, ChatResponse, Source
from app.core.intent_router import IntentRouter
from app.core.rag_chain import build_rag_chain
from app.core.hybrid_retriever import HybridRetriever
from app.core.session import SessionManager
from app.core.errors import ErrorCode, AppException

router = APIRouter()
intent_router = IntentRouter()
session_mgr = SessionManager()

product_chain = None
policy_chain = None


def init_chains():
    global product_chain, policy_chain

    try:
        product_retriever = HybridRetriever.from_knowledge_base("product")
        if product_retriever.chroma_index is not None or product_retriever.bm25_index is not None:
            product_chain = build_rag_chain(product_retriever)
    except Exception as e:
        print(f"[WARNING] 商品知识库加载失败（跳过）: {e}")

    try:
        policy_retriever = HybridRetriever.from_knowledge_base("policy")
        if policy_retriever.chroma_index is not None or policy_retriever.bm25_index is not None:
            policy_chain = build_rag_chain(policy_retriever)
    except Exception as e:
        print(f"[WARNING] 政策知识库加载失败（跳过）: {e}")


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t0 = time.time()

    history = session_mgr.get_history(req.session_id)
    route = intent_router.route(req.message)

    try:
        if route["use_rag"]:
            chain = (
                product_chain if route["knowledge_base"] == "product"
                else policy_chain
            )
            if chain is None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": ErrorCode.RAG_EMPTY_KNOWLEDGE_BASE.value,
                        "message": f"知识库 '{route['knowledge_base']}' 为空，请先上传文档",
                    },
                )
            answer = chain.invoke({
                "question": req.message,
                "chat_history": history,
            })

        elif route["use_tool"]:
            from app.tools.order_tool import query_order
            answer = query_order(req.message)

        else:
            from app.services.llm import get_llm
            llm = get_llm()
            answer = llm.invoke(req.message).content

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": ErrorCode.RAG_CHAIN_ERROR.value,
                "message": f"处理请求时出错: {str(e)}",
            },
        )

    session_mgr.add_message(req.session_id, "human", req.message)
    session_mgr.add_message(req.session_id, "ai", answer)

    latency_ms = round((time.time() - t0) * 1000, 2)

    return ChatResponse(
        answer=answer,
        sources=[],
        session_id=req.session_id,
        intent=route["intent"].value,
        conversation_id=f"conv_{req.session_id}",
        latency_ms=latency_ms,
    )