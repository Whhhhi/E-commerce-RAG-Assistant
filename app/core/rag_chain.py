from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from operator import itemgetter
from typing import List

from app.core.hybrid_retriever import HybridRetriever
from app.services.llm import get_llm

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个电商客服助手。给定对话历史和最新的用户问题，"
        "将其重写为一个可以在没有历史上下文的独立问题。"
        "如果问题本身就是独立的，直接返回原问题。"
        "只返回问题本身，不要附加任何解释。",
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个专业的电商客服助手。请基于以下提供的上下文信息回答用户问题。\n\n"
        "要求：\n"
        "1. 如果上下文包含足够信息，给出准确、简洁的回答\n"
        "2. 如果上下文不足以回答，如实告知用户你不知道，不要编造\n"
        "3. 在回答末尾用 [来源: 文档名称] 标注信息来源\n"
        "4. 保持回答语气友好、专业\n\n"
        "上下文信息：\n{context}",
    ),
    ("human", "{question}"),
])


def format_docs(docs: List[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        formatted.append(f"[文档 {i+1}] (来源: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(
    retriever: HybridRetriever,
    *,
    use_history: bool = True,
) -> RunnableParallel:
    if use_history:
        chain = (
            RunnableParallel(
                rewritten_question=RunnableParallel(
                    question=itemgetter("question"),
                    chat_history=itemgetter("chat_history"),
                ) | CONDENSE_PROMPT | get_llm() | StrOutputParser(),
                original_question=itemgetter("question"),
                chat_history=itemgetter("chat_history"),
            )
            | RunnableParallel(
                context=(
                    itemgetter("rewritten_question")
                    | retriever
                    | format_docs
                ),
                question=itemgetter("rewritten_question"),
                original_question=itemgetter("original_question"),
            )
        )
    else:
        chain = (
            RunnableParallel(
                context=itemgetter("question") | retriever | format_docs,
                question=itemgetter("question"),
            )
        )

    full_chain = chain | QA_PROMPT | get_llm() | StrOutputParser()

    return full_chain