import pytest
from unittest.mock import Mock, patch
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings


class MockChatModel(BaseChatModel):
    def __init__(self, responses):
        self.responses = responses
        self.index = 0

    def invoke(self, input, **kwargs):
        response = self.responses[self.index % len(self.responses)]
        self.index += 1
        return Mock(content=response)


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.1] * 1536 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 1536


@pytest.fixture
def mock_llm():
    return MockChatModel(["product_inquiry", "这款手机支持5G"])


@pytest.fixture
def mock_embeddings():
    return MockEmbeddings()


@pytest.fixture
def mock_session_manager():
    from app.core.session import SessionManager
    return SessionManager()


@pytest.fixture
def mock_intent_router():
    from app.core.intent_router import IntentRouter
    return IntentRouter(use_llm_fallback=False)