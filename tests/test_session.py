import pytest
from app.core.session import SessionManager


class TestSessionManager:
    def test_get_empty_history(self):
        session = SessionManager()
        assert session.get_history("test_session") == []

    def test_add_message(self):
        session = SessionManager()
        session.add_message("test_session", "human", "你好")
        session.add_message("test_session", "ai", "你好！")
        
        history = session.get_history("test_session")
        assert len(history) == 2
        assert history[0] == ("human", "你好")
        assert history[1] == ("ai", "你好！")

    def test_history_truncation(self):
        session = SessionManager()
        session.MAX_ROUNDS = 2
        
        for i in range(10):
            session.add_message("test_session", "human", f"问{i}")
            session.add_message("test_session", "ai", f"答{i}")
        
        history = session.get_history("test_session")
        assert len(history) == 4

    def test_clear_session(self):
        session = SessionManager()
        session.add_message("test_session", "human", "hello")
        session.clear("test_session")
        
        assert session.get_history("test_session") == []