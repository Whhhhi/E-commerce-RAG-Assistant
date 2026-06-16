from typing import List, Tuple
import time
import threading


class SessionManager:
    MAX_ROUNDS = 20

    def __init__(self):
        self._store: dict[str, list] = {}
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        with self._lock:
            return list(self._store.get(session_id, []))

    def add_message(self, session_id: str, role: str, content: str):
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []
            self._store[session_id].append((role, content))
            max_messages = self.MAX_ROUNDS * 2
            if len(self._store[session_id]) > max_messages:
                self._store[session_id] = self._store[session_id][-max_messages:]

    def clear(self, session_id: str):
        with self._lock:
            self._store.pop(session_id, None)

    def cleanup_stale(self, max_age_seconds: int = 3600):
        pass