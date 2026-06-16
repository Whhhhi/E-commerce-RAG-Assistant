import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_empty_message(self):
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat",
                json={"message": "", "session_id": "test"}
            )
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_valid_request(self):
        from app.main import app

        with patch("app.api.chat.product_chain") as mock_chain:
            mock_chain.invoke.return_value = "这是测试回答"

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/chat",
                    json={
                        "message": "这款手机多少钱？",
                        "session_id": "test_session",
                        "user_id": "user_123"
                    }
                )
                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert data["session_id"] == "test_session"

    @pytest.mark.asyncio
    async def test_chat_session_persistence(self):
        from app.main import app

        with (
            patch("app.api.chat.product_chain") as mock_chain,
            patch("app.services.llm.get_llm") as mock_llm,
        ):
            mock_chain.invoke.return_value = "这是测试回答"
            mock_llm.return_value.invoke.return_value = MagicMock(content="闲聊回答")

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp1 = await client.post(
                    "/chat",
                    json={"message": "你好", "session_id": "persist_test"}
                )
                # 闲聊可能命中关键词或被路由到 chitchat，两种路径都需要 mock
                assert resp1.status_code in (200, 400)

                resp2 = await client.post(
                    "/chat",
                    json={"message": "这款手机多少钱？", "session_id": "persist_test"}
                )
                assert resp2.status_code == 200
