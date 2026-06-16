import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestUploadEndpoint:
    @pytest.mark.asyncio
    async def test_upload_supported_format(self):
        from app.main import app

        with patch("app.api.upload.load_document") as mock_load:
            mock_load.return_value = []
            with patch("app.api.upload.split_documents") as mock_split:
                mock_split.return_value = []
                with patch("app.api.upload.VectorStoreService") as mock_vs:
                    with patch("app.api.upload.BM25StoreService") as mock_bm25:

                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.post(
                                "/upload",
                                files={"file": ("test.txt", b"hello world", "text/plain")}
                            )
                            assert response.status_code == 200
                            data = response.json()
                            assert "document_id" in data
                            assert data["status"] == "indexed"

    @pytest.mark.asyncio
    async def test_upload_unsupported_format(self):
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/upload",
                files={"file": ("test.exe", b"binary content", "application/octet-stream")}
            )
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["code"] == "UPLOAD_UNSUPPORTED_FORMAT"

    @pytest.mark.asyncio
    async def test_upload_empty_file(self):
        from app.main import app

        with patch("app.api.upload.load_document") as mock_load:
            mock_load.return_value = []
            with patch("app.api.upload.split_documents") as mock_split:
                mock_split.return_value = []
                with patch("app.api.upload.VectorStoreService") as mock_vs:
                    with patch("app.api.upload.BM25StoreService") as mock_bm25:

                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.post(
                                "/upload",
                                files={"file": ("test.txt", b"", "text/plain")}
                            )
                            assert response.status_code == 200
