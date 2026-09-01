import pytest
import json
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from backend.main import app
from backend.config import settings

client = TestClient(app)


def test_streaming_chat_completions():
    valid_key = list(settings.valid_api_keys)[0]
    
    mock_chunks = [
        b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Paged"}}]}\n\n',
        b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Attention"}}]}\n\n',
        b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":" works!"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    async def mock_aiter_lines():
        for chunk in [
            'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Paged"}}]}',
            'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Attention"}}]}',
            'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":" works!"}}]}',
            'data: [DONE]'
        ]:
            yield chunk

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = mock_aiter_lines

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp
        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient.stream", return_value=MockStreamContext()):
        res = client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "messages": [{"role": "user", "content": "Explain PagedAttention"}],
                "stream": True,
            },
            headers={"Authorization": f"Bearer {valid_key}"}
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        assert "Paged" in res.text
        assert "Attention" in res.text
        assert "[DONE]" in res.text
