import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from backend.main import app
from backend.config import settings

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["gateway"] == "online"


def test_auth_missing_api_key():
    response = client.get("/v1/models")
    assert response.status_code == 401
    err = response.json()
    assert "error" in err["detail"]
    assert err["detail"]["error"]["code"] == "missing_api_key"


def test_auth_invalid_api_key():
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer sk-invalid-key-xyz"}
    )
    assert response.status_code == 401
    err = response.json()
    assert "error" in err["detail"]
    assert err["detail"]["error"]["code"] == "invalid_api_key"


def test_auth_valid_api_key():
    valid_key = list(settings.valid_api_keys)[0]
    response = client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {valid_key}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1


def test_admin_generate_key():
    admin_key = settings.GATEWAY_ADMIN_KEY
    # Unauthorized attempt
    res = client.post("/admin/keys", json={"name": "test-app"})
    assert res.status_code == 403

    # Authorized admin request
    res = client.post(
        "/admin/keys",
        json={"name": "test-app"},
        headers={"Authorization": f"Bearer {admin_key}"}
    )
    assert res.status_code == 200
    key_data = res.json()
    assert key_data["key"].startswith("sk-antigravity-")
    assert key_data["name"] == "test-app"

    # Verify newly created key works
    new_token = key_data["key"]
    models_res = client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {new_token}"}
    )
    assert models_res.status_code == 200


def test_chat_completions_mock_upstream():
    valid_key = list(settings.valid_api_keys)[0]
    mock_vllm_response = {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I am served via vLLM with PagedAttention."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 12,
            "total_tokens": 22
        }
    }

    from unittest.mock import MagicMock
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_vllm_response
    mock_resp.headers = {"content-type": "application/json"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        payload = {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "messages": [{"role": "user", "content": "Hello vLLM!"}],
            "temperature": 0.7,
            "max_tokens": 50,
        }

        res = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {valid_key}"}
        )

        assert res.status_code == 200
        data = res.json()
        assert data["id"] == "chatcmpl-test-123"
        assert data["choices"][0]["message"]["content"] == "Hello! I am served via vLLM with PagedAttention."
        assert data["usage"]["total_tokens"] == 22
