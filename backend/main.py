import json
import logging
import time
from typing import AsyncGenerator, Dict, List
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from backend.auth import key_manager, verify_admin_key, verify_api_key
from backend.config import settings
from backend.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    KeyCreateRequest,
    KeyInfo,
    ModelCard,
    ModelListResponse,
    UsageInfo,
)

from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vllm-gateway")

# Global persistent HTTP client pool for zero-overhead upstream proxying
http_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.REQUEST_TIMEOUT, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=100, keepalive_expiry=60.0),
    )
    yield
    if http_client and not http_client.is_closed:
        await http_client.aclose()


app = FastAPI(
    title="OpenAI-Compatible LLM Gateway (vLLM Backend)",
    description="High-performance API Gateway with OpenAI-style API Key authentication and proxying to vLLM.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web clients / Playground UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint checking gateway and upstream vLLM connectivity."""
    upstream_status = "unreachable"
    try:
        client = http_client or httpx.AsyncClient(timeout=3.0)
        resp = await client.get(
            f"{settings.VLLM_BASE_URL}/models",
            headers={"Authorization": f"Bearer {settings.VLLM_API_KEY}"},
        )
        if resp.status_code == 200:
            upstream_status = "connected"
    except Exception:
        upstream_status = "offline_or_starting"

    return {
        "status": "ok",
        "gateway": "online",
        "upstream_vllm_url": settings.VLLM_BASE_URL,
        "upstream_vllm_status": upstream_status,
        "active_model": settings.VLLM_MODEL,
        "timestamp": int(time.time()),
    }


@app.get("/v1/models", response_model=ModelListResponse, tags=["OpenAI Compatible"])
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models in OpenAI API format."""
    # Attempt to query upstream vLLM models first, fallback to configured model
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.VLLM_BASE_URL}/models",
                headers={"Authorization": f"Bearer {settings.VLLM_API_KEY}"},
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.warning(f"Could not reach upstream vLLM /models: {e}. Returning default.")

    return ModelListResponse(
        data=[
            ModelCard(
                id=settings.VLLM_MODEL,
                created=int(time.time()),
                owned_by="vllm-gateway",
            )
        ]
    )


async def _stream_vllm_response(
    payload: dict,
    client_api_key: str,
) -> AsyncGenerator[str, None]:
    """Forward streaming response from vLLM as Server-Sent Events (SSE)."""
    model_name = payload.get("model", settings.VLLM_MODEL)
    base_url = settings.get_upstream_url(model_name)
    upstream_url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.VLLM_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt_tokens = 0
    completion_tokens = 0

    try:
        client = http_client or httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT)
        async with client.stream(
            "POST", upstream_url, json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raw_err = error_text.decode('utf-8', errors='ignore')
                err_detail = raw_err
                try:
                    parsed = json.loads(raw_err)
                    if isinstance(parsed, dict):
                        err_detail = parsed.get("message") or parsed.get("detail") or raw_err
                except Exception:
                    pass
                logger.error(f"Upstream error {response.status_code} from {base_url}: {err_detail}")
                yield f"data: {json.dumps({'error': {'message': f'Upstream Error ({response.status_code}): {err_detail}', 'code': response.status_code}})}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_content = line[6:].strip()
                    if data_content == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        chunk = json.loads(data_content)
                        completion_tokens += 1
                        if "usage" in chunk and chunk["usage"]:
                            prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
                            completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)
                    except Exception:
                        pass
                yield f"{line}\n\n"

        # Record token usage for client API key
        key_manager.record_usage(client_api_key, prompt_tokens, max(1, completion_tokens))

    except httpx.ConnectError:
        logger.error(f"Failed to connect to upstream instance at {base_url}.")
        err_msg = json.dumps({
            "error": {
                "message": f"Gateway cannot reach inference server at {base_url}. Ensure upstream server is running.",
                "type": "gateway_connection_error",
                "code": "upstream_unavailable",
            }
        })
        yield f"data: {err_msg}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", tags=["OpenAI Compatible"])
async def chat_completions(
    req: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    OpenAI-compatible Chat Completion endpoint.
    Supports dynamic multi-model routing, non-streaming, and streaming (`stream=True`).
    """
    payload = req.model_dump(exclude_none=True)
    if not payload.get("model"):
        payload["model"] = settings.VLLM_MODEL

    model_name = payload["model"]
    base_url = settings.get_upstream_url(model_name)

    # If client requested SSE streaming
    if req.stream:
        return StreamingResponse(
            _stream_vllm_response(payload, api_key),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming forward
    upstream_url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.VLLM_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        client = http_client or httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT)
        resp = await client.post(upstream_url, json=payload, headers=headers)
        
        if resp.status_code != 200:
            logger.error(f"vLLM upstream error ({resp.status_code}): {resp.text}")
            raise HTTPException(
                status_code=resp.status_code,
                detail=resp.json() if resp.headers.get("content-type") == "application/json" else resp.text,
            )

        data = resp.json()

        # Track usage metrics
        usage = data.get("usage", {})
        p_tokens = usage.get("prompt_tokens", 0)
        c_tokens = usage.get("completion_tokens", 0)
        key_manager.record_usage(api_key, p_tokens, c_tokens)

        return data

    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": {
                    "message": f"Gateway cannot connect to vLLM server at {settings.VLLM_BASE_URL}. Ensure Colab/vLLM is running and the tunnel URL is correctly set.",
                    "type": "gateway_connection_error",
                    "code": "upstream_unavailable",
                }
            },
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"error": {"message": "Upstream vLLM request timed out.", "type": "timeout"}},
        )


# ==========================================
# Admin Management & API Key Issuance API
# ==========================================

@app.post("/admin/keys", response_model=KeyInfo, tags=["Admin Key Management"])
async def create_api_key(
    req: KeyCreateRequest,
    admin_token: str = Depends(verify_admin_key),
):
    """Generate a new OpenAI-style API key (`sk-antigravity-...`) for a client."""
    new_key_info = key_manager.generate_key(name=req.name)
    logger.info(f"Generated new API key for {req.name}: {new_key_info.key[:18]}...")
    return new_key_info


@app.get("/admin/keys", response_model=Dict[str, KeyInfo], tags=["Admin Key Management"])
async def list_api_keys(
    admin_token: str = Depends(verify_admin_key),
):
    """List all registered API keys and their total token consumption."""
    return key_manager.list_keys()
