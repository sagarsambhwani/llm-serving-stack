"""
A lightweight mock vLLM server to enable local testing of the entire stack
and load tester without needing an active Colab GPU instance.
"""

import asyncio
import time
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Mock vLLM Server")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


@app.get("/health")
@app.get("/v1/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    if not authorization or "dev-secret" not in authorization:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return {
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen2.5-1.5B-Instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, authorization: Optional[str] = Header(None)):
    if not authorization or "dev-secret" not in authorization:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Simulate prefill time (~25ms)
    await asyncio.sleep(0.025)

    tokens = [
        "PagedAttention", " efficiently", " manages", " KV", " cache",
        " memory", " blocks", " in", " GPU", " VRAM", " eliminating",
        " fragmentation", " and", " maximizing", " serving", " throughput."
    ]

    if req.stream:
        async def event_generator():
            # Initial TTFT chunk
            for token in tokens:
                # Simulate inter-token latency (~10ms per token = 100 tok/s)
                await asyncio.sleep(0.010)
                chunk = {
                    "id": "chatcmpl-mock-123",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                import json
                yield f"data: {json.dumps(chunk)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    # Non-streaming response: simulate full generation duration
    await asyncio.sleep(len(tokens) * 0.010)
    return {
        "id": "chatcmpl-mock-123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(tokens),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": len(tokens),
            "total_tokens": 15 + len(tokens),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
