# ADR 0001: OpenAI-Compatible FastAPI API Gateway

## Status
**Accepted**

## Context
When serving Large Language Models (LLMs) from remote GPU compute nodes (e.g. Google Colab, AWS EC2, or Kubernetes clusters) to consumer applications or client devices:
1. Direct client-to-engine connections expose internal server ports and raw GPU processes directly to the internet.
2. Production clients and ecosystems (LangChain, LlamaIndex, OpenAI Python/TypeScript SDKs, Cursor, open-webui) expect the standardized OpenAI REST API schema (`Authorization: Bearer sk-...`, `/v1/chat/completions`, `/v1/models`).
3. Enterprise requirements mandate API key authentication, per-client token usage tracking, connection pooling, and request validation before traffic hits the expensive GPU worker.

## Decision
We implemented a dedicated **FastAPI API Gateway** (`backend/main.py`) that sits between client applications and the upstream vLLM inference server.

### Key Architectural Characteristics:
1. **OpenAI Compatibility**: Exposes standard `/v1/chat/completions`, `/v1/models`, and `/health` endpoints.
2. **Bearer Token Authentication**: Enforces `sk-antigravity-...` API key validation with standard OpenAI error envelopes.
3. **Usage Accounting**: Tracks prompt tokens, completion tokens, and total request counts per API key.
4. **Persistent Connection Pooling**: Leverages an asynchronous `lifespan` context manager with a shared `httpx.AsyncClient` pool (`max_keepalive_connections=50`, `max_connections=100`) to reuse TLS sessions and eliminate TCP/TLS handshake latency over tunnels.
5. **Streaming Support**: Full Server-Sent Events (SSE) streaming proxying with chunked token forwarding.

## Consequences

### Positive:
- Any standard OpenAI SDK or application can seamlessly connect to our self-hosted inference server with zero custom client code.
- Protects the GPU worker behind an authentication and validation boundary.
- Connection pooling reduces per-request latency over wide-area networks by $100\text{ms}-250\text{ms}$.

### Negative / Trade-offs:
- Introduces an extra network hop (Client $\to$ Gateway $\to$ vLLM Engine), adding $\sim 2\text{ms}-5\text{ms}$ of gateway processing overhead.
- Streaming responses require careful async generator lifecycle management to prevent connection leaks.
