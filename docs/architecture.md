# 🏗️ System Architecture & Data Flow

This document details the end-to-end architecture, network topology, and execution lifecycle of the **LLM Serving Stack**.

---

## 🏛️ End-to-End Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             CLIENT LAYER                                    │
│                                                                             │
│  [ OpenAI Python SDK ]   [ Interactive Chatbot ]   [ Async Load Tester ]    │
│  (example_client.py)        (chat.py)              (loadtest/benchmark.py)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTPS / REST (Authorization: Bearer sk-...)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FASTAPI API GATEWAY (:9000)                             │
│                                                                             │
│  ├── 🔐 Security & Auth: Validates `sk-antigravity-...` Bearer tokens       │
│  ├── 📊 Token Accounting: Tracks prompt & completion tokens per key        │
│  ├── ⚡ Connection Pool: Persistent httpx.AsyncClient (keep-alive: 50)       │
│  ├── 📡 Streaming Proxy: Asynchronous SSE event generator                  │
│  └── 🛠️ Admin Engine: Dynamic key issuance & usage inspection (/admin/keys)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTPS (Encrypted WAN Transit)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CLOUDFLARE QUICK TUNNEL                                │
│                                                                             │
│  • Public Edge Ingress: https://*.trycloudflare.com                         │
│  • Zero-configuration WAN bridge to private Colab runtime                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP Loopback (:8000)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 GPU INFERENCE WORKER (Google Colab / Cloud GPU)             │
│                                                                             │
│  ├── 🚀 vLLM Engine v0.28.0 (Model: Qwen/Qwen2.5-1.5B-Instruct)             │
│  ├── 🧩 PagedAttention: Non-contiguous KV-cache virtual memory management    │
│  ├── 🔄 Continuous Batching: Iteration-level scheduling                     │
│  ├── 📑 Prefix Caching: Hash-based KV-block reuse across prompts            │
│  ├── ⚡ CUDA Graphs: Full & Piecewise static kernel capture                 │
│  └── 🎮 Hardware: NVIDIA Tesla T4 GPU (16 GB VRAM, Turing Architecture)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Request / Response Lifecycle

### 1. Client Dispatch:
* A client script (e.g. `chat.py`) initiates a `POST /v1/chat/completions` request with payload `{ "model": "Qwen/Qwen2.5-1.5B-Instruct", "messages": [...], "stream": true }` and header `Authorization: Bearer sk-antigravity-dev-key`.

### 2. Gateway Ingestion & Auth Validation:
* FastAPI's dependency injection (`verify_api_key`) intercepts the request and verifies the token against the active key registry in `backend/auth.py`.
* If invalid or missing, it immediately rejects the request with a standard OpenAI `401 Unauthorized` response, shielding the GPU from unauthorized traffic.

### 3. Upstream Forwarding:
* The gateway routes the request over its persistent HTTP connection pool to the upstream Cloudflare Quick Tunnel URL (`https://xxxx.trycloudflare.com/v1/chat/completions`).

### 4. GPU Inference (vLLM Execution):
1. **Prefill Phase:** vLLM checks the prefix cache for matching system prompt blocks. For uncached tokens, the GPU computes attention matrices across Tensor Cores.
2. **Decode Phase:** Tokens are generated autoregressively one by one using PagedAttention memory pages.
3. **Streaming SSE Yield:** As each token is sampled, vLLM yields a chunk (`data: {"choices": [{"delta": {"content": "..."}}]}`).

### 5. Stream Proxying & Usage Finalization:
* The Gateway forwards each SSE chunk directly to the client without buffering.
* Upon completion (`[DONE]`), the Gateway parses the final token count and records the consumption against the client's API key.
