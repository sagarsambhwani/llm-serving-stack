# 🏛️ Architecture Decision Records (ADRs)

This directory documents the key architectural and design decisions made in the **LLM Serving Stack**.

---

## 📑 Index of Records

| ADR # | Title | Status | Date | Summary |
| :--- | :--- | :---: | :---: | :--- |
| **[ADR 0001](file:///e:/Downloads/vLLM/docs/adr/0001-openai-compatible-fastapi-gateway.md)** | OpenAI-Compatible FastAPI API Gateway | Accepted | 2026-08-30 | Decided to place a FastAPI gateway in front of vLLM for `Bearer sk-...` authentication, token accounting, and TLS connection pooling. |
| **[ADR 0002](file:///e:/Downloads/vLLM/docs/adr/0002-vllm-engine-and-pagedattention.md)** | vLLM Engine & PagedAttention Selection | Accepted | 2026-08-30 | Selected vLLM with PagedAttention and automatic prefix caching over traditional static batching frameworks. |
| **[ADR 0003](file:///e:/Downloads/vLLM/docs/adr/0003-multi-model-serving-and-memory-allocation.md)** | Multi-Model Serving & GPU Memory Allocation Strategy | Accepted | 2026-09-08 | Established Multi-LoRA as primary pattern and explicit VRAM partitioning as secondary pattern for multi-model serving on constrained GPUs. |
| **[ADR 0004](file:///e:/Downloads/vLLM/docs/adr/0004-async-load-testing-and-benchmarking.md)** | Asynchronous Multi-Concurrency Load-Testing & Benchmark Suite | Accepted | 2026-08-30 | Implemented async Python load tester with exact TTFT, TPS, and latency percentile tracking. |
