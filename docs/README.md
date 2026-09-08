# 📚 LLM Serving Stack — Documentation Hub

Welcome to the technical documentation repository for the **LLM Serving Stack**.

---

## 📑 Table of Contents

### 🏛️ Architecture & System Design
* **[System Architecture & Data Flow](file:///e:/Downloads/vLLM/docs/architecture.md)** — End-to-end component breakdown, network topology, and request/response lifecycle.
* **[Lab vs. Production Serving](file:///e:/Downloads/vLLM/docs/production_vs_lab.md)** — Detailed comparison between our mini stack and enterprise-scale architectures (PD Disaggregation, RDMA, KEDA).

---

### 🧠 Deep-Dive Engineering Guides
* **[Multi-Model Serving & Memory Allocation](file:///e:/Downloads/vLLM/docs/multi_model_serving.md)** — PagedAttention memory mechanics, prefill compute contention, Multi-LoRA serving, explicit VRAM partitioning, and multimodal VLMs.
* **[Supported Models & Hardware Matrix](file:///e:/Downloads/vLLM/docs/supported_models_matrix.md)** — Comprehensive comparison of parameters, VRAM footprints, benchmarks, licenses, and T4 GPU compatibility.
* **[Benchmarking & Performance Metrics](file:///e:/Downloads/vLLM/docs/benchmarking_guide.md)** — Complete guide to TTFT, token throughput, tail latency percentiles (P50/P95/P99), and continuous batching saturation curves.
* **[Google Colab & Cloudflare Setup Guide](file:///e:/Downloads/vLLM/colab/COLAB_GUIDE.md)** — Step-by-step instructions for launching GPU vLLM workers and tunnels.

---

### 📜 Architecture Decision Records (ADRs)
Explore our formal design decisions and engineering trade-offs in **[`docs/adr/`](file:///e:/Downloads/vLLM/docs/adr/index.md)**:
* **[ADR 0001: OpenAI-Compatible FastAPI Gateway](file:///e:/Downloads/vLLM/docs/adr/0001-openai-compatible-fastapi-gateway.md)**
* **[ADR 0002: vLLM Engine & PagedAttention Selection](file:///e:/Downloads/vLLM/docs/adr/0002-vllm-engine-and-pagedattention.md)**
* **[ADR 0003: Multi-Model Serving & GPU Memory Strategy](file:///e:/Downloads/vLLM/docs/adr/0003-multi-model-serving-and-memory-allocation.md)**
* **[ADR 0004: Asynchronous Multi-Concurrency Load Tester](file:///e:/Downloads/vLLM/docs/adr/0004-async-load-testing-and-benchmarking.md)**

---

### 🗺️ Master Roadmap
* **[Active Feature & Experiment Roadmap](file:///e:/Downloads/vLLM/ROADMAP.md)** — SGLang comparison, Multimodality (VLMs), Quantization, and Fine-Tuning Multi-LoRA integration.
