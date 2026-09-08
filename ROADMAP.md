# 🗺️ LLM Inference Stack — Experiment & Feature Roadmap

A prioritized master roadmap of architectural experiments, engine comparisons, multimodal serving, and production optimizations for this inference stack.

---

## 🎯 Active Roadmap & Experiments

### 1. 🥊 Engine Showdown (vLLM vs SGLang vs TGI)
- [ ] **SGLang Integration**: Deploy SGLang on Colab/GPU with RadixAttention (`--enable-radix-cache`).
- [ ] **Head-to-Head Prefix Caching Benchmark**: Compare vLLM hash-based cache vs SGLang Radix Tree cache under multi-turn conversations and shared system prompts.
- [ ] **Multi-Engine Visualizer**: Generate comparative side-by-side charts plotting Throughput (tok/s), TTFT (ms), and P95 latency across engines.
- [ ] **TGI Evaluation**: Test Hugging Face's Rust router vs Python engines.

---

### 2. 👁️ Multimodality (Vision-Language Models / VLMs)
- [ ] **Serve Small Vision Models on Colab GPU**: Deploy `Qwen/Qwen2-VL-2B-Instruct` or `MiniCPM-V-2_6` on a single Tesla T4 GPU.
- [ ] **Multimodal OpenAI API Support**: Validate gateway and client support for OpenAI multimodal payload schema:
  ```json
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "What is depicted in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }
  ```
- [ ] **Vision CLI / Interactive Chatbot**: Extend [`chat.py`](file:///e:/Downloads/vLLM/chat.py) with `/image <path/url>` command to analyze local images or diagrams in real time.
- [ ] **VLM Performance Benchmark**: Measure ViT (Vision Transformer) image patch encoding latency vs LLM decode latency across image resolutions ($256\times 256$, $512\times 512$, $1080\text{p}$).

---

### 3. ⚡ Quantization & Memory Optimization
- [ ] **AWQ & GPTQ 4-Bit Serving**: Serve `Qwen2.5-7B-Instruct-AWQ` or `Mistral-7B-AWQ` on a single 16GB T4 GPU (models that otherwise cannot fit in FP16).
- [ ] **FP8 Serving (on Ada / Hopper GPUs)**: Benchmark FP8 Tensor Cores vs FP16 precision.
- [ ] **Memory & KV Cache Capacity Benchmark**: Measure maximum concurrent streams gained by quantization ($4\text{-bit}$ weights free up VRAM for $3\times$ larger KV caches).

---

### 4. 🚀 Advanced Serving Techniques
- [ ] **Speculative Decoding**: Pair a small draft model (`Qwen2.5-0.5B`) with a larger target model (`Qwen2.5-7B`) to achieve $1.8\times - 2.5\times$ faster single-stream decode speed.
- [ ] **Structured / Constrained JSON Generation**: Benchmark Outlines / XGrammar / SGLang Regex FSM for guaranteeing valid JSON schema output at zero latency penalty.
- [ ] **Chunked Prefill**: Test how interleaving chunked prompt prefill with token decoding eliminates TTFT spikes under heavy concurrency.

---

### 5. 🎯 Fine-Tuning Integration & Multi-LoRA Serving
- [ ] **Dynamic Multi-LoRA Adapter Serving**: Serve single base model + multiple task-specific LoRA adapters dynamically via `--enable-lora`.
- [ ] **Fine-Tuned Quality & Accuracy Benchmark**: Automated test harness running held-out test splits through the gateway to measure JSON validity, schema compliance, and field-level exact match.
- [ ] **Base vs. Fine-Tuned A/B Benchmark**: Measure latency, throughput, and accuracy delta between zero-shot base models and domain-adapted LoRA models.
- [ ] **Unified Monorepo Architecture**: Combine fine-tuning pipelines (QLoRA / SFT) and serving stack (vLLM / SGLang / Gateway) into a flagship end-to-end LLM engineering repository.

---

### 6. 🛡️ Production Gateway & Infrastructure
- [ ] **Redis Rate Limiting**: Implement distributed Token Bucket rate limiting (Requests Per Minute & Tokens Per Minute) per API key.
- [ ] **Prefix-Affinity Load Balancer**: Implement a routing proxy that inspects prompt hashes and routes requests to the GPU node with warm KV cache.
- [ ] **Prometheus & Grafana Observability**: Live dashboard scraping `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_factor`, and `vllm:e2e_request_latency_seconds`.
