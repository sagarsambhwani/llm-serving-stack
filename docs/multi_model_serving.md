# 🧠 Multi-Model Serving & GPU Memory Allocation Deep Dive

This document provides a comprehensive technical analysis of serving **multiple LLMs or multimodal models on a single GPU** (e.g., Google Colab NVIDIA Tesla T4 16GB, A100 40GB/80GB), examining memory mechanics (PagedAttention & KV-Cache), compute contention (Prefill vs. Decode), and concrete architectural serving patterns.

---

## 🛑 1. The Core Problem: The GPU Memory Wall

When serving Large Language Models, GPU VRAM is consumed by three distinct entities:

$$\text{Total VRAM} = \text{Model Weights} + \text{Activation Memory / CUDA Context} + \text{KV-Cache Pool}$$

```
Single 16GB GPU VRAM (Default vLLM Behavior):
┌───────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Model Weights (~3 GB)     │  Pre-allocated PagedAttention KV-Cache Pool (~11.5 GB)     │
│ (Qwen 2.5 1.5B FP16)      │  (gpu_memory_utilization = 0.90)                            │
└───────────────────────────┴─────────────────────────────────────────────────────────────┘
```

### Why Naive Multi-Model Serving Fails:
By default, inference engines like vLLM pre-allocate **$90\%$ of total GPU VRAM** to construct a massive KV-cache block pool for maximum batch concurrency. 
If an engineer attempts to start a second model server on another port (e.g., Model A on `:8000` and Model B on `:8001`), the second process crashes immediately with a **`CUDA Out of Memory (OOM)`** error because the first process already claimed all available memory.

---

## 🔬 2. Memory & Compute Dynamics: Under the Hood

### A. PagedAttention & KV-Cache Fragmentation
In traditional serving frameworks, memory for a request's KV cache is allocated as a contiguous block of maximum context length ($4096$ tokens), wasting $60\%-80\%$ of VRAM due to internal and external memory fragmentation.

**PagedAttention** divides the KV-cache into fixed-size virtual blocks (e.g., $16$ tokens per block), dynamically allocating blocks on-demand:

```
Virtual Memory Pages ──► [ Page 0 (16 tok) ] ──► [ Page 3 (16 tok) ] (Non-contiguous in VRAM)
```

#### What happens if you run multiple instances?
If you split VRAM into multiple isolated inference processes, **each process maintains its own closed PagedAttention block manager**. 
* **KV-Cache Fragmentation:** If Model A is idle, its allocated KV-cache memory cannot be borrowed by Model B. Model B will experience out-of-memory errors or queue stalls even though $50\%$ of the physical GPU VRAM is sitting idle.

---

### B. Compute Contention: Prefill Spikes vs. Decode Bandwidth
Inference consists of two distinct phases with opposite hardware bottlenecks:

1. **Prefill Phase (Prompt Evaluation):**
   - **Characteristics:** Compute-bound (Matrix-Matrix multiplication $\text{GEMM}$).
   - **Hardware Load:** Saturates **$100\%$ of GPU Tensor Cores**.
2. **Decode Phase (Token Generation):**
   - **Characteristics:** Memory-Bandwidth-bound (Matrix-Vector multiplication $\text{GEMV}$).
   - **Hardware Load:** Reads entire model weights from VRAM for every single token generated.

```
GPU Streaming Multiprocessors (SM) Timeline:
Time ──►
┌──────────────────────────────────────┬───────────────────────────────────────┐
│ Model A Prefill (100% Tensor Cores)  │ Model B Prefill (Queued / Serialized) │
└──────────────────────────────────────┴───────────────────────────────────────┘
                                       ▲
                   Notice: Model B experiences severe TTFT latency spike
```

* **The Multi-Model Collision:** If Model A and Model B receive requests simultaneously, their prefill phases will compete for the same Streaming Multiprocessors (SMs), forcing CUDA streams to serialize and causing large **Time to First Token (TTFT) spikes**.

---

## 🛠️ 3. The 4 Multi-Model Architectural Strategies

### Strategy 1: Dynamic Multi-LoRA Serving (The Gold Standard 🏆)

Instead of loading separate base models for different domains (e.g., coding, legal, JSON extraction, summarization), you load **one shared base model** and attach lightweight **LoRA adapters** dynamically.

```
Single 16GB GPU VRAM (Multi-LoRA):
┌───────────────────────────┬────────┬────────┬───────────────────────────────────────────┐
│ Shared Base Model (3 GB)  │ LoRA 1 │ LoRA 2 │ Unified Shared PagedAttention Pool (10GB) │
│ (Qwen 2.5 1.5B)           │ (30MB) │ (30MB) │ Shared dynamically across ALL tasks!      │
└───────────────────────────┴────────┴────────┴───────────────────────────────────────────┘
```

#### Key Advantages:
1. **Zero Base Weight Redundancy:** The 1.5B/7B base parameters are loaded into memory exactly once.
2. **Unified KV Cache:** All adapters share the exact same PagedAttention pool. If Task 1 is idle, Task 2 can utilize $100\%$ of the KV-cache.
3. **Dynamic Routing:** Handled automatically in the forward pass using `model="<adapter_name>"`.

#### Implementation:
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --enable-lora \
    --lora-modules \
        json_extractor=/path/to/json_lora \
        code_assistant=/path/to/code_lora \
    --max-loras 8 \
    --max-lora-rank 64 \
    --port 8000
```

---

### Strategy 2: Explicit VRAM Partitioning (Heterogeneous Base Models)

When you must serve two completely different model architectures (e.g., `Qwen2.5-1.5B` for chat and `Llama-3.2-1B` for classification), you can partition VRAM using `--gpu-memory-utilization`.

```
Single 16GB GPU VRAM (Explicit Partitioning):
┌───────────────────┬───────────────────┬───────────────────┬───────────────────┐
│ Model A (3 GB)    │ KV Cache A (4 GB) │ Model B (2.5 GB)  │ KV Cache B (4 GB) │
│ Process 1: Port 8000 (util=0.45)      │ Process 2: Port 8001 (util=0.45)      │
└───────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

#### Exact Memory Calculation (16 GB T4 GPU):
* Process 1 (`util=0.45`): $16\text{ GB} \times 0.45 = 7.2\text{ GB Total}$
  - Model Weights ($3.0\text{ GB}$) + KV Cache ($4.2\text{ GB}$) $\approx 38\times$ concurrent 4k streams.
* Process 2 (`util=0.45`): $16\text{ GB} \times 0.45 = 7.2\text{ GB Total}$
  - Model Weights ($2.5\text{ GB}$) + KV Cache ($4.7\text{ GB}$) $\approx 42\times$ concurrent 4k streams.

#### How the Gateway Routes Requests:
```python
# In FastAPI Gateway router:
MODEL_ROUTES = {
    "Qwen/Qwen2.5-1.5B-Instruct": "http://localhost:8000/v1",
    "meta-llama/Llama-3.2-1B-Instruct": "http://localhost:8001/v1",
}
```

---

### Strategy 3: Multimodal Vision-Language Serving (Unified Graph)

For vision-language models (e.g., `Qwen2-VL-2B-Instruct`), you do **not** need two separate servers for vision and language.

The architecture combines a **Vision Transformer (ViT)** and an **Autoregressive LLM** in a single unified computation graph:
1. **Vision Encoder:** Processes image patches into visual tokens during the prefill phase.
2. **LLM Decoder:** Autoregressively generates text conditioned on both visual and text tokens.

```
Image Input ──► [ ViT Patch Encoder ] ──► [ Visual Tokens ] ──┐
                                                              ├──► [ LLM Decoder ] ──► Text Output
Text Prompt ──► [ Text Tokenizer ]    ──► [ Text Tokens   ] ──┘
```

* **VRAM Footprint on T4:** Model weights ($\sim 4.5\text{ GB}$) + PagedAttention KV Cache ($\sim 9.5\text{ GB}$). Runs completely in a single process.

---

### Strategy 4: Dynamic Model Offloading / Swapping (Cold Swapping)

For environments hosting 5+ large models where requests arrive intermittently:
* **Mechanism:** The Gateway keeps 1 model active in GPU VRAM and holds other models in CPU RAM or fast local NVMe storage.
* **On-Demand Swap:** When a request for Model B arrives, Model A is unmapped from VRAM and Model B's weights are loaded via PCIe into VRAM.
* **Trade-off:**
  - Saves VRAM ($100\%$ capacity available for active model).
  - High cold-start latency ($2\text{s} - 8\text{s}$ swap time per request).

---

## 📊 Comparison Matrix

| Strategy | Base Model Redundancy | KV-Cache Efficiency | TTFT / Latency | Implementation Complexity | Best Used For |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Multi-LoRA** | **Zero** ($1$ Base Model) | **Optimal** (100% Shared Pool) | **Ultra-Fast** (Sub-100ms) | Low | Domain-specific task models (JSON, Code, Chat) |
| **VRAM Partitioning** | High ($2+$ Base Models) | Fragmented (Isolated Pools) | Good (Moderate TTFT) | Medium | Different model families on 1 GPU |
| **Unified VLM** | N/A (Multimodal Graph) | **Optimal** (Unified Pool) | Fast (Image prefill dependent) | Low | Image-to-text, visual QA, OCR |
| **Dynamic Swapping** | Zero | Full per active model | Slow ($2-8\text{s}$ swap latency) | High | Multi-tenant batch workloads with idle gaps |
