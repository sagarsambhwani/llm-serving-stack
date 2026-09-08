# ADR 0003: Multi-Model Serving & GPU Memory Allocation Strategy

## Status
**Accepted**

## Context
Deploying multiple LLMs or task-specific models on constrained hardware (such as a single 16GB Tesla T4 GPU in Google Colab or cloud VMs) presents critical memory and compute trade-offs:
1. **The VRAM Wall:** Naive multi-process hosting causes immediate `CUDA Out of Memory (OOM)` errors due to simultaneous $90\%$ KV-cache pre-allocation.
2. **KV-Cache Fragmentation:** Splitting GPU VRAM statically into multiple vLLM instances prevents idle models from sharing their unused KV-cache pool with active models.
3. **Prefill vs. Decode Compute Contention:** Simultaneous prefill requests for different models saturate $100\%$ of GPU Tensor Cores, causing prefill queueing and TTFT spikes.

## Decision
We establish a tiered architectural strategy for multi-model serving based on use-case requirements:

```
                               ┌───────────────────────────────────────────────┐
                               │         MULTI-MODEL SERVING DECISION TREE      │
                               └───────────────────────┬───────────────────────┘
                                                       │
                           ┌───────────────────────────┴───────────────────────────┐
                           │ Is it multiple task experts on the same base model?   │
                           └───┬───────────────────────────────────────────────┬───┘
                               │ YES                                           │ NO
                               ▼                                               ▼
               ┌───────────────────────────────┐               ┌───────────────────────────────┐
               │    STRATEGY 1: MULTI-LoRA     │               │   Are the base models small   │
               │ • 1 Base Model loaded in VRAM │               │   enough to fit concurrently? │
               │ • Lightweight task adapters   │               └───┬───────────────────────┬───┘
               │ • 100% Shared KV-Cache Pool   │                   │ YES                   │ NO
               └───────────────────────────────┘                   ▼                       ▼
                                                   ┌─────────────────────────┐ ┌───────────────┐
                                                   │ STRATEGY 2: PARTITIONING│ │  STRATEGY 4:  │
                                                   │ • util=0.45 per process │ │ DYNAMIC SWAP  │
                                                   │ • Gateway route mapping │ │ • Cold swap   │
                                                   └─────────────────────────┘ └───────────────┘
```

### Strategy Hierarchy:
1. **Primary Pattern — Dynamic Multi-LoRA Serving:**
   - For domain-specific task variants (e.g., Code Assistant, JSON Extractor, Medical QA), load a single base model (`Qwen2.5-1.5B`) and serve multiple LoRA adapters dynamically using vLLM's `--enable-lora`.
   - **Rationale:** Base weights are loaded only once ($3\text{ GB}$), while all adapters share a unified $11.5\text{ GB}$ PagedAttention KV-cache pool.
2. **Secondary Pattern — Explicit VRAM Partitioning:**
   - When hosting distinct model architectures (e.g., `Qwen2.5-1.5B` + `Llama-3.2-1B`), configure separate vLLM worker processes using `--gpu-memory-utilization 0.45` and route requests at the Gateway layer.
   - **Rationale:** Prevents OOM while bounding memory allocations predictably.
3. **Multimodal Pattern — Unified Vision-Language Serving:**
   - For image-to-text models (e.g. `Qwen2-VL-2B-Instruct`), serve both the Vision Transformer (ViT) patch encoder and LLM autoregressive decoder in a single unified graph process.

## Consequences

### Positive:
- **Multi-LoRA** provides maximum memory efficiency, zero weight duplication, and dynamically shared KV-cache capacity.
- Clear operational guidelines prevent developer errors (OOM crashes, prefill contention blindspots).

### Negative / Trade-offs:
- **VRAM Partitioning** incurs KV-cache fragmentation; an idle model's memory cannot be repurposed by an active model.
- Simultaneous prefill requests on multi-model setups will serialize across GPU Streaming Multiprocessors, causing brief TTFT delays.
