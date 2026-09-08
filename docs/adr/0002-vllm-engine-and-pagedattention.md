# ADR 0002: vLLM Inference Engine & PagedAttention Selection

## Status
**Accepted**

## Context
Traditional LLM serving frameworks (like naive Hugging Face `transformers.pipeline` or basic PyTorch servers) suffer from severe memory fragmentation and static batching bottlenecks:
1. **Contiguous KV-Cache Allocation:** Allocating continuous memory for maximum sequence length ($4096$ tokens) wastes $60\%-80\%$ of GPU VRAM.
2. **Static Batching:** Requests in a batch must wait for the longest sequence to complete before releasing resources.
3. **Prefix Redundancy:** Recomputing common system prompts across multi-turn queries wastes expensive prefill compute.

## Decision
We selected **vLLM** with **PagedAttention** and **Automatic Prefix Caching** as the primary serving engine for our GPU worker node.

### Key Architectural Characteristics:
1. **PagedAttention:** Virtual memory paging for KV-cache blocks (non-contiguous allocation), achieving $>95\%$ VRAM utilization.
2. **Continuous Batching:** Dynamic iteration-level scheduling that inserts new requests and evicts finished sequences on every decode step.
3. **Prefix Caching (`--enable-prefix-caching`):** Hash-based matching that caches KV blocks for common system prompts, reducing TTFT by up to $80\%$ on repetitive tasks.
4. **CUDA Graph Optimization:** Capturing static execution graphs to eliminate PyTorch CPU-launch overhead during token decoding.

## Consequences

### Positive:
- Achieved **$4.5\times$ throughput scaling** ($34.8 \to 156.4\text{ tokens/s}$) on a single Colab Tesla T4 GPU with no latency penalty under concurrency.
- Accommodates up to **$82\times$ concurrent 4k context streams** in 9.01 GB of KV-cache memory.

### Negative / Trade-offs:
- Memory greediness: By default, vLLM claims $90\%$ of VRAM, requiring careful configuration if co-locating processes.
- Compilation warmup: Requires $30\text{s}-60\text{s}$ at startup to capture CUDA graphs and warm up PyTorch JIT kernels.
