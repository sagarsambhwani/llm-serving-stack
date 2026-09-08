# ADR 0004: Asynchronous Multi-Concurrency Load-Testing & Benchmark Suite

## Status
**Accepted**

## Context
Standard HTTP benchmarking tools (such as ApacheBench `ab` or basic `curl` scripts) are inadequate for evaluating Large Language Model inference servers because:
1. **Streaming & TTFT Blindness:** Traditional tools only measure end-to-end response time, completely missing **Time to First Token (TTFT)**, which dictates perceived human responsiveness.
2. **Token Awareness:** Traditional tools report requests per second, failing to compute **Output Tokens per Second (Throughput)** or prompt-vs-generation token ratios.
3. **Continuous Batching Saturation:** A test suite must evaluate multiple concurrency tiers ($1 \to 2 \to 4 \to 8 \to 16$) to discover the GPU's memory bandwidth saturation knee.

## Decision
We implemented a custom, asynchronous, multi-concurrency benchmark engine in Python (`loadtest/benchmark.py` and `loadtest/visualize.py`).

### Key Architectural Characteristics:
1. **Asynchronous Concurrency Control:** Uses `asyncio.Semaphore` and `httpx.AsyncClient` to simulate exact concurrent user streams.
2. **Dual-Mode Evaluation:**
   - **Streaming Mode (`--stream`):** Parses Server-Sent Events chunks in real time to capture exact TTFT timestamps.
   - **Non-Streaming Mode:** Measures batch API payload round-trip latency.
3. **Percentile Distribution:** Calculates P50, P90, P95, and P99 metrics for both latency and TTFT using `numpy`.
4. **Automated Visualization:** Reads benchmark JSON outputs and plots 4-panel comparison charts (Output Tok/s, Req/s, Latency percentiles, and TTFT).

## Consequences

### Positive:
- Discovered the exact continuous batching sweet spot ($C=4$, $156.4\text{ tok/s}$) and the saturation knee ($C=8$, prefill queue buildup) on the Tesla T4 GPU.
- Validates both local FastAPI gateway overhead and remote Cloudflare tunnel transit latency under load.

### Negative / Trade-offs:
- Running high-concurrency benchmarks against remote tunnels requires sufficient local CPU threads to avoid client-side benchmarking bottlenecks.
