# 📊 LLM Inference Benchmarking & Performance Guide

This guide details what metrics we measure, why they matter, the mathematical formulas used, and how to interpret continuous batching saturation curves.

---

## 📐 1. The Core LLM Performance Metrics

### A. Time to First Token (TTFT)
* **Definition:** The time elapsed from when a user sends a request to when the first token arrives.
* **Represents:** The **Prefill Phase** (processing and encoding the input prompt into the KV-cache).
* **Formula:**
  $$\text{TTFT} = t_{\text{first\_token\_received}} - t_{\text{request\_sent}}$$
* **Code Implementation in [`loadtest/benchmark.py`](file:///e:/Downloads/vLLM/loadtest/benchmark.py):**
  ```python
  start_time = time.perf_counter()
  first_token_time = None

  async for line in response.aiter_lines():
      if line.startswith("data: ") and content != "[DONE]":
          if first_token_time is None:
              first_token_time = time.perf_counter() - start_time
  ```

---

### B. Output Token Throughput (Generation Speed)
* **Definition:** Total completion tokens generated across all concurrent users per unit of wall-clock time.
* **Represents:** Total GPU decode capacity and hardware utilization efficiency.
* **Formula:**
  $$\text{Output Tokens / sec} = \frac{\sum_{i=1}^{N} \text{completion\_tokens}_i}{\text{Total Benchmark Duration (seconds)}}$$

---

### C. Request Throughput (Requests Per Second - RPS)
* **Definition:** Completed end-to-end user queries resolved per second.
* **Formula:**
  $$\text{RPS} = \frac{\text{Total Successful Requests}}{\text{Total Benchmark Duration (seconds)}}$$

---

### D. Tail Latency Percentiles (P50, P90, P95, P99)
* **Why percentiles instead of averages?** Simple averages hide bad spikes ("jitter") caused by prefill preemption or network stalls.
  - **P50 (Median):** Typical user experience ($50\%$ of requests were faster).
  - **P95:** $95\%$ of requests were faster ($5\%$ were slower).
  - **P99:** The worst $1\%$ outlier user experience.

---

## 📈 2. Understanding Continuous Batching Saturation

When benchmarking vLLM or SGLang, plotting **Concurrency vs. Throughput** reveals two distinct operational regions:

```
Throughput (tok/s)
    ▲
160 ┼                      * (C=4: Sweet Spot: 156.4 tok/s)
    │                     / \
120 ┼                    /   * (C=8: Saturation Knee: 140.6 tok/s)
    │                   /
 80 ┼          * (C=2: 66.8 tok/s)
    │         /
 40 ┼  * (C=1: 34.8 tok/s)
    └──┴───────┴────────┴──────┴────────► Concurrency (C)
       1       2        4      8
```

### 1. The Linear Scaling Region ($C=1 \to 4$):
* **Behavior:** Throughput increases from $34.8 \to 156.4\text{ tok/s}$ ($\mathbf{4.5\times}$ increase) while latency remains constant ($\sim 1.8\text{s}$).
* **Why:** Single requests leave the GPU memory bus largely idle. vLLM batches multiple decode steps together in the same memory read cycle, absorbing the concurrent requests for "free".

### 2. The Saturation Knee ($C=8+$):
* **Behavior:** Throughput plateaus ($\sim 140\text{ tok/s}$), and latency doubles from $1.8\text{s} \to 3.56\text{s}$.
* **Why:** The GPU memory bandwidth and compute units reach $100\%$ capacity. Incoming prefill prompts must queue behind ongoing decode iterations, leading to prefill contention.

---

## 🚀 3. How to Run the Benchmark Suite

```bash
# 1. Short prompt baseline test
python loadtest/benchmark.py --concurrency-levels 1,2,4,8 --scenario short

# 2. Long prefill stress test
python loadtest/benchmark.py --concurrency-levels 1,2,4,8 --scenario long

# 3. Prefix caching efficiency test
python loadtest/benchmark.py --concurrency-levels 1,2,4,8 --scenario prefix_cache

# 4. Generate visualization charts
python loadtest/visualize.py
```
