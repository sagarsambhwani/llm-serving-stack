# vLLM Inference Stack & Async Load-Testing Suite

A high-performance, modular LLM serving architecture featuring:
- **FastAPI API Gateway**: OpenAI-compatible endpoint router (`/v1/chat/completions`, `/v1/models`, `/health`) with OpenAI-style API Key authentication (`Bearer sk-...`) and token accounting.
- **Upstream vLLM GPU Server**: Google Colab / cloud instance scripts serving Qwen2.5 with PagedAttention and prefix caching over Cloudflare Quick Tunnels.
- **Asynchronous Load-Testing Engine**: High-throughput benchmark client testing continuous batching saturation, measuring Time-to-First-Token (TTFT), P50/P95/P99 latency, req/s, and token throughput.
- **Visualizer**: Generates latency percentiles and throughput comparison charts.

---

## 🏗️ Architecture

```
[ Load-Test Client / Python Apps / OpenAI SDK ]
                     │
                     ▼ (Authorization: Bearer sk-antigravity-...)
┌─────────────────────────────────────────────────────────────┐
│ FastAPI API Gateway (Port 9000)                             │
│ • Validates Bearer API keys (`sk-...`)                      │
│ • Tracks token usage & active request counts                │
│ • Manages streaming SSE & standard JSON completions         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼ (HTTPS / Cloudflare Quick Tunnel)
┌─────────────────────────────────────────────────────────────┐
│ Google Colab / GPU Worker (Port 8000)                       │
│ • vLLM Engine (`Qwen/Qwen2.5-1.5B-Instruct`)                │
│ • PagedAttention & Automatic Prefix Caching                 │
│ • NVIDIA GPU Hardware Acceleration (T4 / A100)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Set Up Environment & Dependencies

Activate the project virtual environment and install requirements:
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### 2. Launch vLLM in Google Colab (GPU)

Follow the step-by-step instructions in [`colab/COLAB_GUIDE.md`](file:///e:/Downloads/vLLM/colab/COLAB_GUIDE.md) or run the notebook cells:
1. Open Google Colab with a **T4 GPU** runtime.
2. Install vLLM and Cloudflared:
   ```bash
   !pip install -q -U vllm
   !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && dpkg -i cloudflared-linux-amd64.deb
   ```
3. Start vLLM:
   ```bash
   vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000 --api-key dev-secret --enable-prefix-caching
   ```
4. Expose via Cloudflare Quick Tunnel:
   ```bash
   !cloudflared tunnel --url http://localhost:8000
   ```
5. Copy the generated `https://xxxx.trycloudflare.com` URL into your local `.env`:
   ```env
   VLLM_BASE_URL=https://xxxx.trycloudflare.com/v1
   VLLM_API_KEY=dev-secret
   VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
   ```

---

### 3. Start the Local FastAPI Gateway

```bash
.\.venv\Scripts\python -m uvicorn backend.main:app --reload --port 9000
```

- **Health Check**: `http://localhost:9000/health`
- **Interactive Swagger Docs**: `http://localhost:9000/docs`

---

### 4. Connect Using Standard OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="sk-antigravity-dev-key",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Explain KV caching."}],
)
print(response.choices[0].message.content)
```

Or run the sample script:
```bash
.\.venv\Scripts\python example_client.py
```

---

### 5. Run the Multi-Concurrency Load Test

Benchmark performance across multiple concurrency tiers (e.g. 1, 2, 4, 8, 16):

```bash
# Test against FastAPI Gateway
.\.venv\Scripts\python loadtest/benchmark.py \
    --target-url http://localhost:9000/v1 \
    --api-key sk-antigravity-dev-key \
    --concurrency-levels 1,2,4,8,16 \
    --requests-per-level 20 \
    --max-tokens 100 \
    --scenario short

# Generate visual performance graphs
.\.venv\Scripts\python loadtest/visualize.py
```

#### Benchmark Scenarios Available:
- `--scenario short`: Quick query testing (20-50 tokens).
- `--scenario long`: Heavy prefill context testing (400-800 tokens).
- `--scenario prefix_cache`: Shared system prompt testing to evaluate vLLM's automatic prefix caching.
