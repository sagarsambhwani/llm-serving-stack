# Google Colab vLLM Server & Cloudflare Quick Tunnel Guide

This guide gives you the exact code to start a GPU-accelerated **vLLM** inference server on Google Colab and expose it over a secure **Cloudflare Quick Tunnel** to your local FastAPI Gateway and Load Tester.

---

## 📓 Available Colab Notebooks

We provide two pre-configured, ready-to-run Jupyter Notebooks in the repository:

1. **[Qwen 2.5 GPU Server](file:///e:/Downloads/vLLM/colab/vllm_gpu_server.ipynb)** (`colab/vllm_gpu_server.ipynb`):
   - Optimized for `Qwen/Qwen2.5-1.5B-Instruct` in FP16 with PagedAttention and prefix caching.
2. **[Ungated & Uncensored Models Server](file:///e:/Downloads/vLLM/colab/vllm_uncensored_models_server.ipynb)** (`colab/vllm_uncensored_models_server.ipynb`):
   - Features an interactive dropdown selector for **Dolphin 2.9.3 (1.5B)**, **Hermes 3 (8B)**, **Llama 3.2 (3B Abliterated)**, and **Llama 3.1 (8B Abliterated)** with zero Hugging Face gating.

---

## Step 1: Open Google Colab with GPU Runtime

1. Go to [Google Colab](https://colab.research.google.com/).
2. Upload either notebook (`colab/vllm_gpu_server.ipynb` or `colab/vllm_uncensored_models_server.ipynb`).
3. Click **Runtime** → **Change runtime type** → Select **T4 GPU** (or A100 if you have Colab Pro).

---

## Step 2: Colab Notebook Cells

### Cell 1: Check GPU Details
```python
!nvidia-smi
```

### Cell 2: Install vLLM & Cloudflare Tunnel
```python
# Fix Colab CUDA/TorchAudio mismatch
!pip uninstall -y torchaudio

# Install latest vLLM
!pip install -q -U vllm

# Download and install cloudflared
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb
```

### Cell 3: Start vLLM Server in the Background
```python
import subprocess
import time

# Start vLLM serving Qwen2.5-1.5B-Instruct on port 8000
# Notice --enable-prefix-caching for automatic KV cache reuse!
vllm_cmd = [
    "vllm", "serve", "Qwen/Qwen2.5-1.5B-Instruct",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--api-key", "dev-secret",
    "--enable-prefix-caching",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.90"
]

print("Starting vLLM server...")
vllm_process = subprocess.Popen(vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Wait for server to load weights and be ready
for line in vllm_process.stdout:
    print(line, end="")
    if "Application startup complete" in line or "Uvicorn running on" in line:
        print("\n✅ vLLM is ready to accept requests!")
        break
```

### Cell 4: Start Cloudflare Quick Tunnel
```python
import re
import subprocess

tunnel_cmd = ["cloudflared", "tunnel", "--url", "http://localhost:8000"]
tunnel_proc = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

print("Opening Cloudflare Quick Tunnel...")
for line in tunnel_proc.stdout:
    print(line, end="")
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        public_url = match.group(0)
        print(f"\n=======================================================")
        print(f"🎉 PUBLIC TUNNEL URL: {public_url}")
        print(f"Set this in your local .env as:")
        print(f"VLLM_BASE_URL={public_url}/v1")
        print(f"=======================================================\n")
        break
```

### Cell 5 (Optional): In-Colab Zero-Network Latency Benchmark
To measure true GPU + vLLM continuous batching performance without internet latency:

```python
# Run vLLM's official benchmark command directly inside Colab
!python3 -m vllm.entrypoints.openai.bench_serving \
    --backend vllm \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len 128 \
    --num-prompts 50 \
    --request-rate 10 \
    --host localhost \
    --port 8000 \
    --api-key dev-secret
```

---

## Step 3: Connect Your Local Gateway

1. Copy the `https://xxxx.trycloudflare.com` URL printed in Colab Cell 4.
2. In your local `e:\Downloads\vLLM\.env` file, update:
   ```env
   VLLM_BASE_URL=https://xxxx.trycloudflare.com/v1
   VLLM_API_KEY=dev-secret
   VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
   ```
3. Start the FastAPI gateway locally:
   ```bash
   .\.venv\Scripts\uvicorn backend.main:app --reload --port 9000
   ```
4. Run the load test:
   ```bash
   .\.venv\Scripts\python loadtest/benchmark.py --concurrency-levels 1,2,4,8,16 --requests-per-level 20
   ```
