"""
Standalone runner script for serving Ungated & Uncensored models on Google Colab or remote GPU boxes.
Installs dependencies, applies Colab PyTorch fixes, launches vLLM, and exposes the Cloudflare Quick Tunnel.
"""

import argparse
import os
import re
import subprocess
import sys
import time

DEFAULT_MODEL = "cognitivecomputations/dolphin-2.9.3-qwen2-1.5b"
API_KEY = os.getenv("API_KEY", "dev-secret")
PORT = os.getenv("PORT", "8000")


def run_cmd(cmd, check=True):
    print(f">> Executing: {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def install_dependencies():
    print("📦 Installing vLLM and Cloudflare Tunnel with Colab compatibility fixes...")
    run_cmd("pip uninstall -y torchaudio", check=False)
    run_cmd("pip install -q -U vllm")
    run_cmd("wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb")
    run_cmd("dpkg -i cloudflared-linux-amd64.deb")


def start_vllm(model_name: str, port: str, api_key: str, max_model_len: int = 4096, gpu_util: float = 0.90):
    print(f"🚀 Launching vLLM with uncensored model: {model_name}...")
    vllm_cmd = [
        "vllm", "serve", model_name,
        "--host", "0.0.0.0",
        "--port", port,
        "--api-key", api_key,
        "--enable-prefix-caching",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_util),
    ]
    proc = subprocess.Popen(vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in proc.stdout:
        print(line, end="")
        if "Application startup complete" in line or "Uvicorn running on" in line:
            print("\n✅ vLLM Server is UP and accepting requests!\n")
            break
    return proc


def start_tunnel(port: str):
    print("🌐 Spawning Cloudflare Quick Tunnel...")
    tunnel_cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"]
    proc = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in proc.stdout:
        print(line, end="")
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            url = match.group(0)
            print("\n" + "=" * 65)
            print(f"🎉 TUNNEL READY: {url}")
            print(f"\n📝 Update your local .env with:")
            print(f"VLLM_BASE_URL={url}/v1")
            print(f"VLLM_API_KEY={API_KEY}")
            print("=" * 65 + "\n")
            break
    return proc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Uncensored Model Runner")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=[
            "cognitivecomputations/dolphin-2.9.3-qwen2-1.5b",
            "failspy/Llama-3.2-3B-Instruct-abliterated",
            "NousResearch/Hermes-3-Llama-3.1-8B",
            "failspy/Meta-Llama-3.1-8B-Instruct-abliterated",
            "cognitivecomputations/dolphin-2.9.2-qwen2-7b",
        ],
        help="Uncensored model identifier on Hugging Face",
    )
    parser.add_argument("--port", type=str, default=PORT, help="Port to serve on")
    parser.add_argument("--api-key", type=str, default=API_KEY, help="API key for vLLM auth")
    parser.add_argument("--skip-install", action="store_true", help="Skip package installation")

    args = parser.parse_args()

    if not args.skip_install:
        install_dependencies()

    vllm_proc = start_vllm(args.model, args.port, args.api_key)
    tunnel_proc = start_tunnel(args.port)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTerminating processes...")
        vllm_proc.terminate()
        tunnel_proc.terminate()
