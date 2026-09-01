"""
Standalone runner script for Google Colab or remote GPU instance.
Installs cloudflared, launches vLLM with Qwen 2.5, and prints the Cloudflare Quick Tunnel URL.
"""

import os
import re
import subprocess
import sys
import time

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
API_KEY = os.getenv("API_KEY", "dev-secret")
PORT = os.getenv("PORT", "8000")


def run_cmd(cmd, check=True):
    print(f">> Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, check=check)
    return res


def install_dependencies():
    print("📦 Installing vLLM and Cloudflare Tunnel...")
    run_cmd("pip uninstall -y torchaudio")
    run_cmd("pip install -q -U vllm")
    run_cmd("wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb")
    run_cmd("dpkg -i cloudflared-linux-amd64.deb")


def start_vllm():
    print(f"🚀 Starting vLLM server with model {MODEL_NAME}...")
    vllm_cmd = [
        "vllm", "serve", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", PORT,
        "--api-key", API_KEY,
        "--enable-prefix-caching",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
    ]
    proc = subprocess.Popen(vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in proc.stdout:
        print(line, end="")
        if "Application startup complete" in line or "Uvicorn running on" in line:
            print("\n✅ vLLM is UP and accepting requests!\n")
            break
    return proc


def start_tunnel():
    print("🌐 Starting Cloudflare Quick Tunnel...")
    tunnel_cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{PORT}"]
    proc = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in proc.stdout:
        print(line, end="")
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            url = match.group(0)
            print("\n" + "=" * 60)
            print(f"🎉 TUNNEL READY: {url}")
            print(f"VLLM_BASE_URL={url}/v1")
            print("=" * 60 + "\n")
            break
    return proc


if __name__ == "__main__":
    install_dependencies()
    vllm_p = start_vllm()
    tunnel_p = start_tunnel()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTerminating processes...")
        vllm_p.terminate()
        tunnel_p.terminate()
