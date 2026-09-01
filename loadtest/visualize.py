import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_benchmark_results(json_path: str = "loadtest/results/benchmark_results.json", output_img: str = "loadtest/results/benchmark_charts.png"):
    if not os.path.exists(json_path):
        print(f"Error: Results file '{json_path}' not found. Run benchmark.py first.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("Error: No data found in benchmark results.")
        return

    df = pd.DataFrame(data)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("vLLM / Gateway Inference Benchmark Performance", fontsize=16, fontweight="bold")

    # 1. Output Token Throughput vs Concurrency
    axs[0, 0].plot(df["concurrency"], df["output_tokens_per_sec"], marker="o", color="#2ca02c", linewidth=2.5, markersize=8)
    axs[0, 0].set_title("Output Tokens / Second (Continuous Batching Saturation)")
    axs[0, 0].set_xlabel("Concurrency (Concurrent Streams)")
    axs[0, 0].set_ylabel("Tokens / sec")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)
    axs[0, 0].set_xticks(df["concurrency"])

    # 2. Request Throughput vs Concurrency
    axs[0, 1].plot(df["concurrency"], df["requests_per_sec"], marker="s", color="#1f77b4", linewidth=2.5, markersize=8)
    axs[0, 1].set_title("Requests / Second Throughput")
    axs[0, 1].set_xlabel("Concurrency")
    axs[0, 1].set_ylabel("Req / sec")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)
    axs[0, 1].set_xticks(df["concurrency"])

    # 3. End-to-End Latency Percentiles (P50, P95, P99)
    axs[1, 0].plot(df["concurrency"], df["p50_latency_s"], marker="o", label="P50 Latency", color="#3498db")
    axs[1, 0].plot(df["concurrency"], df["p90_latency_s"], marker="^", label="P90 Latency", color="#f39c12")
    axs[1, 0].plot(df["concurrency"], df["p95_latency_s"], marker="x", label="P95 Latency", color="#e74c3c")
    axs[1, 0].plot(df["concurrency"], df["p99_latency_s"], marker="d", label="P99 Latency", color="#8e44ad")
    axs[1, 0].set_title("End-to-End Latency vs Concurrency")
    axs[1, 0].set_xlabel("Concurrency")
    axs[1, 0].set_ylabel("Latency (seconds)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)
    axs[1, 0].set_xticks(df["concurrency"])

    # 4. Time To First Token (TTFT)
    if "p50_ttft_s" in df.columns and df["p50_ttft_s"].notna().any():
        axs[1, 1].plot(df["concurrency"], df["p50_ttft_s"], marker="o", label="P50 TTFT", color="#1abc9c")
        axs[1, 1].plot(df["concurrency"], df["p95_ttft_s"], marker="x", label="P95 TTFT", color="#e67e22")
        axs[1, 1].set_title("Time to First Token (TTFT / Prefill Delay)")
        axs[1, 1].set_xlabel("Concurrency")
        axs[1, 1].set_ylabel("TTFT (seconds)")
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle="--", alpha=0.6)
        axs[1, 1].set_xticks(df["concurrency"])
    else:
        axs[1, 1].text(0.5, 0.5, "TTFT data not available\n(Run benchmark with --stream)", ha="center", va="center")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_file = Path(output_img)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    print(f"Chart successfully saved to {out_file}")


if __name__ == "__main__":
    json_input = sys.argv[1] if len(sys.argv) > 1 else "loadtest/results/benchmark_results.json"
    plot_benchmark_results(json_input)
