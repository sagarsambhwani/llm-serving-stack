import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
import numpy as np
from rich.console import Console
from rich.table import Table

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add workspace root to sys.path to import scenarios
sys.path.insert(0, str(Path(__file__).parent.parent))
from loadtest.scenarios import (
    generate_long_prompts,
    generate_prefix_caching_prompts,
    generate_short_prompts,
)

console = Console(highlight=False)


@dataclass
class RequestMetric:
    success: bool
    status_code: int
    latency: float  # seconds
    ttft: Optional[float] = None  # time to first token in seconds
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None


@dataclass
class ConcurrencySummary:
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    requests_per_sec: float
    output_tokens_per_sec: float
    total_tokens_per_sec: float
    avg_latency_s: float
    p50_latency_s: float
    p90_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    avg_ttft_s: Optional[float] = None
    p50_ttft_s: Optional[float] = None
    p95_ttft_s: Optional[float] = None


async def send_single_request(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    stream: bool,
) -> RequestMetric:
    start_time = time.perf_counter()
    first_token_time = None
    prompt_tokens = 0
    completion_tokens = 0

    try:
        if stream:
            payload["stream"] = True
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    err_body = await resp.aread()
                    return RequestMetric(
                        success=False,
                        status_code=resp.status_code,
                        latency=time.perf_counter() - start_time,
                        error=err_body.decode("utf-8", errors="ignore")[:100],
                    )

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        content = line[6:].strip()
                        if content == "[DONE]":
                            break
                        if first_token_time is None:
                            first_token_time = time.perf_counter() - start_time
                        try:
                            chunk = json.loads(content)
                            if "usage" in chunk and chunk["usage"]:
                                prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
                                completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)
                            else:
                                # Estimate token increment if usage not in chunk
                                completion_tokens += 1
                        except Exception:
                            completion_tokens += 1

            total_latency = time.perf_counter() - start_time
            return RequestMetric(
                success=True,
                status_code=200,
                latency=total_latency,
                ttft=first_token_time or total_latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=max(1, completion_tokens),
            )
        else:
            payload["stream"] = False
            resp = await client.post(url, json=payload, headers=headers)
            total_latency = time.perf_counter() - start_time

            if resp.status_code != 200:
                return RequestMetric(
                    success=False,
                    status_code=resp.status_code,
                    latency=total_latency,
                    error=resp.text[:100],
                )

            data = resp.json()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            return RequestMetric(
                success=True,
                status_code=200,
                latency=total_latency,
                ttft=total_latency,  # For non-streaming, TTFT is total latency
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

    except Exception as e:
        return RequestMetric(
            success=False,
            status_code=0,
            latency=time.perf_counter() - start_time,
            error=str(e)[:100],
        )


async def run_benchmark_tier(
    concurrency: int,
    num_requests: int,
    endpoint_url: str,
    api_key: str,
    model: str,
    prompts: List[List[Dict[str, str]]],
    max_tokens: int,
    stream: bool,
) -> ConcurrencySummary:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    semaphore = asyncio.Semaphore(concurrency)
    metrics: List[RequestMetric] = []

    async with httpx.AsyncClient(timeout=180.0) as client:
        async def worker(prompt_idx: int):
            async with semaphore:
                prompt_messages = prompts[prompt_idx % len(prompts)]
                payload = {
                    "model": model,
                    "messages": prompt_messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                }
                m = await send_single_request(
                    client,
                    f"{endpoint_url.rstrip('/')}/chat/completions",
                    headers,
                    payload,
                    stream=stream,
                )
                metrics.append(m)

        tier_start = time.perf_counter()
        tasks = [asyncio.create_task(worker(i)) for i in range(num_requests)]
        await asyncio.gather(*tasks)
        tier_duration = time.perf_counter() - tier_start

    successful = [m for m in metrics if m.success]
    failed = [m for m in metrics if not m.success]

    latencies = [m.latency for m in successful] if successful else [0.0]
    ttfts = [m.ttft for m in successful if m.ttft is not None]

    total_out_tokens = sum(m.completion_tokens for m in successful)
    total_in_tokens = sum(m.prompt_tokens for m in successful)

    req_per_sec = len(successful) / tier_duration if tier_duration > 0 else 0
    out_tok_per_sec = total_out_tokens / tier_duration if tier_duration > 0 else 0
    tot_tok_per_sec = (total_out_tokens + total_in_tokens) / tier_duration if tier_duration > 0 else 0

    return ConcurrencySummary(
        concurrency=concurrency,
        total_requests=num_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        duration_seconds=round(tier_duration, 2),
        requests_per_sec=round(req_per_sec, 2),
        output_tokens_per_sec=round(out_tok_per_sec, 2),
        total_tokens_per_sec=round(tot_tok_per_sec, 2),
        avg_latency_s=round(float(np.mean(latencies)), 3),
        p50_latency_s=round(float(np.percentile(latencies, 50)), 3),
        p90_latency_s=round(float(np.percentile(latencies, 90)), 3),
        p95_latency_s=round(float(np.percentile(latencies, 95)), 3),
        p99_latency_s=round(float(np.percentile(latencies, 99)), 3),
        avg_ttft_s=round(float(np.mean(ttfts)), 3) if ttfts else None,
        p50_ttft_s=round(float(np.percentile(ttfts, 50)), 3) if ttfts else None,
        p95_ttft_s=round(float(np.percentile(ttfts, 95)), 3) if ttfts else None,
    )


async def main():
    parser = argparse.ArgumentParser(description="LLM Inference Async Load-Test & Benchmark Suite")
    parser.add_argument("--target-url", type=str, default="http://localhost:9000/v1", help="Target API endpoint (e.g. http://localhost:9000/v1 or http://localhost:8000/v1)")
    parser.add_argument("--api-key", type=str, default="sk-antigravity-dev-key", help="API key for authentication")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Target model name")
    parser.add_argument("--concurrency-levels", type=str, default="1,2,4,8,16", help="Comma-separated concurrency tiers to test")
    parser.add_argument("--requests-per-level", type=int, default=20, help="Number of requests per concurrency tier")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max completion tokens per request")
    parser.add_argument("--scenario", type=str, choices=["short", "long", "prefix_cache"], default="short", help="Workload scenario")
    parser.add_argument("--stream", action="store_true", default=True, help="Enable SSE streaming to accurately record TTFT")
    parser.add_argument("--output-json", type=str, default="loadtest/results/benchmark_results.json", help="Path to save benchmark JSON results")

    args = parser.parse_args()
    concurrency_list = [int(c.strip()) for c in args.concurrency_levels.split(",") if c.strip()]

    console.print(f"[bold cyan]=== Starting LLM Inference Benchmark ===[/bold cyan]")
    console.print(f"  Target URL: [green]{args.target_url}[/green]")
    console.print(f"  Model: [yellow]{args.model}[/yellow]")
    console.print(f"  Scenario: [magenta]{args.scenario}[/magenta]")
    console.print(f"  Concurrency Tiers: {concurrency_list}")
    console.print(f"  Requests Per Tier: {args.requests_per_level}")
    console.print(f"  Max Output Tokens: {args.max_tokens}")
    console.print(f"  Streaming (TTFT tracking): {args.stream}\n")

    # Generate prompts based on scenario
    if args.scenario == "short":
        prompts = generate_short_prompts(args.requests_per_level)
    elif args.scenario == "long":
        prompts = generate_long_prompts(args.requests_per_level)
    else:
        prompts = generate_prefix_caching_prompts(args.requests_per_level)

    results: List[ConcurrencySummary] = []

    for c in concurrency_list:
        console.print(f"[bold blue]Testing Concurrency = {c} ({args.requests_per_level} requests)...[/bold blue]")
        summary = await run_benchmark_tier(
            concurrency=c,
            num_requests=args.requests_per_level,
            endpoint_url=args.target_url,
            api_key=args.api_key,
            model=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            stream=args.stream,
        )
        results.append(summary)
        console.print(f"  [+] Done in {summary.duration_seconds}s | Req/s: [bold]{summary.requests_per_sec}[/bold] | Out Tokens/s: [bold green]{summary.output_tokens_per_sec}[/bold green] | P95 Latency: {summary.p95_latency_s}s | P95 TTFT: {summary.p95_ttft_s}s\n")

    # Print summary table
    table = Table(title="[Benchmark Results Summary]", show_header=True, header_style="bold magenta")
    table.add_column("Concurrency", justify="right", style="cyan")
    table.add_column("Req/s", justify="right", style="bold green")
    table.add_column("Out Tok/s", justify="right", style="bold yellow")
    table.add_column("Avg Latency (s)", justify="right")
    table.add_column("P50 Latency (s)", justify="right")
    table.add_column("P95 Latency (s)", justify="right")
    table.add_column("P99 Latency (s)", justify="right")
    table.add_column("P50 TTFT (s)", justify="right", style="magenta")
    table.add_column("P95 TTFT (s)", justify="right", style="magenta")
    table.add_column("Success / Total", justify="center")

    for r in results:
        table.add_row(
            str(r.concurrency),
            f"{r.requests_per_sec:.2f}",
            f"{r.output_tokens_per_sec:.2f}",
            f"{r.avg_latency_s:.3f}",
            f"{r.p50_latency_s:.3f}",
            f"{r.p95_latency_s:.3f}",
            f"{r.p99_latency_s:.3f}",
            f"{r.p50_ttft_s:.3f}" if r.p50_ttft_s is not None else "N/A",
            f"{r.p95_ttft_s:.3f}" if r.p95_ttft_s is not None else "N/A",
            f"{r.successful_requests}/{r.total_requests}",
        )

    console.print(table)

    # Save results to JSON
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    console.print(f"\n[green]Results saved to {out_path}[/green]")
    console.print("Run [bold cyan]python loadtest/visualize.py[/bold cyan] to generate visual comparison charts.")


if __name__ == "__main__":
    asyncio.run(main())
