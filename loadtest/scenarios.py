import random
from typing import List, Dict

# Synthetic base texts for generating prompt variations
SYSTEM_PROMPT_PREFIX = """You are a high-performance, precise AI assistant specializing in software engineering, distributed systems, and machine learning infrastructure. Always provide clear, structured, and insightful explanations with relevant technical context."""

SAMPLE_TOPICS = [
    "Explain how PagedAttention solves KV cache memory fragmentation in vLLM.",
    "Compare continuous batching with traditional static batching in LLM inference servers.",
    "How does FlashAttention optimize GPU SRAM memory access patterns?",
    "What are the latency vs throughput trade-offs when increasing concurrency in an inference engine?",
    "Explain the role of prefix caching in multi-turn conversations and agentic workflows.",
    "How does Tensor Parallelism differ from Pipeline Parallelism when serving 70B parameter models?",
    "Describe the difference between prefill phase (compute-bound) and decode phase (memory-bandwidth-bound).",
    "What are the benefits of AWQ and FP8 quantization for GPU memory footprint and memory bandwidth?",
]


def generate_short_prompts(count: int = 20) -> List[List[Dict[str, str]]]:
    """Generate short prompts (~20-50 tokens)."""
    prompts = []
    for i in range(count):
        topic = SAMPLE_TOPICS[i % len(SAMPLE_TOPICS)]
        prompts.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Briefly explain in 2-3 sentences: {topic}"}
        ])
    return prompts


def generate_long_prompts(count: int = 20, target_words: int = 300) -> List[List[Dict[str, str]]]:
    """Generate long prompts (~400-800 tokens) to stress prefill compute."""
    base_context = (
        "Background Context on Distributed Inference Systems:\n"
        "Large language models (LLMs) have transformed generative AI, but their serving infrastructure "
        "imposes extreme computational and memory bandwidth demands. During inference, memory is consumed "
        "by both the model parameters and the dynamically allocated Key-Value (KV) cache. "
        "Traditional serving frameworks pre-allocated contiguous memory blocks for the maximum possible sequence "
        "length, leading to severe internal and external memory fragmentation (often wasting over 60-80% of VRAM). "
        "vLLM introduced PagedAttention, an algorithm inspired by virtual memory paging in operating systems, "
        "allowing non-contiguous memory allocation of KV cache blocks. This drastically improves memory utilization "
        "and enables higher batch sizes, leading to significantly higher request throughput.\n\n"
    ) * (target_words // 50)

    prompts = []
    for i in range(count):
        topic = SAMPLE_TOPICS[i % len(SAMPLE_TOPICS)]
        prompts.append([
            {"role": "system", "content": "You are an expert distributed systems architect."},
            {"role": "user", "content": f"{base_context}\n\nBased on the above background context, please provide a detailed analysis of the following question:\n{topic}"}
        ])
    return prompts


def generate_prefix_caching_prompts(count: int = 20) -> List[List[Dict[str, str]]]:
    """
    Generate prompts that share a heavy common prefix (e.g. 500+ tokens system instructions)
    followed by unique user queries. This tests vLLM's automatic prefix caching efficiency.
    """
    shared_system_prompt = (
        SYSTEM_PROMPT_PREFIX + "\n" +
        "Guidelines:\n"
        "1. Prioritize algorithmic efficiency and low-latency architectural patterns.\n"
        "2. Break down prefill vs decoding bottlenecks.\n"
        "3. Detail memory layout: HBM, SRAM, KV cache blocks.\n"
        "4. Include concise mathematical intuitions where applicable (FLOPs, Memory Bandwidth in GB/s).\n"
    ) * 4

    prompts = []
    for i in range(count):
        topic = SAMPLE_TOPICS[i % len(SAMPLE_TOPICS)]
        prompts.append([
            {"role": "system", "content": shared_system_prompt},
            {"role": "user", "content": f"Query #{i+1}: {topic}"}
        ])
    return prompts
