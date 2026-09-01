"""
Example script demonstrating standard OpenAI Python SDK usage against our FastAPI vLLM Gateway.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Point OpenAI client to our custom FastAPI gateway (or directly to vLLM)
client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="sk-antigravity-dev-key",  # Or your generated key from /admin/keys
)

MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def test_non_streaming():
    print("\n--- 1. Testing Non-Streaming Chat Completion ---")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful and concise AI assistant."},
                {"role": "user", "content": "Explain KV cache in 2 sentences."}
            ],
            max_tokens=60,
            temperature=0.7,
        )
        print("Response:", response.choices[0].message.content)
        print("Usage:", response.usage)
    except Exception as e:
        print(f"Error: {e}")


def test_streaming():
    print("\n--- 2. Testing Streaming (SSE) Chat Completion ---")
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert AI tutor."},
                {"role": "user", "content": "What is PagedAttention in vLLM?"}
            ],
            max_tokens=100,
            stream=True,
        )
        print("Stream Output: ", end="", flush=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_non_streaming()
    test_streaming()
