"""
Interactive Terminal Chatbot powered by vLLM (Qwen 2.5 GPU) & FastAPI Gateway.
Supports multi-turn conversation memory and real-time token streaming.
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration
BASE_URL = os.getenv("GATEWAY_URL", "http://localhost:9000/v1")
API_KEY = os.getenv("GATEWAY_API_KEY", "sk-antigravity-dev-key")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# ANSI Color Codes for sleek terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner():
    print(f"{CYAN}{BOLD}" + "=" * 65 + f"{RESET}")
    print(f"{CYAN}{BOLD}  🤖 Interactive vLLM Terminal Chatbot{RESET}")
    print(f"{DIM}  Model:    {MODEL}{RESET}")
    print(f"{DIM}  Endpoint: {BASE_URL}{RESET}")
    print(f"{DIM}  Auth:     Bearer {API_KEY[:14]}...{RESET}")
    print(f"{CYAN}{BOLD}" + "=" * 65 + f"{RESET}")
    print(f"{YELLOW}Commands:{RESET}")
    print(f"  {BOLD}/clear{RESET}       - Reset conversation history")
    print(f"  {BOLD}/system <msg>{RESET} - Change assistant persona")
    print(f"  {BOLD}/exit{RESET}        - Quit chat\n")


def main():
    print_banner()

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    system_prompt = "You are a helpful, insightful, and concise AI assistant."
    history = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            # User prompt input
            user_input = input(f"{GREEN}{BOLD}You > {RESET}").strip()

            if not user_input:
                continue

            # Command handling
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                print(f"\n{CYAN}Goodbye! 👋{RESET}")
                break

            if user_input.lower() == "/clear":
                history = [{"role": "system", "content": system_prompt}]
                print(f"{YELLOW}🧹 Conversation history cleared.{RESET}\n")
                continue

            if user_input.startswith("/system "):
                system_prompt = user_input[8:].strip()
                history = [{"role": "system", "content": system_prompt}]
                print(f"{YELLOW}⚙️ System prompt updated: \"{system_prompt}\"{RESET}\n")
                continue

            # Append user message
            history.append({"role": "user", "content": user_input})

            # Stream assistant response
            print(f"{CYAN}{BOLD}AI  > {RESET}", end="", flush=True)

            stream = client.chat.completions.create(
                model=MODEL,
                messages=history,
                temperature=0.7,
                max_tokens=512,
                stream=True,
            )

            assistant_reply = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    print(token, end="", flush=True)
                    assistant_reply.append(token)

            print("\n")

            # Save assistant reply to conversation history
            full_reply = "".join(assistant_reply)
            history.append({"role": "assistant", "content": full_reply})

        except KeyboardInterrupt:
            print(f"\n\n{CYAN}Session ended. Goodbye! 👋{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}{BOLD}Error:{RESET} {e}\n")


if __name__ == "__main__":
    main()
