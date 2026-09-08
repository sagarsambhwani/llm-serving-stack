# 📊 Open-Weights LLM & VLM Model Reference Matrix

A comprehensive reference guide comparing performance benchmarks, parameter counts, context windows, VRAM hardware footprints, licensing restrictions, and best use cases for models compatible with this serving stack.

---

## 🧭 Executive Model Selection Matrix

| Model Identifier | Parameter Count | Native Context | Precision & VRAM Footprint (Weights Only) | T4 (16GB) Colab Fit | License & Gating | Primary Strength / Specialty |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Qwen2.5-1.5B-Instruct** | 1.54B | 32k (128k max) | FP16: **3.0 GB** | ⭐⭐⭐⭐⭐ (Pure FP16, 12GB KV-cache) | **Apache 2.0** (Ungated) | Best 1.5B coding, multilingual vocab (152k tokens) |
| **Llama-3.2-1B-Instruct** | 1.23B | 128k | FP16: **2.5 GB** | ⭐⭐⭐⭐⭐ (Pure FP16, 12.5GB KV-cache) | **Llama 3.2 Community** (Gated) | Ultra-low latency, fast token classification & routing |
| **Llama-3.2-3B-Instruct** | 3.21B | 128k | FP16: **6.4 GB** | ⭐⭐⭐⭐⭐ (Pure FP16, 8.5GB KV-cache) | **Llama 3.2 Community** (Gated) | SOTA tool calling, general chat, agentic orchestration |
| **Phi-3.5-mini-instruct** | 3.82B | 128k | FP16: **7.6 GB** | ⭐⭐⭐⭐⭐ (Pure FP16, 7.5GB KV-cache) | **MIT License** (Ungated) | Highest reasoning, logic, and math density in $<4\text{B}$ |
| **Gemma-2-2b-it** | 2.61B | 8k | FP16: **5.2 GB** | ⭐⭐⭐⭐⭐ (Pure FP16, 9.8GB KV-cache) | **Gemma Terms** (Gated) | Gemini-distilled prose, factual summarization |
| **SmolLM2-1.7B-Instruct** | 1.71B | 8k | FP16: **3.4 GB** | ⭐⭐⭐⭐⭐ (Pure FP16, 11.6GB KV-cache) | **Apache 2.0** (Ungated) | Clean curated dataset, on-device agent prototyping |
| **Qwen2.5-7B-Instruct** | 7.61B | 128k | FP16: 15.2 GB / AWQ: **4.8 GB** | ⭐⭐⭐⭐ (Requires AWQ on T4) | **Apache 2.0** (Ungated) | SOTA 7B code generation, math, and JSON extraction |
| **Llama-3.1-8B-Instruct** | 8.03B | 128k | FP16: 16.1 GB / AWQ: **5.5 GB** | ⭐⭐⭐⭐ (Requires AWQ on T4) | **Llama 3.1 Community** (Gated) | Industry standard 8B model, 128k multi-turn reasoning |
| **Mistral-7B-Instruct-v0.3**| 7.25B | 32k | FP16: 14.5 GB / AWQ: **4.9 GB** | ⭐⭐⭐⭐ (Requires AWQ on T4) | **Apache 2.0** (Ungated) | Native function calling, robust JSON schemas |
| **Qwen2-VL-2B-Instruct** (VLM)| 2.21B | 32k | FP16: **4.5 GB** | ⭐⭐⭐⭐⭐ (Pure FP16, 10.5GB KV-cache) | **Apache 2.0** (Ungated) | High-res image/video QA, document OCR, chart analysis |
| **Phi-3.5-vision-instruct** (VLM)| 4.15B | 128k | FP16: **8.3 GB** | ⭐⭐⭐⭐⭐ (Pure FP16, 6.7GB KV-cache) | **MIT License** (Ungated) | Multi-frame image reasoning, table extraction, OCR |
| **MiniCPM-V-2_6** (VLM) | 8.00B | 32k | FP16: 16.0 GB / AWQ: **5.8 GB** | ⭐⭐⭐⭐ (Requires AWQ on T4) | **Apache 2.0** (Ungated) | SOTA open vision benchmarks (beats GPT-4V on OCR) |
| **Dolphin-2.9.3-Qwen2-1.5B** (Uncensored)| 1.54B | 32k | FP16: **3.0 GB** | ⭐⭐⭐⭐⭐ (Pure FP16, 12GB KV-cache) | **Apache 2.0** (Ungated) | Uncensored, unfiltered coding, raw technical assistance |
| **Hermes-3-Llama-3.1-8B** (Uncensored) | 8.03B | 128k | FP16: 16.1 GB / AWQ: **5.5 GB** | ⭐⭐⭐⭐ (Requires AWQ on T4) | **Apache 2.0** (Ungated) | Flagship steerable model, zero refusal, deep reasoning |
| **Llama-3.1-8B-Abliterated** (Uncensored)| 8.03B | 128k | FP16: 16.1 GB / AWQ: **5.5 GB** | ⭐⭐⭐⭐ (Requires AWQ on T4) | **Llama 3.1** (Ungated Download) | Refusal-vector orthogonalized, zero benchmark degradation |
| **Llama-3.2-3B-Abliterated** (Uncensored)| 3.21B | 128k | FP16: **6.4 GB** | ⭐⭐⭐⭐⭐ (Pure FP16, 8.5GB KV-cache) | **Llama 3.2** (Ungated Download) | Fast 3B uncensored model for tool calling & agent loops |

---

## 🔍 Detailed Model Profiles

### 1. Ultra-Lightweight Tier (1B – 3B Parameters)

#### A. Qwen 2.5 (1.5B Instruct)
* **Hugging Face ID:** `Qwen/Qwen2.5-1.5B-Instruct`
* **License:** Apache 2.0 (Completely open, commercial use, ungated).
* **Context Window:** Up to $128\text{k}$ tokens ($32\text{k}$ default).
* **Benchmark Profile:** MMLU: $\sim 61\%$, HumanEval (Code): $\sim 65\%$.
* **Strengths:** Huge $152\text{k}$ token vocabulary (very fast tokenization of code and non-English text).
* **Serving Command:**
  ```bash
  vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 --gpu-memory-utilization 0.90
  ```

#### B. Meta Llama 3.2 (3B & 1B Instruct)
* **Hugging Face IDs:** `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`
* **License:** Meta Llama 3.2 Community License (Gated on Hugging Face; requires accepting terms. Free commercial use up to 700M monthly active users).
* **Context Window:** Native $128\text{k}$ tokens.
* **Benchmark Profile:** 3B MMLU: $\sim 63.4\%$, GSM8k: $\sim 77.7\%$.
* **Strengths:** Outstanding instruction following, structured tool calling, compact memory footprint.
* **Serving Command:**
  ```bash
  vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8000 --gpu-memory-utilization 0.90
  ```

#### C. Google Gemma 2 (2B IT)
* **Hugging Face ID:** `google/gemma-2-2b-it`
* **License:** Gemma Terms of Use (Gated on Hugging Face; free commercial use with safety restrictions).
* **Context Window:** $8\text{k}$ tokens.
* **Benchmark Profile:** Distilled directly from Gemini Ultra models; exceptional natural language generation.
* **Strengths:** Highest English prose quality and factual summarization in the sub-3B weight class.

---

### 2. High-Reasoning & Dense Tier (3.5B – 4B Parameters)

#### Microsoft Phi-3.5-mini (3.8B Instruct)
* **Hugging Face ID:** `microsoft/Phi-3.5-mini-instruct`
* **License:** **MIT License** (Fully open, ungated, unrestricted commercial use).
* **Context Window:** $128\text{k}$ tokens.
* **Benchmark Profile:** MMLU: $\sim 69\%$, GSM8k (Math): $\sim 86\%$ (approaches 8B model performance).
* **Strengths:** Trained heavily on synthetic "textbook quality" reasoning data. Punches way above its weight for logic, step-by-step reasoning, and Python programming.
* **Serving Command:**
  ```bash
  vllm serve microsoft/Phi-3.5-mini-instruct --trust-remote-code --port 8000
  ```

---

### 3. Workhorse Tier (7B – 9B Parameters)

> [!NOTE]
> On a **16GB Tesla T4 GPU**, 7B–9B models in unquantized FP16 require $14-18\text{ GB}$ just for weights, leaving zero memory for KV-cache. **Always serve them using 4-bit AWQ or GPTQ on T4**, which reduces weight size to $\sim 5\text{ GB}$, leaving **$10\text{ GB}$ for PagedAttention!**

#### A. Meta Llama 3.1 (8B Instruct - AWQ)
* **Hugging Face ID:** `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`
* **License:** Meta Llama 3.1 Community License (Gated).
* **Context Window:** $128\text{k}$ tokens.
* **Strengths:** The industry gold standard for general intelligence, agentic workflows, long-context document retrieval, and complex coding.
* **Serving Command:**
  ```bash
  vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --port 8000 --gpu-memory-utilization 0.90
  ```

#### B. Qwen 2.5 (7B Instruct - AWQ)
* **Hugging Face ID:** `Qwen/Qwen2.5-7B-Instruct-AWQ`
* **License:** Apache 2.0 (Ungated).
* **Context Window:** $128\text{k}$ tokens.
* **Strengths:** Top scoring open 7B model on HumanEval coding benchmarks ($84\%+$) and MATH benchmarks.

---

### 4. Multimodal Vision-Language Tier (VLMs)

#### A. Qwen2-VL (2B Instruct)
* **Hugging Face ID:** `Qwen/Qwen2-VL-2B-Instruct`
* **License:** Apache 2.0 (Ungated).
* **Input Types:** Text + High-Resolution Images + Videos.
* **Architecture:** Naive Dynamic Resolution ViT + Qwen2 1.5B LLM Decoder.
* **T4 Fit:** $\sim 4.5\text{ GB}$ VRAM in FP16. Leaves $>10\text{ GB}$ for KV-cache!
* **Serving Command:**
  ```bash
  vllm serve Qwen/Qwen2-VL-2B-Instruct --port 8000 --limit-mm-per-prompt image=4
  ```

#### B. Microsoft Phi-3.5-Vision (4.2B Instruct)
* **Hugging Face ID:** `microsoft/Phi-3.5-vision-instruct`
* **License:** MIT License (Ungated).
* **Strengths:** Highly optimized for technical diagrams, multi-page PDF documents, financial tables, and chart comprehension.
* **T4 Fit:** $\sim 8.3\text{ GB}$ VRAM in FP16.

---

### 5. Ungated & Uncensored Tier (Zero Gating & No Refusal Filters)

#### A. Dolphin 2.9.3 (Qwen2 1.5B)
* **Hugging Face ID:** `cognitivecomputations/dolphin-2.9.3-qwen2-1.5b`
* **License:** Apache 2.0 (100% Ungated).
* **Strengths:** Ultra-lightweight uncensored model. Fits in **$3.0\text{ GB}$ of VRAM** in pure FP16 on Colab T4. Zero corporate refusal filtering, direct instruction-following.
* **Serving Command:**
  ```bash
  vllm serve cognitivecomputations/dolphin-2.9.3-qwen2-1.5b --port 8000 --gpu-memory-utilization 0.90
  ```

#### B. Hermes 3 (Llama 3.1 8B)
* **Hugging Face ID:** `NousResearch/Hermes-3-Llama-3.1-8B`
* **License:** Apache 2.0 (100% Ungated).
* **Strengths:** Flagship steerable model created by Nous Research. Exceptional at complex multi-turn reasoning, structured JSON outputs, and red-teaming/cybersecurity workflows without preachy refusals.
* **Serving Command:**
  ```bash
  vllm serve NousResearch/Hermes-3-Llama-3.1-8B --port 8000 --gpu-memory-utilization 0.90
  ```

#### C. Meta Llama 3.1 8B Abliterated
* **Hugging Face ID:** `failspy/Meta-Llama-3.1-8B-Instruct-abliterated`
* **License:** Llama 3.1 (Direct ungated download on Hugging Face).
* **Strengths:** Uses mathematical weight orthogonalization to neutralize the refusal feature direction while preserving $100\%$ of base Llama 3.1 reasoning benchmarks.
* **Serving Command:**
  ```bash
  vllm serve failspy/Meta-Llama-3.1-8B-Instruct-abliterated --port 8000 --gpu-memory-utilization 0.90
  ```

---

## 📜 Licensing & Restriction Guide

| License Type | Examples | Commercial Use Allowed? | Gated on HuggingFace? | Restrictions / Conditions |
| :--- | :--- | :---: | :---: | :--- |
| **Apache 2.0 / MIT** | Qwen 2.5, Phi-3.5, Hermes 3, Dolphin | **YES** (Unrestricted) | ❌ **No** (Direct download) | Retain copyright notice. |
| **Meta Llama Community** | Llama 3.1, Llama 3.2 | **YES** | ✅ **Yes** (Accept click-through) | Free if $<700\text{M}$ Monthly Active Users. |
| **Gemma Terms of Use** | Gemma 2 (2B, 9B) | **YES** | ✅ **Yes** (Accept click-through) | Must follow Google prohibited use policy. |
