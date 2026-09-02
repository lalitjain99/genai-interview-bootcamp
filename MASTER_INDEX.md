# 🚀 GenAI Interview Bootcamp
### From Software Engineer → Senior / Staff / Lead GenAI Engineer

> **Don't memorize AI. Understand it deeply enough that you could have invented it.**

---

## 🎯 What this Bootcamp Is About

This repository is a **first-principles roadmap for GenAI interviews**. Every major idea is introduced as an engineering problem, followed by the limitation of the existing solution and the reasoning that leads to the next one.

```text
Problem
  ↓
Existing Solution
  ↓
Limitation
  ↓
New Idea
  ↓
Intuition
  ↓
Mathematics
  ↓
Engineering & Interview Reasoning
```

### 📚 Standard Topic Structure

| File | Purpose |
|---|---|
| 📘 `01_Lecture.md` | Story-driven learning from first principles |
| 📝 `02_Revision.md` | Fast revision notes |
| 🎤 `03_Interview.md` | Interview questions from beginner → Staff level |
| 💡 `04_Assignment.md` | Applied reasoning and debugging |
| ⚡ `05_CheatSheet.md` | High-yield last-minute revision |
| 🚫 `06_Common_Mistakes.md` | Misconceptions and interview traps |

> Some smaller conceptual topics may intentionally use fewer files.

---

# 📍 Current Position

| | Status |
|---|---|
| 🧠 Deep Learning Foundations | ✅ Complete |
| 🔁 Sequence-to-Sequence Models | ✅ Complete |
| 🔦 Attention | 🟨 In Progress |
| 📖 Current Chapter | **02 — Attention Mechanism** |
| ⏭️ Next Chapter | **03 — Bahdanau / Additive Attention** |
| 🧱 Transformers | ⬜ Not Started |
| 🚀 Overall Bootcamp | 🟨 In Progress |

### 🧭 Where We Are in the Story

```text
RNN
 ↓
LSTM / GRU
 ↓
Encoder–Decoder
 ↓
Seq2Seq
 ↓
Fixed Context Bottleneck
 ↓
Teacher Forcing
 ↓
Autoregressive Inference
 ↓
Beam Search
 ↓
Attention Motivation
 ↓
👉 Attention Mechanism  ← CURRENT
 ↓
Bahdanau Attention
 ↓
Luong Attention
 ↓
Attention Limitations
 ↓
Transformers
```

### Progress Legend

| Symbol | Meaning |
|---|---|
| ✅ | Complete |
| 🟨 | In Progress |
| ⬜ | Not Started |
| ➖ | Intentionally skipped / covered elsewhere |

---

# 🗺️ Course Roadmap

> Modules are collapsed by default so the roadmap stays readable. Open any module to see its detailed syllabus.

<details>
<summary><strong>✅ Module 1 — Deep Learning Foundations</strong></summary>

> Build the neural-network foundations required to understand modern sequence models and LLMs.

**Status:** ✅ Complete  
**Difficulty:** ⭐⭐⭐⭐☆

| # | Topic | Status |
|---|---|---|
| 01 | Perceptron | ✅ |
| 02 | Hidden Layers & XOR | ✅ |
| 03 | Activation Functions | ✅ |
| 04 | Training Neural Networks | ✅ |
| 05 | Optimizers | ✅ |
| 06 | Generalization & Regularization | ✅ |
| 07 | Weight Initialization | ✅ |
| 08 | Residual Connections | ✅ |
| 09 | Universal Approximation Theorem | ✅ |
| 10 | Convolutional Neural Networks | ✅ |

**Core progression**

```text
Perceptron
→ Non-Linearity
→ Hidden Layers
→ Loss & Backpropagation
→ Optimization
→ Generalization
→ Stable Initialization
→ Deep Networks
→ CNNs
```

</details>

<details>
<summary><strong>✅ Module 2 — Sequence-to-Sequence Models</strong></summary>

> Understand how neural networks model ordered data and generate variable-length sequences.

**Status:** ✅ Complete  
**Difficulty:** ⭐⭐⭐⭐☆

| # | Topic | Status |
|---|---|---|
| 01 | Why Sequence Models | ✅ |
| 02 | Recurrent Neural Network (RNN) | ✅ |
| 03 | Backpropagation Through Time (BPTT) | ✅ |
| 04 | Vanishing & Exploding Gradients | ✅ |
| 05 | LSTM | ✅ |
| 06 | GRU | ✅ |
| 07 | Bidirectional RNN | ✅ |
| 08 | Encoder–Decoder Architecture | ✅ |
| 09 | Seq2Seq Model | ✅ |
| 10 | Context Vector Bottleneck | ✅ |
| 11 | Teacher Forcing | ✅ |
| 12 | Inference & Autoregressive Decoding | ✅ |
| 13 | Beam Search | ✅ |
| ⭐ | Module Summary | ✅ |

**Core progression**

```text
Need Sequence Memory
→ RNN
→ Long-Term Dependency Problem
→ LSTM / GRU
→ Encoder–Decoder
→ Seq2Seq
→ Fixed Context Bottleneck
→ Training with Teacher Forcing
→ Autoregressive Inference
→ Beam Search
→ Source Access Still Needs Improvement
```

</details>

<details open>
<summary><strong>🟨 Module 3 — Attention</strong></summary>

> Replace one fixed source summary with dynamic, relevance-based source access.

**Status:** 🟨 In Progress  
**Difficulty:** ⭐⭐⭐⭐☆

| # | Topic | Status |
|---|---|---|
| 01 | Attention Motivation | ✅ |
| 02 | Attention Mechanism | 🟨 **CURRENT** |
| 03 | Bahdanau / Additive Attention | ⬜ **NEXT** |
| 04 | Luong / Multiplicative Attention | ⬜ |
| 05 | Attention Alignment & Visualization | ⬜ |
| 06 | Attention Limitations | ⬜ |
| ⭐ | Module Summary | ⬜ |

**Current conceptual bridge**

```text
Fixed Context
↓
Keep All Encoder States
↓
Compare Decoder Need with Every Source State
↓
Compatibility Scores
↓
Softmax
↓
Attention Weights
↓
Weighted Context c_t
```

> **Important boundary:** Self-Attention, Query/Key/Value, Scaled Dot-Product Attention and Multi-Head Attention belong to the **Transformer module**, not this classical Attention module.

</details>

---

# 🚀 Transformers & GenAI Modules

> From this point onward, **Transformers start as a separate top-level learning stage rather than remaining inside Deep Learning**.

<details>
<summary><strong>⬜ Module 4 — Transformer Architecture</strong></summary>

> Understand why recurrence became the next bottleneck and how Transformers rebuilt sequence modeling around attention.

**Difficulty:** ⭐⭐⭐⭐⭐

| # | Topic | Status |
|---|---|---|
| 01 | Why Transformers Were Invented | ⬜ |
| 02 | Self-Attention | ⬜ |
| 03 | Query, Key, Value | ⬜ |
| 04 | Scaled Dot-Product Attention | ⬜ |
| 05 | Multi-Head Attention | ⬜ |
| 06 | Positional Encoding | ⬜ |
| 07 | Feed-Forward Networks | ⬜ |
| 08 | Layer Normalization | ⬜ |
| 09 | Residual Connections in Transformers | ⬜ |
| 10 | Transformer Encoder | ⬜ |
| 11 | Transformer Decoder | ⬜ |
| 12 | Encoder–Decoder Architecture | ⬜ |
| 13 | Causal Attention | ⬜ |
| 14 | GPT | ⬜ |
| 15 | BERT | ⬜ |
| 16 | T5 | ⬜ |
| 17 | LLaMA | ⬜ |
| ⭐ | Module Summary | ⬜ |

**Planned progression**

```text
Why Recurrence Is Limiting
→ Self-Attention
→ Q / K / V
→ Scaled Dot Product
→ Multi-Head Attention
→ Position Information
→ Transformer Blocks
→ Encoder / Decoder
→ GPT / BERT / T5 / LLaMA
```

</details>

<details>
<summary><strong>⬜ Module 5 — Large Language Models</strong></summary>

> Move from Transformer architecture to how modern LLMs are trained, adapted, and served efficiently.

- Tokenization
- Embeddings
- Pretraining
- Fine-Tuning
- Instruction Tuning
- RLHF
- PPO
- DPO
- LoRA
- QLoRA
- PEFT
- Quantization
- KV Cache
- Speculative Decoding
- Mixture of Experts (MoE)

</details>

<details>
<summary><strong>⬜ Module 6 — Retrieval-Augmented Generation (RAG)</strong></summary>

> Build production-grade systems that ground model generation in external knowledge.

- Embeddings Deep Dive
- Chunking Strategies
- Vector Databases
- Similarity Search
- Approximate Nearest Neighbor Search
- BM25 & Hybrid Search
- Cross Encoders & Re-ranking
- End-to-End RAG
- Advanced RAG
- Graph RAG
- Agentic RAG
- RAG Evaluation
- Hallucination Detection

</details>

<details>
<summary><strong>⬜ Module 7 — Agentic AI & MCP</strong></summary>

> Move from single model calls to systems that reason, plan, use tools, and collaborate.

- Function Calling
- Tool Calling
- Model Context Protocol (MCP)
- AI Agents
- Planning
- Reflection
- Memory
- Multi-Agent Systems
- LangGraph
- CrewAI
- AutoGen

</details>

<details>
<summary><strong>⬜ Module 8 — Prompt Engineering & Evaluation</strong></summary>

> Learn how to control model behavior and systematically measure quality.

- Prompt Engineering
- Chain of Thought concepts
- ReAct
- Structured Output
- Prompt Evaluation
- LLM-as-a-Judge
- BLEU
- ROUGE
- BERTScore
- RAGAS
- DeepEval
- Human Evaluation

</details>

<details>
<summary><strong>⬜ Module 9 — Production GenAI</strong></summary>

> Learn the engineering required to operate GenAI systems reliably at scale.

- FastAPI for LLM APIs
- vLLM
- TensorRT-LLM
- Ray Serve
- Kubernetes Deployment
- GPU Optimization
- Caching
- Monitoring & Logging
- Guardrails
- Cost Optimization
- Security
- Observability

</details>

<details>
<summary><strong>⬜ Module 10 — GenAI System Design</strong></summary>

> Senior / Staff / Principal-level architecture preparation.

- ChatGPT System Design
- Enterprise RAG Architecture
- Multi-Tenant RAG
- AI Coding Assistant
- AI Search Engine
- AI Customer Support Platform
- AI Document Intelligence
- AI Agent Platform
- AI Memory Systems
- LLMOps

</details>

---

# 🧠 The Full Learning Journey

```text
Deep Learning Foundations
        ↓
Sequence Models
        ↓
Classic Attention
        ↓
Transformers
        ↓
Large Language Models
        ↓
RAG
        ↓
Agentic AI / MCP
        ↓
Evaluation
        ↓
Production GenAI
        ↓
GenAI System Design
```

---

# 🎯 End Goal

By the end of the bootcamp, the goal is to be able to:

- derive core Deep Learning and Transformer concepts from first principles;
- explain not only **how** an architecture works, but **why it had to be invented**;
- reason about LLM training, inference, RAG, agents, evaluation, and production trade-offs;
- design production-grade GenAI systems;
- handle Senior / Staff / Lead GenAI interview discussions with architectural and mathematical depth.

---

> ### ⭐ Bootcamp Principle
> **Problem → Limitation → Insight → Mathematics → Architecture → Production Reasoning**
