# 🚀 GenAI Interview Bootcamp

> **A first-principles roadmap from Software Engineer → Senior / Staff / Lead GenAI Engineer.**
>
> **Don't memorize AI. Understand it deeply enough that you could have invented it.**

This repository is my structured learning journey through **Deep Learning, Sequence Models, Attention, Transformers, NLP Foundations, Large Language Model Internals, RAG, Agentic AI, MCP, Evaluation, Production GenAI, and System Design**.

The goal is not only to prepare for interviews. It is to build enough intuition, mathematical understanding, and engineering depth to **design, build, debug, and explain production-grade GenAI systems**.

---

# 🎯 Why This Repository Exists

Many interview-preparation resources start with definitions and finish with a list of questions.

This bootcamp follows a different path:

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
Engineering Reasoning
  ↓
Interview Reasoning
```

Every major topic is learned by asking:

> **What problem were engineers trying to solve, why did the previous approach fail, and what idea naturally came next?**

That is why the course progresses as a connected story rather than a collection of disconnected topics.

---

# 🧠 Learning Philosophy

Three principles are non-negotiable throughout the bootcamp:

- 📖 **Story-based explanations** — understand how the idea emerged.
- 🧠 **Intuition before mathematics** — build the mental model first.
- 📐 **Mathematical foundations** — understand what the equations actually mean.

The deeper objective is:

```text
Don't memorize architecture diagrams.
Understand why each block had to exist.
```

---

# 📍 Current Position

| Area | Status |
|---|---|
| 🧠 Deep Learning Foundations | ✅ Complete |
| 🔁 Sequence-to-Sequence Models | ✅ Complete |
| 🔦 Attention | 🟨 In Progress |
| 📖 Current Chapter | **02 — Attention Mechanism** |
| ⏭️ Next Chapter | **03 — Bahdanau / Additive Attention** |
| 🧱 Transformers | ⬜ Not Started |
| 🚀 Overall Bootcamp | 🟨 In Progress |

### 🧭 Current Learning Story

```text
Perceptron
↓
Hidden-Layer Revolution
↓
Activation Functions
↓
Training & Optimization
↓
Generalization / Initialization / Residual Learning
↓
CNNs
↓
RNN → BPTT → Vanishing / Exploding Gradients
↓
LSTM / GRU / Bidirectional RNN
↓
Encoder–Decoder → Seq2Seq
↓
Context Vector Bottleneck
↓
Teacher Forcing → Autoregressive Inference → Beam Search
↓
Attention Motivation
↓
👉 Attention Mechanism  ← CURRENT
↓
Bahdanau Attention
↓
Luong Attention
↓
Transformers
↓
NLP Foundations
↓
LLM Internals
↓
RAG → Agents / MCP → Evaluation → Production → System Design
```

> 📚 For the complete topic-by-topic learning history, current status, and future syllabus, see **[MASTER_INDEX.md](MASTER_INDEX.md)**.

---

# 🗺️ Bootcamp Roadmap

The phase-based structure is intentionally preserved because it makes the long learning journey easier to reason about. Each phase has a clear purpose and feeds the next one.

## Phase 1 — Deep Learning Foundations ✅

Build the neural-network foundations required for everything that follows.

- Perceptron
- Hidden-Layer Revolution & XOR
- Activation Functions
- Loss Functions
- Gradient Descent & Backpropagation
- Optimizers
- Generalization & Regularization
- Weight Initialization
- Residual Connections
- Universal Approximation Theorem
- Convolutional Neural Networks

---

## Phase 2 — Sequence-to-Sequence Models ✅

Understand how neural networks model ordered data and generate variable-length sequences.

- Why Sequence Models
- RNN
- Backpropagation Through Time
- Vanishing & Exploding Gradients
- LSTM
- GRU
- Bidirectional RNN
- Encoder–Decoder
- Seq2Seq
- Context Vector Bottleneck
- Teacher Forcing
- Autoregressive Decoding
- Beam Search

---

## Phase 3 — Attention 🟨

Solve the fixed-context bottleneck by giving the decoder dynamic access to source representations.

- Attention Motivation ✅
- Attention Mechanism 🟨
- Bahdanau / Additive Attention
- Luong / Multiplicative Attention
- Attention Alignment & Visualization
- Attention Limitations

---

## Phase 4 — Transformer Architecture ⬜

Start a new architectural stage: remove recurrence as the primary sequence-processing mechanism and rebuild sequence modeling around attention.

- Why Transformers Were Invented
- Self-Attention
- Query, Key, Value
- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Feed-Forward Networks
- Layer Normalization
- Residual Connections in Transformers
- Transformer Encoder
- Transformer Decoder
- Encoder–Decoder Transformer
- Causal Attention
- GPT Architecture
- BERT Architecture
- T5 Architecture
- LLaMA Architecture

> This phase focuses primarily on **architecture**: how Transformer blocks work and how major model families are structurally organized.

---

## Phase 5 — NLP Foundations ⬜

Build the language-representation concepts that connect classical NLP, Transformers, modern embeddings, retrieval, and LLM systems.

- Text preprocessing and normalization
- Tokenization fundamentals
- Vocabulary and out-of-vocabulary problems
- One-Hot Encoding
- Bag of Words
- TF-IDF
- Distributional semantics
- Word2Vec
- GloVe
- FastText
- Subword tokenization
- Byte Pair Encoding (BPE)
- WordPiece / SentencePiece concepts
- Contextual embeddings
- Sentence Embeddings
- Bi-Encoder vs Cross-Encoder
- Similarity measures for language representations

**Conceptual progression**

```text
Words as Symbols
↓
Sparse Statistical Representations
↓
Dense Word Embeddings
↓
Subword Representations
↓
Contextual Representations
↓
Sentence / Semantic Representations
```

> NLP Foundations remains a dedicated phase rather than being hidden inside the LLM or RAG modules. These ideas are important enough to study independently and are reused throughout later phases.

---

## Phase 6 — LLM Internals ⬜

Move beyond Transformer architecture and understand how modern Large Language Models are actually trained, aligned, adapted, and served efficiently.

- Language Modeling Objective
- Next-Token Prediction
- Pretraining
- Training Data and Corpus Preparation
- Scaling intuition
- Fine-Tuning
- Instruction Tuning
- Supervised Fine-Tuning (SFT)
- Alignment concepts
- RLHF
- Reward Models
- PPO
- DPO
- Parameter-Efficient Fine-Tuning (PEFT)
- LoRA
- QLoRA
- Quantization
- KV Cache
- Prefill vs Decode
- Sampling and Generation Controls
- Speculative Decoding
- Mixture of Experts (MoE)
- LLM inference efficiency

**Important boundary**

```text
Transformer Phase
→ How the architecture works

NLP Foundations
→ How language is represented

LLM Internals
→ How large Transformer language models are trained, aligned, adapted and run
```

This separation prevents three very large topics from being mixed into one module.

---

## Phase 7 — Retrieval-Augmented Generation (RAG) ⬜

Build systems that ground model generation in external knowledge.

- Embeddings Deep Dive
- Chunking Strategies
- Vector Databases
- Similarity Search
- Approximate Nearest Neighbor Search
- BM25 & Hybrid Search
- Bi-Encoder / Cross-Encoder retrieval
- Re-ranking
- Query Transformation
- Context Compression
- End-to-End RAG
- Advanced RAG
- Graph RAG
- Agentic RAG
- RAG Evaluation
- Hallucination Detection

---

## Phase 8 — Agentic AI & MCP ⬜

Move from single LLM calls to systems that can reason, use tools, plan, maintain state, and collaborate.

- Function Calling
- Tool Calling
- AI Agents
- Planning
- Reflection
- Memory
- Multi-Agent Systems
- Model Context Protocol (MCP)
- MCP Clients / Servers
- Tools, Resources, and Prompts
- Agent orchestration patterns
- LangGraph
- CrewAI
- AutoGen

> MCP is grouped with Agentic AI because it is best understood in the context of tool use, external capabilities, and context exchange in modern AI systems.

---

## Phase 9 — Prompt Engineering & Evaluation ⬜

Learn how to control model behavior and measure whether the system actually works.

- Prompt Design
- Zero-Shot / Few-Shot Prompting
- Chain-of-Thought concepts
- ReAct
- Structured Outputs
- Prompt Evaluation
- LLM-as-a-Judge
- BLEU
- ROUGE
- BERTScore
- RAGAS
- DeepEval
- Hallucination Evaluation
- Human Evaluation

---

## Phase 10 — Production GenAI & Cloud ⬜

Learn the engineering required to run GenAI systems reliably, securely, and economically.

- LLM API serving concepts
- vLLM
- TensorRT-LLM
- Ray Serve
- Kubernetes
- GPU Optimization
- Caching
- Monitoring & Logging
- Observability
- Guardrails
- Security
- Cost Optimization
- Model Deployment
- Azure AI / Azure OpenAI
- AWS Bedrock
- Google Vertex AI

> Cloud remains part of Production GenAI because deployment, observability, scaling, security, and managed AI platforms belong to the same production-engineering problem.

---

## Phase 11 — GenAI System Design ⬜

Bring everything together for Senior / Staff / Principal-level architecture interviews.

- LLD for AI applications
- HLD for GenAI platforms
- ChatGPT-style System Design
- Enterprise RAG
- Multi-Tenant RAG
- AI Coding Assistant
- AI Search Engine
- AI Customer Support Platform
- AI Document Intelligence
- Agent Platforms
- AI Memory Systems
- Scaling LLM Applications
- LLMOps

---

# 📂 Repository Scope

This repository is intentionally focused on **Generative AI learning**.

General Python/backend interview preparation — such as Python internals, DSA, FastAPI, SQL, design patterns, and general backend engineering — will live in a **separate repository**.

This keeps the GenAI bootcamp focused and prevents the curriculum from becoming unnecessarily large.

---

# 📂 Repository Structure

The repository is organized around **major GenAI learning stages as top-level modules**.

```text
genai-interview-bootcamp/
│
├── README.md
├── MASTER_INDEX.md
├── Progress_Tracker.md
├── CHANGELOG.md
│
├── 01_DeepLearning/
│   ├── 01-Perceptron/
│   ├── 02-Hidden Layer Revolution/
│   ├── 03-Activation Functions/
│   ├── ...
│   └── 10_Convolutional_Neural_Network/
│
├── 02_Sequence_to_Sequence_Models/
│   ├── 01_Why_Sequence_Models/
│   ├── 02_Recurrent_Neural_Network/
│   ├── ...
│   └── 13_Beam_Search/
│
├── 03_Attention/
│   ├── 01_Attention_Motivation/
│   └── 02_Attention_Mechanism/
│
└── future top-level GenAI modules are added as the bootcamp progresses
```

The planned top-level progression is:

```text
01_DeepLearning
      ↓
02_Sequence_to_Sequence_Models
      ↓
03_Attention
      ↓
04_Transformers
      ↓
05_NLP_Foundations
      ↓
06_LLM_Internals
      ↓
07_RAG
      ↓
08_Agentic_AI_And_MCP
      ↓
09_Prompt_Engineering_And_Evaluation
      ↓
10_Production_GenAI_And_Cloud
      ↓
11_GenAI_System_Design
```

---

# 📚 Standard Topic Structure

Core topics generally use six learning artifacts:

```text
Topic/
├── 01_Lecture.md
├── 02_Revision.md
├── 03_Interview.md
├── 04_Assignment.md
├── 05_CheatSheet.md
└── 06_Common_Mistakes.md
```

Optional hands-on topics may also contain:

```text
Code.ipynb
```

### 📘 `01_Lecture.md`

Story-driven explanation from first principles:

```text
Problem → Intuition → Mathematics → Architecture → Limitation → Next Idea
```

### 📝 `02_Revision.md`

Compact revision material for quickly reconstructing the topic before an interview.

### 🎤 `03_Interview.md`

Questions progress from:

```text
Beginner
→ Intermediate
→ Advanced
→ Staff Engineer Challenge
```

### 💡 `04_Assignment.md`

Applied reasoning rather than simple repetition:

- trace calculations
- diagnose broken designs
- reason about tensor shapes
- debug training / inference behavior
- compare architectural choices
- analyze production trade-offs

### ⚡ `05_CheatSheet.md`

High-yield last-minute revision.

### 🚫 `06_Common_Mistakes.md`

Misconceptions, traps, and distinctions that commonly cause incorrect interview answers.

> 📌 Smaller bridge topics may intentionally use fewer files when a full six-file structure would add repetition rather than value.

---

# 🧭 How to Use This Repository

For a new topic, the recommended order is:

```text
1. 📘 Lecture
      ↓
2. 💡 Assignment
      ↓
3. 📝 Revision
      ↓
4. 🎤 Interview Questions
      ↓
5. 🚫 Common Mistakes
      ↓
6. ⚡ Cheat Sheet
```

For interview revision:

```text
Cheat Sheet
↓
Revision
↓
Interview Questions
↓
Common Mistakes
```

For the full curriculum and historical learning trail, use:

- **[MASTER_INDEX.md](MASTER_INDEX.md)** — complete syllabus, topic history, and roadmap
- **[Progress_Tracker.md](Progress_Tracker.md)** — learning progress tracking

---

# 👨‍💻 About This Journey

This is a personal learning project built from the perspective of an experienced software engineer moving deeper into **Generative AI architecture and engineering**.

The repository itself remains focused on GenAI. Broader Python/backend interview preparation will be maintained separately.

---

# 🤝 Contributions

Suggestions, corrections, and technical discussions are welcome.

If you find an explanation unclear or identify a technical mistake, feel free to open an issue or contribute an improvement.

---

# ⭐ If You Find This Useful

If this repository helps you understand AI more deeply or prepare for interviews, consider giving it a ⭐.

---

# 🎯 Final Goal

> **Become capable of deriving, designing, building, debugging, and explaining production-grade Generative AI systems with confidence.**

Not just passing interviews.

Not just memorizing architectures.

But understanding the technology deeply enough to reason from first principles.