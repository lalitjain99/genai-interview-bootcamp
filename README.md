# 🚀 GenAI Interview Bootcamp

> **A first-principles roadmap from Software Engineer → Senior / Staff / Lead GenAI Engineer.**
>
> **Don't memorize AI. Understand it deeply enough that you could have invented it.**

This repository is my structured learning journey through **Deep Learning, Sequence Models, Attention, Transformers, Large Language Models, RAG, Agentic AI, MCP, Evaluation, Production GenAI, and System Design**.

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
```

> 📚 For the complete topic-by-topic learning history, current status, and future syllabus, see **[MASTER_INDEX.md](MASTER_INDEX.md)**.

---

# 🗺️ Bootcamp Roadmap

The original phase-based idea is preserved, but the phases now follow the actual conceptual progression of the course and the new repository structure.

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

Start a new architectural stage: remove recurrence as the primary sequence-processing mechanism and rebuild around attention.

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
- GPT
- BERT
- T5
- LLaMA

---

## Phase 5 — Large Language Models ⬜

Move from Transformer architecture to how modern language models represent text, train, align, adapt, and generate efficiently.

Planned areas include:

- Tokenization
- Embeddings and language representations
- Pretraining
- Fine-Tuning
- Instruction Tuning
- RLHF
- PPO
- DPO
- LoRA / QLoRA / PEFT
- Quantization
- KV Cache
- Speculative Decoding
- Mixture of Experts

> NLP foundations such as tokenization and embeddings remain part of the journey, but they are now placed where they naturally support modern LLM understanding instead of living as a disconnected standalone phase.

---

## Phase 6 — Retrieval-Augmented Generation (RAG) ⬜

Build systems that ground model generation in external knowledge.

- Embeddings Deep Dive
- Chunking Strategies
- Vector Databases
- Similarity Search
- Approximate Nearest Neighbor Search
- BM25 & Hybrid Search
- Bi-Encoder / Cross-Encoder concepts
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

## Phase 7 — Agentic AI & MCP ⬜

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
- Tool & Resource Integration
- LangGraph
- CrewAI
- AutoGen

> MCP is now grouped with Agentic AI because it is most useful when studied as part of the tool-and-context architecture of modern agent systems.

---

## Phase 8 — Prompt Engineering & Evaluation ⬜

Learn how to control model behavior and measure whether the system actually works.

- Prompt Design
- Few-Shot Prompting
- Chain-of-Thought concepts
- ReAct
- Structured Outputs
- Prompt Evaluation
- LLM-as-a-Judge
- BLEU / ROUGE / BERTScore
- RAGAS
- DeepEval
- Hallucination Evaluation
- Human Evaluation

---

## Phase 9 — Production GenAI & Cloud ⬜

Learn the engineering required to run GenAI systems reliably, securely, and economically.

- FastAPI for LLM APIs
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

> Cloud is now combined with Production GenAI because deployment, observability, security, scaling, and cloud services are part of the same production-engineering problem.

---

## Phase 10 — GenAI System Design ⬜

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

# 💻 Parallel Engineering Track

Coding practice should not wait until the final phase. It runs **in parallel with the GenAI curriculum**.

```text
Python
├── Core Python
├── Async Programming
├── FastAPI
├── Design Patterns
└── Testing

Problem Solving
├── DSA
└── SQL

Engineering
├── Git
├── Docker
├── Kubernetes
└── API / Backend Design
```

This keeps interview coding and backend engineering skills growing alongside GenAI depth.

---

# 📂 Repository Structure

The repository is now organized around **major learning stages as top-level modules**.

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
└── future top-level modules are added as the bootcamp progresses
```

The structure intentionally mirrors the learning journey:

```text
01_DeepLearning
      ↓
02_Sequence_to_Sequence_Models
      ↓
03_Attention
      ↓
04_Transformers
      ↓
05_LLMs
      ↓
...
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

The repository is intentionally designed to connect:

```text
Mathematics
+
Machine Learning
+
Python / Backend Engineering
+
Cloud / Production Engineering
+
System Design
```

because Senior and Staff GenAI roles require all of them together.

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