# 🚀 GenAI Interview Bootcamp
### From Software Engineer → Senior / Staff / Lead GenAI Engineer

> **Don't memorize AI. Understand it deeply enough that you could have invented it.**

---

# 📖 About this Repository

This repository is a **first-principles GenAI interview roadmap**. The objective is not to memorize disconnected definitions or interview questions. Every topic is introduced as a problem, the limitation of the existing solution is identified, and the next idea is derived naturally.

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

## 📚 Standard Topic Structure

| File | Purpose |
|---|---|
| 📘 `01_Lecture.md` | Story-driven learning from first principles |
| 📝 `02_Revision.md` | Fast revision notes |
| 🎤 `03_Interview.md` | Beginner → Staff-level interview questions |
| 💡 `04_Assignment.md` | Applied reasoning, calculations and debugging |
| ⚡ `05_CheatSheet.md` | High-yield last-minute revision |
| 🚫 `06_Common_Mistakes.md` | Misconceptions and interview traps |
| 💻 `Code.ipynb` | Optional hands-on implementation |

> 📌 Smaller bridge topics may intentionally use fewer artifacts.

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

## 🧭 Where We Are in the Story

```text
Perceptron
↓
Hidden-Layer Revolution / XOR
↓
Activation Functions
↓
Training Neural Networks
↓
Optimizers
↓
Generalization & Regularization
↓
Weight Initialization
↓
Residual Connections
↓
Universal Approximation Theorem
↓
CNNs
↓
Sequence Models
↓
RNN → BPTT → Vanishing / Exploding Gradients
↓
LSTM / GRU → Bidirectional RNN
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
Attention Limitations
↓
TRANSFORMERS
```

## 📈 Progress Legend

| Symbol | Meaning |
|---|---|
| ✅ | Complete |
| 🟨 | In Progress |
| ⬜ | Not Started |
| ➖ | Intentionally skipped / covered elsewhere |

---

# 🗺️ Course Roadmap

> The complete learning history is preserved below. Modules are collapsible so the page remains easy to navigate.

---

<details>
<summary><strong>✅ Module 1 — Deep Learning Foundations</strong></summary>

> Learn how neural networks work from first principles before moving into sequence architectures and LLMs.

**Status:** ✅ Complete  
**Difficulty:** ⭐⭐⭐⭐☆

## 📌 Module Progress

| # | Topic | Status |
|---|---|---|
| 01 | Perceptron | ✅ |
| 02 | Hidden-Layer Revolution & XOR | ✅ |
| 03 | Activation Functions | ✅ |
| 04 | Training Neural Networks | ✅ |
| 05 | Optimization / Optimizers | ✅ |
| 06 | Generalization & Regularization | ✅ |
| 07 | Weight Initialization | ✅ |
| 08 | Residual Connections | ✅ |
| 09 | Universal Approximation Theorem | ✅ |
| 10 | Convolutional Neural Networks | ✅ |

### 🧠 Core Learning Progression

```text
Biological Neuron
↓
Perceptron
↓
Linear Decision Boundary
↓
XOR Failure
↓
Hidden Layers
↓
Need Non-Linearity
↓
Activation Functions
↓
Need to Learn Parameters
↓
Loss + Gradient Descent + Backpropagation
↓
Need Better Optimization
↓
Modern Optimizers
↓
Need Better Generalization
↓
Regularization
↓
Need Stable Signal / Gradient Flow
↓
Weight Initialization
↓
Need Much Deeper Networks
↓
Residual Connections
↓
Representational Power
↓
Universal Approximation
↓
Need Spatial Inductive Bias
↓
CNNs
```

<details>
<summary><strong>🔹 01 — Perceptron ✅</strong></summary>

**Repository:** `DeepLearning/01-Perceptron`

Key concepts covered:

- Biological-neuron intuition
- Inputs and features
- Weights and what their magnitude/sign mean
- Weighted sum
- Bias and why it is required
- Activation / thresholding
- Decision boundary
- Linear separability
- Learning intuition
- Perceptron limitations
- Perceptron vs linear regression

**Core mental model**

```text
Inputs
↓
Weighted Evidence
+
Bias
↓
Decision Boundary
↓
Prediction
```

</details>

<details>
<summary><strong>🔹 02 — Hidden-Layer Revolution & XOR ✅</strong></summary>

**Repository:** `DeepLearning/02-Hidden Layer Revolution`

Key concepts covered:

- Why a single Perceptron fails on XOR
- Linear separability
- Why stacking only linear transformations is still linear
- Hidden layers
- Intermediate representations
- Combining multiple decision boundaries
- Why nonlinear activation is essential
- How hidden layers make nonlinear problems representable

**Key transition**

```text
Single Linear Boundary
↓
XOR Failure
↓
Multiple Hidden Units
↓
Multiple Boundaries
↓
Nonlinear Decision Regions
```

</details>

<details>
<summary><strong>🔹 03 — Activation Functions ✅</strong></summary>

**Repository:** `DeepLearning/03-Activation Functions`

| Topic | Status |
|---|---|
| Step Function | ✅ |
| Sigmoid | ✅ |
| Tanh | ✅ |
| ReLU | ✅ |
| Leaky ReLU | ✅ |
| GELU | ✅ |
| Softmax | ✅ |

Key concepts covered:

- Why activation functions are required
- Nonlinearity
- Saturation
- Vanishing-gradient intuition
- Dead ReLU neurons
- Sparse activations
- Why GELU is common in Transformers
- Logits vs probabilities
- Softmax distributions

**Key transition**

```text
Linear Layer + Linear Layer
= Still Linear
↓
Need Non-Linearity
↓
Activation Functions
```

</details>

<details>
<summary><strong>🔹 04 — Training Neural Networks ✅</strong></summary>

**Repository:** `DeepLearning/04-Training Neural Network`

| Topic | Status |
|---|---|
| How Does a Neural Network Learn? | ✅ |
| Loss Functions | ✅ |
| Mean Squared Error (MSE) | ✅ |
| Mean Absolute Error (MAE) | ✅ |
| Cross Entropy | ✅ |
| Why Cross Entropy Uses Log | ✅ |
| Binary Cross Entropy | ✅ |
| Negative Log Likelihood | ✅ |
| BCEWithLogitsLoss | ✅ |
| Gradient Descent | ✅ |
| Derivatives | ✅ |
| Partial Derivatives | ✅ |
| Gradient Vector | ✅ |
| Chain Rule | ✅ |
| Computational Graph | ✅ |
| Backpropagation | ✅ |
| Training Pipeline | ✅ |

**Core training loop**

```text
Forward Pass
↓
Prediction
↓
Loss
↓
Backpropagation
↓
Gradients
↓
Optimizer Update
↓
Repeat
```

</details>

<details>
<summary><strong>🔹 05 — Optimization / Optimizers ✅</strong></summary>

**Repository:** `DeepLearning/05-Optimization`

| Topic | Status |
|---|---|
| Learning Rate | ✅ |
| Batch Gradient Descent | ✅ |
| Stochastic Gradient Descent | ✅ |
| Mini-Batch Gradient Descent | ✅ |
| Momentum | ✅ |
| AdaGrad | ✅ |
| RMSProp | ✅ |
| Adam | ✅ |
| AdamW | ✅ |

**Evolution studied from limitations**

```text
Gradient Descent
↓
SGD
↓
Momentum
↓
AdaGrad
↓
RMSProp
↓
Adam
↓
AdamW
```

Key concepts:

- Learning-rate trade-offs
- Mini-batch noise
- Momentum
- Moving averages
- Adaptive learning rates
- First and second moments
- Bias correction
- L2 regularization vs weight decay
- Decoupled weight decay

</details>

<details>
<summary><strong>🔹 06 — Generalization & Regularization ✅</strong></summary>

**Repository:** `DeepLearning/06-Generalization & Regularization`

| Topic | Status |
|---|---|
| Why Models Overfit | ✅ |
| Bias–Variance Trade-off | ✅ |
| L1 Regularization | ✅ |
| L2 Regularization | ✅ |
| Elastic Net | ✅ |
| Weight Decay | ➖ Covered deeply with AdamW |
| Early Stopping | ✅ |
| Data Augmentation | ✅ |
| Dropout | ✅ |
| Batch Normalization | ✅ |
| Regularization Comparison | ✅ |

**Important distinction**

```text
Optimization
→ fit the training objective

Regularization
→ improve generalization
```

</details>

<details>
<summary><strong>🔹 07 — Weight Initialization ✅</strong></summary>

**Repository:** `DeepLearning/07-Weight_Initialization`

| Topic | Status |
|---|---|
| Why Weight Initialization Matters | ✅ |
| Zero Initialization | ✅ |
| Random Initialization | ✅ |
| Symmetry Problem | ✅ |
| Vanishing / Exploding Activations & Gradients | ✅ |
| Xavier / Glorot Initialization | ✅ |
| He / Kaiming Initialization | ✅ |
| Initialization Comparison | ✅ |

**Evolution**

```text
Zero Initialization
↓
Symmetry Problem
↓
Random Initialization
↓
Symmetry Broken
↓
But Scale Matters
↓
Controlled Variance
↓
Xavier / He
```

Important formulas:

- Xavier: `Var(W) = 2 / (fan_in + fan_out)`
- He: `Var(W) ≈ 2 / fan_in`

</details>

<details>
<summary><strong>🔹 08 — Residual Connections ✅</strong></summary>

**Repository:** `DeepLearning/08_Residual_Connection`

Core equation:

`y = F(x) + x`

Key concepts covered:

- Degradation problem in very deep networks
- Degradation vs overfitting
- Residual learning
- Identity mapping
- Shortcut / skip connection
- Gradient shortcut
- Identity shortcut
- Projection shortcut
- Dimension mismatch
- ResNet intuition

When dimensions differ:

`y = F(x) + W_s x`

**Core mental model**

```text
Existing Representation
+
Learned Correction
↓
Improved Representation
```

</details>

<details>
<summary><strong>🔹 09 — Universal Approximation Theorem ✅</strong></summary>

**Repository:** `DeepLearning/09_Universal_Approximation_Theorm`

Key concepts covered:

- Representational capacity
- Suitable nonlinear activation
- Sufficient hidden units
- Continuous functions
- Compact / bounded domains
- Arbitrarily close approximation
- Width vs depth
- Shallow vs deep networks
- ReLU piecewise-linear approximation

**Most important distinction**

```text
Can Represent
≠
Can Train Easily
≠
Can Generalize Well
```

</details>

<details>
<summary><strong>🔹 10 — Convolutional Neural Networks ✅</strong></summary>

**Repository:** `DeepLearning/10_Convolutional_Neural_Network`

| Topic | Status |
|---|---|
| Why CNNs Were Invented | ✅ |
| Image Representation | ✅ |
| Convolution Operation | ✅ |
| Filters / Kernels | ✅ |
| Feature Maps | ✅ |
| Stride | ✅ |
| Padding | ✅ |
| Output Dimension Calculation | ✅ |
| Multiple Filters / Channels | ✅ |
| Pooling | ✅ |
| CNN Feature Hierarchy | ✅ |
| CNN Architecture | ✅ |
| CNN Backpropagation | ✅ |
| Parameter Sharing | ✅ |
| Receptive Field | ✅ |
| CNN vs Fully Connected Networks | ✅ |
| CNN Architecture Comparison | ✅ |
| Module Summary | ✅ |

Key concepts:

- Local connectivity
- Parameter sharing
- Translation equivariance
- Filters / kernels
- Feature maps
- Channels and output channels
- Stride and padding
- Pooling
- Receptive field
- Hierarchical feature learning
- ResNet
- DenseNet
- Inception
- MobileNet
- EfficientNet

**Core progression**

```text
Images
↓
Fully Connected Networks Ignore Spatial Bias
↓
Local Connectivity
↓
Convolution
↓
Shared Filters
↓
Feature Maps
↓
Hierarchical Features
↓
Modern CNN Architectures
```

</details>

</details>

---

<details>
<summary><strong>✅ Module 2 — Sequence-to-Sequence Models</strong></summary>

> Understand sequential data, recurrent architectures, encoder-decoder learning, training/inference mismatch, and target-side decoding.

**Status:** ✅ Complete  
**Difficulty:** ⭐⭐⭐⭐☆

## 📌 Module Progress

| # | Topic | Status |
|---|---|---|
| 01 | Why Sequence Models | ✅ |
| 02 | Recurrent Neural Network (RNN) | ✅ |
| 03 | Backpropagation Through Time (BPTT) | ✅ |
| 04 | Vanishing & Exploding Gradients | ✅ |
| 05 | Long Short-Term Memory (LSTM) | ✅ |
| 06 | Gated Recurrent Unit (GRU) | ✅ |
| 07 | Bidirectional RNN | ✅ |
| 08 | Encoder–Decoder Architecture | ✅ |
| 09 | Seq2Seq Model | ✅ |
| 10 | Context Vector Bottleneck | ✅ |
| 11 | Teacher Forcing | ✅ |
| 12 | Inference & Autoregressive Decoding | ✅ |
| 13 | Beam Search | ✅ |
| ⭐ | Module Summary | ✅ |

### 🧠 Complete Learning Progression

```text
Sequential Data
↓
Need Memory / Context
↓
RNN
↓
Need to Train Through Time
↓
BPTT
↓
Long Gradient Chains
↓
Vanishing / Exploding Gradients
↓
LSTM / GRU
↓
Need Both Past and Future Source Context
↓
Bidirectional RNN
↓
Need Variable-Length Source → Variable-Length Target
↓
Encoder–Decoder
↓
Seq2Seq
↓
Fixed Context Bottleneck
↓
Need Stable Decoder Training
↓
Teacher Forcing
↓
Teacher Disappears at Deployment
↓
Autoregressive Decoding
↓
Greedy Search Is Myopic
↓
Beam Search
↓
Target Search Improved — Source Access Still Fixed
↓
ATTENTION
```

<details>
<summary><strong>🔹 01 — Why Sequence Models ✅</strong></summary>

- Why order matters
- Why fixed feed-forward processing is insufficient for sequential context
- Variable-length sequences
- Temporal / contextual dependency
- Need for persistent state

</details>

<details>
<summary><strong>🔹 02 — Recurrent Neural Network (RNN) ✅</strong></summary>

Core recurrence:

`a_t = W_xh x_t + W_hh h_(t-1) + b_h`

`h_t = tanh(a_t)`

Key concepts:

- Hidden state
- Recurrence
- Parameter sharing across timesteps
- Dynamic state vs learned parameters
- Sequence memory as compressed context
- Variable sequence lengths
- Stateful / streaming intuition
- Sequential computation across time

</details>

<details>
<summary><strong>🔹 03 — Backpropagation Through Time (BPTT) ✅</strong></summary>

- Unrolling the recurrent computation graph
- Shared-parameter gradient accumulation
- Temporal credit assignment
- Repeated Jacobian products
- Full BPTT vs truncated BPTT
- Numerical state can continue while gradient history is detached
- BPTT ≠ backward RNN

</details>

<details>
<summary><strong>🔹 04 — Vanishing & Exploding Gradients ✅</strong></summary>

- Repeated gradient multiplication through time
- Contractive products → vanishing gradients
- Expansive products → exploding gradients
- Directional / matrix intuition
- Tanh saturation
- Gradient clipping for exploding gradients
- Why long-range dependencies remain difficult

</details>

<details>
<summary><strong>🔹 05 — LSTM ✅</strong></summary>

Core cell update:

`c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t`

Key concepts:

- Cell state vs hidden state
- Forget gate
- Input gate
- Candidate memory
- Output gate
- Sigmoid for control
- Tanh for candidate/content
- Additive memory path
- Better long-range gradient flow

**Mental model:** `Cell State = Notebook`, `c_t = KEEP + WRITE`

</details>

<details>
<summary><strong>🔹 06 — GRU ✅</strong></summary>

Core convention used in this bootcamp:

`h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

Key concepts:

- Update gate
- Reset gate
- Candidate hidden state
- No separate cell state
- Simpler gating than LSTM
- Fewer parameters than comparable LSTM
- Reset gate affects candidate construction
- Update gate controls final old/new blend

</details>

<details>
<summary><strong>🔹 07 — Bidirectional RNN ✅</strong></summary>

- Separate forward and backward recurrent networks
- Past + future context for source representation
- Concatenated forward/backward states
- BiRNN / BiLSTM / BiGRU
- Why full bidirectionality is unsuitable for strict causal streaming
- Bidirectional RNN ≠ BPTT

</details>

<details>
<summary><strong>🔹 08 — Encoder–Decoder Architecture ✅</strong></summary>

```text
Source
↓
Encoder
↓
Context / Representation
↓
Decoder
↓
Target
```

Key concepts:

- Separate source understanding from target generation
- Variable-length source and target sequences
- Encoder state transfer
- Decoder initialization / conditioning
- Autoregressive target factorization
- LSTM may transfer `(h_T, c_T)`; GRU transfers hidden state

</details>

<details>
<summary><strong>🔹 09 — Seq2Seq Model ✅</strong></summary>

Full pipeline:

```text
Source IDs
↓
Source Embeddings
↓
Encoder
↓
Context
↓
Decoder
↓
Output Projection
↓
Vocabulary Logits
↓
Softmax
↓
Target Tokens
```

Core factorization:

`P(Y|X) = Π_t P(y_t | y_<t, X)`

Key concepts:

- Source/target embeddings
- Target shifting
- `<SOS>/<BOS>`
- `<EOS>`
- Padding masks
- Teacher-forced negative log-likelihood
- Training vs generation pipeline

</details>

<details>
<summary><strong>🔹 10 — Context Vector Bottleneck ✅</strong></summary>

Core problem:

> All variable-length source information must pass through a fixed source-to-decoder communication interface.

Key concepts:

- Fixed context `c = h_T`
- Compression pressure as source information grows
- Capacity vs accessibility
- Missing entities, dates, numbers and early clauses
- Larger hidden state may increase capacity but does not change access pattern
- LSTM / GRU / BiRNN do not fundamentally remove the fixed interface
- Motivation to preserve `h_1 ... h_T`

</details>

<details>
<summary><strong>🔹 11 — Teacher Forcing ✅</strong></summary>

- Feed ground-truth previous target token during decoder training
- Teacher Forcing gives previous answer, not current answer
- Stable next-token learning
- Teacher Forcing ratio
- Scheduled sampling
- Training / inference mismatch
- Exposure bias
- Exposure bias vs error propagation

</details>

<details>
<summary><strong>🔹 12 — Inference & Autoregressive Decoding ✅</strong></summary>

```text
<BOS>
↓
Predict
↓
Select Token
↓
Feed Back
↓
Repeat
↓
<EOS>
```

Key concepts:

- No ground-truth target at deployment
- Autoregressive feedback
- Logits vs softmax probabilities
- Softmax does not choose the token
- Greedy vs sampling
- `<EOS>` vs `max_length`
- Error propagation
- Sequential generation latency

</details>

<details>
<summary><strong>🔹 13 — Beam Search ✅</strong></summary>

Core loop:

`Expand → Score → Rank → Prune`

Key concepts:

- Keep top `k` sequence hypotheses
- `Beam(k=1) ≈ Greedy`
- Cumulative log-probability scoring
- Active vs completed hypotheses
- `<EOS>` handling
- Length bias / length penalty
- Search error vs model error
- Beam width vs latency/memory
- Beam Search is approximate
- Wider beam does not fix poor source representation

</details>

</details>

---

<details open>
<summary><strong>🟨 Module 3 — Attention</strong></summary>

> Seq2Seq discovered the source-access bottleneck. This module develops the classical attention mechanism that solves it.

**Status:** 🟨 In Progress  
**Difficulty:** ⭐⭐⭐⭐☆

## 📌 Module Progress

| # | Topic | Status |
|---|---|---|
| 01 | Attention Motivation | ✅ |
| 02 | Attention Mechanism | 🟨 **CURRENT** |
| 03 | Bahdanau / Additive Attention | ⬜ **NEXT** |
| 04 | Luong / Multiplicative Attention | ⬜ |
| 05 | Attention Alignment & Visualization | ⬜ |
| 06 | Attention Limitations | ⬜ |
| ⭐ | Module Summary | ⬜ |

### 🔹 01 — Attention Motivation ✅

**Repository:** `DeepLearning/12_Attention/01_Attention_Motivation`

Completed artifacts:

- 📘 Lecture ✅
- 📝 Revision ✅
- 🎤 Interview ✅
- 💡 Assignment ✅
- ⚡ Cheat Sheet ✅
- 🚫 Common Mistakes ✅

Key concepts:

- Revisit the fixed context bottleneck
- Capacity vs accessibility
- Preserve all encoder states
- Dynamic context `c_t`
- Soft source-target alignment
- Searchlight / dynamic retrieval intuition
- Attention vs LSTM
- Attention vs Teacher Forcing
- Attention vs Beam Search
- Attention vs vanishing-gradient problem
- Direct source-to-decoder information paths
- Computational trade-off

Core equation introduced:

`c_t = Σ_i α_(t,i) h_i`

### 🔹 02 — Attention Mechanism 🟨 CURRENT

The current lecture derives where `α_(t,i)` comes from.

```text
Decoder State / Current Need
+
Encoder State h_i
↓
Compatibility Score e_(t,i)
↓
Softmax Across Source Positions
↓
Attention Weight α_(t,i)
↓
Weighted Sum
↓
Dynamic Context c_t
```

Core equations:

`e_(t,i) = score(s_(t-1), h_i)`

`α_(t,i) = exp(e_(t,i)) / Σ_j exp(e_(t,j))`

`c_t = Σ_i α_(t,i) h_i`

Core mental model:

> **Compare → Normalize → Aggregate**

### 🔜 Attention Module Progression

```text
Attention Motivation
↓
Generic Attention Mechanism
↓
How Should score(...) Be Calculated?
↓
Bahdanau / Additive Attention
↓
Luong / Multiplicative Attention
↓
Alignment & Visualization
↓
Attention Limitations
↓
TRANSFORMERS
```

> **Boundary:** Self-Attention, Query/Key/Value, Scaled Dot-Product Attention and Multi-Head Attention are intentionally reserved for the Transformer module.

</details>

---

# 🚀 Transformer & GenAI Learning Stages

> From here, **Transformers begin as a new top-level learning stage** rather than remaining inside `DeepLearning/`.

<details>
<summary><strong>⬜ Module 4 — Transformer Architecture</strong></summary>

> The next major architectural revolution: remove recurrence as the central sequence-processing mechanism and build around attention.

**Status:** ⬜ Not Started  
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
↓
Self-Attention
↓
Query / Key / Value
↓
Scaled Dot-Product Attention
↓
Multi-Head Attention
↓
Positional Information
↓
Feed-Forward + Residual + Normalization
↓
Transformer Encoder / Decoder
↓
Causal Attention
↓
GPT / BERT / T5 / LLaMA
```

</details>

<details>
<summary><strong>⬜ Module 5 — Large Language Models</strong></summary>

> Learn how modern LLMs are trained, adapted, aligned and served efficiently.

| # | Topic | Status |
|---|---|---|
| 01 | Tokenization | ⬜ |
| 02 | Embeddings | ⬜ |
| 03 | Pretraining | ⬜ |
| 04 | Fine-Tuning | ⬜ |
| 05 | Instruction Tuning | ⬜ |
| 06 | RLHF | ⬜ |
| 07 | PPO | ⬜ |
| 08 | DPO | ⬜ |
| 09 | LoRA | ⬜ |
| 10 | QLoRA | ⬜ |
| 11 | PEFT | ⬜ |
| 12 | Quantization | ⬜ |
| 13 | KV Cache | ⬜ |
| 14 | Speculative Decoding | ⬜ |
| 15 | Mixture of Experts (MoE) | ⬜ |

</details>

<details>
<summary><strong>⬜ Module 6 — Retrieval-Augmented Generation (RAG)</strong></summary>

> Build enterprise-grade systems that ground generation in external knowledge.

| # | Topic | Status |
|---|---|---|
| 01 | Embeddings Deep Dive | ⬜ |
| 02 | Chunking Strategies | ⬜ |
| 03 | Vector Databases | ⬜ |
| 04 | Similarity Search | ⬜ |
| 05 | Approximate Nearest Neighbor (ANN) | ⬜ |
| 06 | Hybrid Search | ⬜ |
| 07 | BM25 | ⬜ |
| 08 | Cross Encoder | ⬜ |
| 09 | Re-ranking | ⬜ |
| 10 | End-to-End RAG Pipeline | ⬜ |
| 11 | Advanced RAG | ⬜ |
| 12 | Graph RAG | ⬜ |
| 13 | Agentic RAG | ⬜ |
| 14 | RAG Evaluation | ⬜ |
| 15 | Hallucination Detection | ⬜ |

</details>

<details>
<summary><strong>⬜ Module 7 — Agentic AI & MCP</strong></summary>

> Move from single model calls to systems that plan, use tools, maintain state and collaborate.

| # | Topic | Status |
|---|---|---|
| 01 | Function Calling | ⬜ |
| 02 | Tool Calling | ⬜ |
| 03 | Model Context Protocol (MCP) | ⬜ |
| 04 | AI Agents | ⬜ |
| 05 | Planning | ⬜ |
| 06 | Reflection | ⬜ |
| 07 | Memory | ⬜ |
| 08 | Multi-Agent Systems | ⬜ |
| 09 | LangGraph | ⬜ |
| 10 | CrewAI | ⬜ |
| 11 | AutoGen | ⬜ |

</details>

<details>
<summary><strong>⬜ Module 8 — Prompt Engineering & Evaluation</strong></summary>

> Learn how to control model behavior and measure quality systematically.

| # | Topic | Status |
|---|---|---|
| 01 | Prompt Engineering | ⬜ |
| 02 | Chain of Thought Concepts | ⬜ |
| 03 | ReAct | ⬜ |
| 04 | Structured Output | ⬜ |
| 05 | Prompt Evaluation | ⬜ |
| 06 | LLM-as-a-Judge | ⬜ |
| 07 | BLEU | ⬜ |
| 08 | ROUGE | ⬜ |
| 09 | BERTScore | ⬜ |
| 10 | RAGAS | ⬜ |
| 11 | DeepEval | ⬜ |
| 12 | Human Evaluation | ⬜ |

</details>

<details>
<summary><strong>⬜ Module 9 — Production GenAI</strong></summary>

> Learn the engineering required to run GenAI systems reliably at scale.

| # | Topic | Status |
|---|---|---|
| 01 | FastAPI for LLM APIs | ⬜ |
| 02 | vLLM | ⬜ |
| 03 | TensorRT-LLM | ⬜ |
| 04 | Ray Serve | ⬜ |
| 05 | Kubernetes Deployment | ⬜ |
| 06 | GPU Optimization | ⬜ |
| 07 | Caching Strategies | ⬜ |
| 08 | Monitoring & Logging | ⬜ |
| 09 | Guardrails | ⬜ |
| 10 | Cost Optimization | ⬜ |
| 11 | Security | ⬜ |
| 12 | Observability | ⬜ |

</details>

<details>
<summary><strong>⬜ Module 10 — GenAI System Design</strong></summary>

> Senior / Staff / Principal-level architecture preparation.

| # | Topic | Status |
|---|---|---|
| 01 | ChatGPT System Design | ⬜ |
| 02 | Enterprise RAG Architecture | ⬜ |
| 03 | Multi-Tenant RAG | ⬜ |
| 04 | AI Coding Assistant | ⬜ |
| 05 | AI Search Engine | ⬜ |
| 06 | AI Customer Support Platform | ⬜ |
| 07 | AI Document Intelligence | ⬜ |
| 08 | AI Agent Platform | ⬜ |
| 09 | AI Memory Systems | ⬜ |
| 10 | LLMOps | ⬜ |

</details>

---

# 🧠 Full Learning Journey

```text
DEEP LEARNING FOUNDATIONS
Perceptron
→ Hidden Layers
→ Activations
→ Training
→ Optimization
→ Generalization
→ Initialization
→ Residual Connections
→ Universal Approximation
→ CNN
        ↓
SEQUENCE MODELS
RNN
→ BPTT
→ LSTM / GRU
→ BiRNN
→ Encoder–Decoder
→ Seq2Seq
→ Context Bottleneck
→ Teacher Forcing
→ Autoregressive Decoding
→ Beam Search
        ↓
ATTENTION
Motivation
→ Mechanism
→ Bahdanau
→ Luong
→ Alignment
→ Limitations
        ↓
TRANSFORMERS
        ↓
LARGE LANGUAGE MODELS
        ↓
RAG
        ↓
AGENTIC AI / MCP
        ↓
PROMPTING & EVALUATION
        ↓
PRODUCTION GENAI
        ↓
GENAI SYSTEM DESIGN
```

---

# 🎯 End Goal

By the end of this bootcamp, the goal is to be able to:

- explain every major Deep Learning concept from first principles;
- understand **why** each architecture was invented, not only how it works;
- derive the important mathematics behind sequence models, Attention and Transformers;
- reason about LLM training, inference, RAG, agents and evaluation;
- design and operate production-grade GenAI systems;
- handle Senior / Staff / Lead GenAI interview discussions with architectural and mathematical depth.

---

> ### ⭐ Bootcamp Principle
> **Problem → Limitation → Insight → Mathematics → Architecture → Production Reasoning**
