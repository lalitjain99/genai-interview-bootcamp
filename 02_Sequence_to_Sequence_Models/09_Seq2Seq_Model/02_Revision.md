# 📝 Revision — Sequence-to-Sequence (Seq2Seq) Model

> **Goal:** Quickly revise how a classic Seq2Seq model maps one sequence to another, how encoder and decoder interact, how training differs from inference, and why the fixed context vector becomes the major limitation.

---

# 🌟 1. What Is Seq2Seq?

Seq2Seq means:

# **Sequence → Sequence**

It maps:

```text
Input Sequence
↓
Output Sequence
```

Examples:

```text
English
→ French
```

```text
Article
→ Summary
```

```text
Audio
→ Text
```

---

# 🧠 2. Seq2Seq vs Encoder–Decoder

### Encoder–Decoder

Architecture pattern:

```text
Encoder
→ Representation
→ Decoder
```

### Seq2Seq

Concrete use of that pattern for:

```text
Sequence
→ Sequence
```

So:

> Encoder–Decoder is the architecture idea.
> Seq2Seq is the complete sequence-transformation model.

---

# ⭐ 3. Overall Pipeline

```text
Source Tokens
↓
Source Embeddings
↓
Encoder
↓
Context Vector
↓
Decoder
↓
Target Token Probabilities
↓
Target Sequence
```

This is the classic recurrent Seq2Seq pipeline.

---

# 🧠 4. Source Sequence

Let:

`X = (x_1, x_2, ..., x_T)`

Example:

```text
"I love AI"
```

Tokenized:

```text
"I"
"love"
"AI"
```

These tokens are first converted to IDs and then embeddings.

---

# 📐 5. Source Embeddings

If source vocabulary size is:

`V_src`

and embedding dimension is:

`D_src`

then:

`E_src ∈ R^(V_src × D_src)`

Each token ID retrieves one source embedding vector.

So:

```text
Token ID
↓
Embedding Lookup
↓
x_t
```

The encoder consumes:

> embedding vectors, not raw words.

---

# ⭐ 6. Encoder Recurrence

For a recurrent encoder:

`h_t = f_enc(x_t, h_(t-1))`

Example:

```text
x1 + h0
↓
h1

x2 + h1
↓
h2

x3 + h2
↓
h3
```

In the classic fixed-context model:

# `c = h_T`

---

# 🧠 7. What Does the Context Vector Mean?

The context vector:

`c`

is:

> a learned compressed representation of the source sequence.

It does **not** mean:

> exact storage of every source token.

Think:

```text
Whole Source
↓
Encoder
↓
Compressed Representation
```

---

# ⭐ 8. Decoder Initialization

A simple setup:

# `d_0 = c`

or:

`d_0 = W_c c + b_c`

if encoder and decoder dimensions differ.

So:

```text
Encoder Final State
↓
Decoder Starting State
```

---

# 🚀 9. Start of Decoding

The decoder starts with:

# `<SOS>` / `<BOS>`

Example:

```text
<SOS>
↓
Decoder
↓
First Target Token
```

This provides the first decoder input.

---

# 🧠 10. Decoder Uses Target Embeddings

The previous target token is converted into an embedding.

If target vocabulary size is:

`V_tgt`

and target embedding size:

`D_tgt`

then:

`E_tgt ∈ R^(V_tgt × D_tgt)`

So:

```text
Previous Target Token
↓
Target Embedding
↓
Decoder
```

---

# 📐 11. Decoder Recurrence

A simplified decoder equation:

`d_t = f_dec(e(y_(t-1)), d_(t-1), c)`

where:

* `e(y_(t-1))` = embedding of previous target token
* `d_(t-1)` = previous decoder state
* `c` = source context

Some implementations use `c` mainly to initialize the decoder rather than explicitly feeding it at every step.

---

# ⭐ 12. Decoder State to Vocabulary Distribution

Decoder hidden state:

`d_t`

is converted to logits:

`z_t = W_o d_t + b_o`

Then:

`softmax(z_t)`

produces:

`P(y_t | y_<t, X)`

Flow:

```text
Decoder State
↓
Linear Projection
↓
Logits
↓
Softmax
↓
Vocabulary Probabilities
```

---

# 🧮 13. Output Projection Dimensions

If:

```text
Decoder hidden size = H
Target vocabulary size = V_tgt
```

then:

```text
d_t ∈ R^H

W_o ∈ R^(V_tgt × H)

b_o ∈ R^(V_tgt)

z_t ∈ R^(V_tgt)
```

Softmax gives:

> one probability per target token.

---

# 🔄 14. Autoregressive Decoding

Seq2Seq decoder generates:

```text
<SOS>
↓
y1
↓
y2
↓
y3
↓
...
↓
<EOS>
```

Each output depends on:

> previous target tokens + source information.

This is autoregressive generation.

---

# 📐 15. Core Probability Equation

Seq2Seq models:

# `P(Y | X)`

where:

`X` = source sequence

`Y` = target sequence

Autoregressive factorization:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

This is one of the most important equations to remember.

---

# 🧮 16. Sequence Probability Example

Suppose:

```text
P(A | X) = 0.8
P(B | A,X) = 0.7
P(<EOS> | A,B,X) = 0.9
```

Then:

`P(Y|X) = 0.8 × 0.7 × 0.9`

# `= 0.504`

The full sequence probability is the product of conditional next-token probabilities.

---

# ⭐ 17. Target Shift During Training

Suppose target is:

```text
J'aime l'IA <EOS>
```

Decoder input:

```text
<SOS> J'aime l'IA
```

Expected output:

```text
J'aime l'IA <EOS>
```

So:

```text
Decoder Input:
<SOS>     J'aime     l'IA

Target:
J'aime    l'IA       <EOS>
```

This is:

> target shifting by one position.

---

# 🧠 18. Why Shift the Target?

Because the decoder learns:

```text
Given <SOS>
→ predict y1

Given y1
→ predict y2

Given y2
→ predict <EOS>
```

This directly matches:

`P(y_t | y_<t, X)`

---

# 🎓 19. Teacher Forcing Intuition

During training, the true previous target token may be fed into the decoder.

Example:

```text
True "J'aime"
↓
Predict "l'IA"
```

This is:

> teacher forcing.

We will study it separately in detail.

---

# ⚠️ 20. Training vs Inference

## Training

Often:

```text
Correct Previous Target
↓
Predict Next Target
```

## Inference

```text
Model's Previous Prediction
↓
Predict Next Target
```

So:

# **Training flow ≠ inference flow**

---

# 🚨 21. Error Propagation at Inference

Suppose the model should generate:

```text
I am happy
```

but predicts:

```text
I is ...
```

Then:

```text
wrong token
↓
wrong next input/history
↓
changed decoder state
↓
possibly more wrong tokens
```

Early errors can propagate.

---

# 🛑 22. End of Sequence

The decoder stops when it predicts:

# `<EOS>`

This means:

> target generation is complete.

So output length need not be known in advance.

---

# ↔️ 23. Input and Output Lengths Can Differ

Example:

```text
Source length = 8
Target length = 5
```

This is perfectly valid.

So:

# `T_src ≠ T_tgt`

Seq2Seq naturally supports different lengths.

---

# 🧠 24. Source and Target Vocabularies Can Differ

Translation may use:

```text
English Vocabulary
→ Encoder Embeddings
```

and:

```text
French Vocabulary
→ Decoder Embeddings
```

So:

# `V_src ≠ V_tgt`

is completely normal.

---

# 🔗 25. Seq2Seq with LSTM

Classic architecture:

```text
LSTM Encoder
↓
(h_T, c_T)
↓
LSTM Decoder
```

Possible initialization:

`h_0^dec = h_T^enc`

`c_0^dec = c_T^enc`

or projected versions.

---

# 🔗 26. Seq2Seq with GRU

GRU encoder:

```text
Source
↓
GRU Encoder
↓
h_T
```

GRU decoder:

```text
h_0^dec = h_T^enc
```

or projected version.

No separate cell state is transferred.

---

# ↔️ 27. Bidirectional Encoder + Causal Decoder

A common design is:

```text
BiLSTM Encoder
↓
Context
↓
Unidirectional Decoder
```

Why?

### Encoder

Source sequence is already fully available.

### Decoder

Future target tokens do not exist yet.

Therefore:

```text
Encoder
→ can often be bidirectional

Decoder
→ usually causal
```

---

# 📐 28. Bidirectional Context Dimension

If:

```text
→h_T ∈ R^H

←h_1 ∈ R^H
```

and we concatenate:

`c = [→h_T ; ←h_1]`

then:

# `c ∈ R^(2H)`

The decoder may require a projection if its hidden size differs.

---

# 🧮 29. Training Objective

At every decoder step:

`L_t = -log P(y_t^correct)`

Total sequence loss:

# `L = Σ_t L_t`

Usually:

* summed
* averaged
* or normalized

across valid target positions.

---

# 🧠 30. Why Cross-Entropy?

If correct token gets high probability:

```text
P(correct) = 0.9
```

loss is small.

If:

```text
P(correct) = 0.01
```

loss is large.

So training encourages:

> high probability for the correct next token.

---

# 🔄 31. End-to-End Training

Loss is computed at decoder outputs, but gradients flow through:

```text
Output Loss
↓
Decoder
↓
Context
↓
Encoder
```

So both:

* encoder
* decoder

learn together.

This is:

# **end-to-end learning**

---

# 🔗 32. BPTT on Both Sides

Because both encoder and decoder are recurrent:

```text
Encoder
→ BPTT through source sequence

Decoder
→ BPTT through target sequence
```

The full computational graph connects them through:

> context/state transfer.

---

# 📦 33. Padding in Batches

Different sequence lengths require padding.

Example:

```text
A: x x x x PAD PAD
B: x x x x x   x
C: x x x PAD PAD PAD
```

Padding lets sequences fit:

> common batch dimensions.

---

# 🚫 34. Mask Padding

Padding should usually not contribute to target loss.

Example:

```text
A B <EOS> PAD PAD
```

Loss:

```text
A      → include
B      → include
<EOS>  → include
PAD    → ignore
PAD    → ignore
```

Otherwise the model wastes effort learning artificial positions.

---

# ⭐ 35. Greedy Decoding

At each inference step:

> select the token with highest probability.

Example:

```text
A → 0.60
B → 0.30
C → 0.10
```

Greedy chooses:

`A`

Simple and fast.

---

# ⚠️ 36. Greedy Is Not Globally Optimal

The locally best token may lead to:

> a poor full sequence.

Example:

```text
Step 1:
A = 0.55
B = 0.45
```

`A` may look better locally, while sequences starting with `B` may have higher total probability.

This motivates:

# **Beam Search**

later.

---

# 🧠 37. Seq2Seq Is Conditional Generation

Normal language model:

`P(Y)`

Seq2Seq:

# `P(Y | X)`

Example:

```text
P(French sentence | English sentence)
```

The source sequence directly conditions target generation.

---

# ⚠️ 38. Context Can Be Used in Different Ways

Depending on implementation:

### Option A

Context initializes decoder:

```text
c
↓
d_0
```

### Option B

Context is also fed at decoder steps:

`d_t = f(y_(t-1), d_(t-1), c)`

So do not assume every Seq2Seq implementation uses context identically.

---

# 🚨 39. Classic Seq2Seq's Major Limitation

Classic Seq2Seq compresses:

```text
x1 x2 x3 ... xT
↓
ONE context c
↓
Decoder
```

The same-sized vector must represent:

```text
5-token source
50-token source
500-token source
```

This creates:

# **Fixed Context Vector Bottleneck**

---

# 🧠 40. Why This Becomes Difficult

For long sequences, the decoder may need information about:

> a specific source phrase.

But it cannot directly say:

```text
Give me h_17
```

It only receives:

`c`

So important source details may become difficult to recover.

---

# 🔗 41. Why This Leads to Attention

Instead of only:

`c = h_T`

what if decoder could access:

```text
h1, h2, h3, ..., hT
```

and decide:

> which source states matter at each target step?

That idea becomes:

# **Attention**

---

# 🚫 42. High-Yield Interview Traps

### ❌ Seq2Seq requires equal source and target length

No.

### ❌ Token IDs are semantic embeddings

No.

### ❌ Source and target vocabularies must be the same

No.

### ❌ Decoder state is already a probability distribution

No.

### ❌ Training and inference are identical

No.

### ❌ Padding should contribute to loss

No.

### ❌ Greedy decoding always finds the best sequence

No.

### ❌ A bidirectional decoder is always fine

No, not for causal autoregressive generation.

### ❌ Seq2Seq removes the fixed-context bottleneck

No.

Classic Seq2Seq exposes that limitation.

---

# 📐 43. Formula Flash Card

### Source

`X = (x_1, ..., x_T)`

### Target

`Y = (y_1, ..., y_T')`

### Encoder

`h_t = f_enc(x_t, h_(t-1))`

### Context

# `c = h_T`

### Decoder

`d_t = f_dec(e(y_(t-1)), d_(t-1), c)`

### Output

`z_t = W_o d_t + b_o`

### Token Probability

`P(y_t | y_<t, X) = softmax(z_t)`

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Token Loss

`L_t = -log P(y_t^correct)`

### Sequence Loss

# `L = Σ_t L_t`

excluding padded target positions.

---

# 🎤 44. 30-Second Interview Answer

> **A Seq2Seq model maps one variable-length sequence to another using an encoder and decoder. Source tokens are embedded and processed by the encoder into a context representation. The decoder is initialized or conditioned by that representation and generates target tokens autoregressively from `<SOS>` until `<EOS>`. During training, the target sequence is shifted so the model learns next-token prediction, often using teacher forcing, while during inference it feeds back its own predictions. The model is trained end-to-end using token-level cross-entropy, but classic Seq2Seq suffers from a fixed-context-vector bottleneck for long source sequences.**

---

# 🧠 12 Things You Must Know Cold

1. **Seq2Seq = sequence → sequence**
2. **Usually uses Encoder–Decoder**
3. **Source tokens need embeddings**
4. **Target tokens need embeddings**
5. **Source and target vocabularies can differ**
6. **Original context is often `h_T`**
7. **`<SOS>` starts generation**
8. **`<EOS>` stops generation**
9. **Targets are shifted during training**
10. **Decoder is autoregressive**
11. **Training and inference differ**
12. **Fixed context bottleneck motivates attention**

---

# 🧠 Final Mental Model

```text
SOURCE TOKENS
      │
      ▼
SOURCE EMBEDDINGS
      │
      ▼
    ENCODER
      │
      ▼
    CONTEXT
      │
      ▼
    DECODER
      │
      ▼
TOKEN DISTRIBUTION
      │
      ▼
TARGET TOKEN
      │
      └──────→ Feed into next step
```

Probability view:

# `X → P(Y | X)`

Training view:

```text
<SOS> y1 y2
↓
predict
y1    y2 <EOS>
```

Inference view:

```text
<SOS>
↓
prediction
↓
feed prediction back
↓
next prediction
```

And the architectural limitation:

```text
Whole Source
↓
One Context Vector
↓
Too Much Compression
↓
Need Dynamic Access to Encoder States
↓
Attention
```

---

# ⭐ Golden Rule

> **Seq2Seq is conditional autoregressive generation: the encoder represents the source, the decoder predicts the target one token at a time, and the classic model's biggest weakness is forcing the entire source through one fixed context vector.**
