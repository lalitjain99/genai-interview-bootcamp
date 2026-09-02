# ⚡ Cheat Sheet — Sequence-to-Sequence (Seq2Seq) Model

> **Goal:** Fast revision of the complete Seq2Seq pipeline — source embeddings, encoder, context vector, decoder, target shifting, autoregressive inference, loss, masking, and the fixed-context bottleneck.

---

# 🌟 1. Seq2Seq in One Line

> **Seq2Seq = map one variable-length sequence to another variable-length sequence.**

```text
Source Sequence
↓
Encoder
↓
Context
↓
Decoder
↓
Target Sequence
```

Examples:

* Translation
* Summarization
* Speech-to-text
* Sequence transformation

---

# 🧠 2. Encoder–Decoder vs Seq2Seq

### Encoder–Decoder

Architecture pattern:

```text
Encoder
→ Representation
→ Decoder
```

### Seq2Seq

Concrete task/model:

```text
Sequence
→ Sequence
```

So:

> **Encoder–Decoder = architecture**
> **Seq2Seq = sequence transformation using that architecture**

---

# ⭐ 3. Complete Pipeline

```text
Source Token IDs
↓
Source Embeddings
↓
Encoder RNN/LSTM/GRU
↓
Context Vector
↓
Decoder Initial State
↓
Target Embeddings
↓
Decoder
↓
Logits
↓
Softmax
↓
Target Token
```

---

# 🔢 4. Token IDs Are Not Embeddings

Example:

```text
"I"    → 12
"love" → 47
"AI"   → 205
```

These are just IDs.

Use:

```text
Token ID
↓
Embedding Lookup
↓
Dense Vector
```

---

# 📐 5. Source Embedding Matrix

If:

* source vocabulary = `V_src`
* source embedding size = `D_src`

then:

# `E_src ∈ R^(V_src × D_src)`

Each source token retrieves one row.

---

# 📐 6. Target Embedding Matrix

If:

* target vocabulary = `V_tgt`
* target embedding size = `D_tgt`

then:

# `E_tgt ∈ R^(V_tgt × D_tgt)`

Important:

# `V_src` and `V_tgt` can differ

and:

# `D_src` and `D_tgt` can differ

---

# 🧠 7. Encoder Recurrence

For source:

`X = (x_1, ..., x_T)`

Encoder:

# `h_t = f_enc(x_t, h_(t-1))`

Classic context:

# `c = h_T`

Mental model:

```text
Source
↓
Repeated recurrent updates
↓
Final source representation
```

---

# ⭐ 8. Context Vector

Classic Seq2Seq uses:

# `c = h_T`

The context is:

> a learned compressed representation of the entire source sequence.

It is **not**:

> an exact copy of every source token.

---

# 🔗 9. Decoder Initialization

Simple case:

# `d_0 = c`

If dimensions differ:

# `d_0 = W_c c + b_c`

So:

```text
Encoder Context
↓
Decoder Starting State
```

---

# 🚀 10. `<SOS>` Starts Generation

The first decoder input is:

# `<SOS>` / `<BOS>`

Flow:

```text
<SOS>
↓
Decoder
↓
First Target Token
```

---

# 🛑 11. `<EOS>` Stops Generation

Decoder keeps generating until:

# `<EOS>`

is predicted.

This gives:

> variable-length target generation.

---

# 🔄 12. Decoder Is Autoregressive

Generation:

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

Each next token depends on:

* previous target tokens
* decoder state
* source information

---

# 📐 13. Core Probability Equation

Seq2Seq models:

# `P(Y | X)`

with:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

This is the key autoregressive factorization.

---

# 🧠 14. Decoder Recurrence

A simplified decoder:

# `d_t = f_dec(e(y_(t-1)), d_(t-1), c)`

where:

* `e(y_(t-1))` = previous target-token embedding
* `d_(t-1)` = previous decoder state
* `c` = source context

Some implementations use context only to initialize the decoder.

---

# 📐 15. Hidden State → Token Probability

Decoder state:

`d_t`

Output logits:

# `z_t = W_o d_t + b_o`

Then:

# `P(y_t | ...) = softmax(z_t)`

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
Vocabulary Distribution
```

---

# 📐 16. Output Projection Shape

If:

* decoder hidden size = `H`
* target vocabulary size = `V_tgt`

then:

```text
d_t ∈ R^H

W_o ∈ R^(V_tgt × H)

b_o ∈ R^V_tgt

z_t ∈ R^V_tgt
```

---

# ⭐ 17. Target Shifting During Training

Suppose target is:

```text
A B C <EOS>
```

Decoder input:

```text
<SOS> A B C
```

Expected output:

```text
A B C <EOS>
```

Alignment:

```text
Input:   <SOS>   A   B   C
Target:    A     B   C  <EOS>
```

This teaches:

> previous tokens → next token

---

# 🎓 18. Teacher Forcing Intuition

During training:

```text
True Previous Token
↓
Predict Next Token
```

During inference:

```text
Model's Previous Prediction
↓
Predict Next Token
```

This difference is crucial.

---

# ⚠️ 19. Training vs Inference

### Training

Correct previous target may be available.

### Inference

Only model-generated history is available.

So:

# **Training flow ≠ inference flow**

This can cause:

> error propagation.

---

# 🚨 20. Error Propagation

```text
Wrong Token at t
↓
Wrong Next Decoder Input
↓
Changed Decoder State
↓
Possible Later Errors
```

Early decoding errors can cascade.

---

# 🧮 21. Sequence Probability Example

Suppose:

```text
P(A|X) = 0.8
P(B|A,X) = 0.6
P(<EOS>|A,B,X) = 0.9
```

Then:

`P(Y|X) = 0.8 × 0.6 × 0.9`

# `= 0.432`

---

# 📐 22. Training Loss

Per-token loss:

# `L_t = -log P(y_t^correct)`

Total sequence loss:

# `L = Σ_t L_t`

usually excluding padding.

---

# 📦 23. Padding and Masking

Example target:

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

Important:

> **PAD should not contribute to meaningful target loss.**

---

# 🔄 24. End-to-End Learning

Even though loss is calculated at decoder outputs:

```text
Loss
↓
Decoder
↓
Context
↓
Encoder
```

So:

> both encoder and decoder are trained jointly.

---

# 🔗 25. BPTT Happens on Both Sides

```text
Encoder
→ BPTT through source sequence

Decoder
→ BPTT through target sequence
```

Gradients also flow across:

> encoder → decoder state transfer.

---

# 🧠 26. LSTM Seq2Seq

Encoder returns:

```text
h_T^enc
c_T^enc
```

Decoder may initialize:

# `h_0^dec = h_T^enc`

# `c_0^dec = c_T^enc`

or projected versions.

---

# 🧠 27. GRU Seq2Seq

GRU has only:

`h_T`

So:

# `h_0^dec = h_T^enc`

or a transformed version.

No separate:

`c_T`.

---

# ↔️ 28. Bidirectional Encoder

A common design:

```text
BiLSTM / BiGRU Encoder
↓
Context
↓
Unidirectional Decoder
```

Why?

```text
Source
→ fully available

Target future
→ not available
```

So:

> encoder may be bidirectional
> autoregressive decoder usually cannot

---

# 📐 29. Bidirectional Context Dimension

If:

```text
→h_T ∈ R^H

←h_1 ∈ R^H
```

and concatenate:

# `c = [→h_T ; ←h_1]`

then:

# `c ∈ R^(2H)`

---

# ⭐ 30. Greedy Decoding

At each step:

> choose highest-probability token.

Example:

```text
A → 0.6
B → 0.3
C → 0.1
```

Choose:

`A`

Simple, but:

> locally best does not guarantee globally best sequence.

This motivates:

# **Beam Search**

---

# 🚨 31. Classic Seq2Seq Bottleneck

Classic architecture:

```text
x1 x2 x3 ... xT
↓
Encoder
↓
ONE Context Vector
↓
Decoder
```

Whether source has:

```text
5 tokens
50 tokens
500 tokens
```

the decoder still receives:

> the same fixed-dimensional communication channel.

---

# 🧠 32. Why Bigger Context Is Not a Complete Fix

Increasing:

```text
256 → 2048 dimensions
```

may improve capacity.

But the structure remains:

```text
Whole Source
↓
One Vector
↓
Decoder
```

The decoder still cannot directly retrieve:

```text
h_7
h_20
h_54
```

when needed.

---

# ⭐ 33. Why This Leads to Attention

Instead of only:

`c = h_T`

retain:

```text
h1, h2, ..., hT
```

Then allow each decoder step to ask:

> Which encoder states matter right now?

This leads to:

# **Attention**

---

# 🚫 34. High-Yield Traps

### ❌ Seq2Seq requires equal source and target lengths

No.

### ❌ Source and target vocabularies must match

No.

### ❌ Token IDs are embeddings

No.

### ❌ Decoder hidden state is already token probabilities

No.

### ❌ Training and inference are identical

No.

### ❌ Decoder can safely use future target tokens

No.

### ❌ PAD should contribute to target loss

No.

### ❌ Greedy decoding always gives best sequence

No.

### ❌ Bidirectional encoder removes fixed-context bottleneck

No.

### ❌ Bigger hidden size completely solves long-source issues

No.

---

# 📐 35. Formula Flash Card

### Encoder

`h_t = f_enc(x_t, h_(t-1))`

### Context

# `c = h_T`

### Decoder

`d_t = f_dec(e(y_(t-1)), d_(t-1), c)`

### Output

`z_t = W_o d_t + b_o`

### Probability

# `P(y_t | y_<t, X) = softmax(z_t)`

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Loss

# `L_t = -log P(y_t^correct)`

# `L = Σ_t L_t`

with padding masked.

---

# 🎤 36. 30-Second Interview Answer

> **A Seq2Seq model maps one sequence to another using an encoder and autoregressive decoder. Source and target tokens are embedded, the encoder converts the source into a context representation, and the decoder starts from `<SOS>` and generates one target token at a time until `<EOS>`. During training the target is shifted so the model learns next-token prediction, often with teacher forcing, while inference feeds back the model's own predictions. The system is trained end-to-end with token-level cross-entropy. Classic Seq2Seq's major limitation is the fixed context vector, which becomes a bottleneck for long inputs.**

---

# 🧠 12 Things You Must Know Cold

1. **Seq2Seq = Sequence → Sequence**
2. **Usually Encoder + Decoder**
3. **Source tokens use embeddings**
4. **Target tokens use embeddings**
5. **Source/target vocabularies can differ**
6. **Classic context = final encoder state**
7. **`<SOS>` starts generation**
8. **`<EOS>` stops generation**
9. **Target is shifted during training**
10. **Decoder is autoregressive**
11. **Training and inference differ**
12. **Fixed context bottleneck motivates attention**

---

# 🧠 Final Mental Model

```text
SOURCE IDS
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
 LOGITS
   │
   ▼
 SOFTMAX
   │
   ▼
 NEXT TOKEN
   │
   └──────────────→ feed into next decoder step
```

Training:

```text
Input:
<SOS> y1 y2

Target:
y1    y2 <EOS>
```

Inference:

```text
<SOS>
↓
prediction
↓
feed prediction back
↓
next prediction
```

Evolution:

```text
Classic Seq2Seq
↓
One Fixed Context
↓
Long-Sequence Bottleneck
↓
Need Dynamic Encoder-State Access
↓
Attention
```

---

# ⭐ Golden Rule

> **Seq2Seq is conditional autoregressive generation: encode the source, condition the decoder on that source representation, predict the target token-by-token, and remember that classic Seq2Seq's central weakness is forcing the entire source through one fixed context vector.**
