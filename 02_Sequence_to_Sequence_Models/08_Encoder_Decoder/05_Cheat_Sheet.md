# ⚡ Cheat Sheet — Encoder–Decoder Architecture

> **Goal:** Fast revision of why Encoder–Decoder exists, how information flows from source to target, how decoding works, and why the fixed context vector becomes a bottleneck.

---

# 🌟 1. Encoder–Decoder in One Line

> **Encoder–Decoder = understand the input sequence, then generate the output sequence.**

```text
Source Sequence
↓
Encoder
↓
Context / Representation
↓
Decoder
↓
Target Sequence
```

---

# 🧠 2. Why Was It Needed?

Earlier models often solved:

```text
Sequence
→ Label
```

But many tasks require:

```text
Sequence
→ Sequence
```

Examples:

* translation
* summarization
* speech-to-text
* response generation

---

# ⭐ 3. Encoder Role

The encoder reads:

`x_1, x_2, ..., x_T`

and updates:

`h_t = f_enc(x_t, h_(t-1))`

In the original architecture:

# `c = h_T`

where `c` is the context vector.

Think:

> **Encoder = What did I read?**

---

# 🧠 4. Context Vector

The context vector is:

> a learned fixed-dimensional representation of the source sequence.

Important:

```text
Context
≠ exact copy of all source tokens
```

It is:

> compressed task-relevant information.

---

# ⭐ 5. Decoder Role

The decoder generates the target sequence one token at a time.

Conceptually:

`d_t = f_dec(y_(t-1), d_(t-1), c)`

Think:

> **Decoder = What should I generate next, given the source and what I already generated?**

---

# 🚀 6. How Decoding Starts

Use:

# `<SOS>` / `<BOS>`

Example:

```text
<SOS>
↓
Decoder
↓
y1
```

This provides the decoder's first input.

---

# 🛑 7. How Decoding Stops

The decoder learns to generate:

# `<EOS>`

When `<EOS>` appears:

> generation stops.

This enables:

> variable-length output.

---

# 🔄 8. Autoregressive Decoding

Decoder output is generated sequentially:

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

Each next token depends on previous output tokens.

Probability:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

---

# 📐 9. Decoder State to Token Probability

Decoder state:

`d_t`

Output logits:

`z_t = W_o d_t + b_o`

Then:

`P(y_t | ...) = softmax(z_t)`

Flow:

```text
Decoder State
↓
Linear Layer
↓
Logits
↓
Softmax
↓
Vocabulary Probabilities
```

---

# 🧮 10. Vocabulary Shape Example

If:

```text
Decoder hidden size = H
Vocabulary size = V
```

then:

```text
d_t ∈ R^H
W_o ∈ R^(V×H)
b_o ∈ R^V
z_t ∈ R^V
```

Softmax gives:

> one probability per vocabulary token.

---

# 🔗 11. How Encoder Connects to Decoder

A simple setup:

# `d_0 = h_T`

If dimensions differ:

# `d_0 = W_c h_T + b_c`

So:

```text
Encoder Final State
↓
Decoder Initial State
```

---

# 🧠 12. LSTM vs GRU State Transfer

## LSTM

May transfer:

```text
h_T^enc → h_0^dec
c_T^enc → c_0^dec
```

## GRU

Usually transfers:

```text
h_T^enc → h_0^dec
```

because GRU has no separate cell state.

---

# ↔️ 13. Input and Output Lengths Can Differ

Encoder may read:

```text
x1 x2 x3 x4 x5
```

Decoder may generate:

```text
y1 y2 y3 <EOS>
```

So:

# `T_input ≠ T_output`

This is a key strength.

---

# 🔗 14. Encoder Can Often Be Bidirectional

If the complete source sequence is known:

```text
Source
→ full sequence available
```

then encoder can use:

* BiLSTM
* BiGRU

because both left and right source context are available.

---

# 🚫 15. Decoder Is Usually Causal

For autoregressive decoding, when predicting:

`y_t`

future target tokens:

`y_(t+1), y_(t+2), ...`

do not exist yet.

So a decoder usually cannot be bidirectional.

Remember:

```text
Encoder
→ can often use both directions

Decoder
→ usually causal
```

---

# 🎓 16. Training vs Inference

## Training

True previous target token may be available.

```text
True Previous Token
↓
Predict Next Token
```

## Inference

Decoder uses its own previous prediction.

```text
Predicted Token
↓
Predict Next Token
```

This difference leads to:

# **Teacher Forcing**

---

# ⚠️ 17. Error Propagation

During inference:

```text
Wrong prediction at t
↓
Wrong input/history at t+1
↓
Possible later errors
```

So autoregressive generation can suffer from:

> error accumulation.

---

# 🔗 18. Encoder–Decoder vs Aligned Many-to-Many

Aligned many-to-many:

```text
x1 → y1
x2 → y2
x3 → y3
```

Example:

> POS tagging

Encoder–Decoder:

```text
x1 x2 x3 x4
↓
Encoder
↓
Context
↓
Decoder
↓
y1 y2 y3
```

No strict one-to-one alignment is required.

---

# 🚨 19. Fixed Context Vector Bottleneck

Original architecture does:

```text
h1 h2 h3 ... hT
             │
             ▼
             c
             │
             ▼
          Decoder
```

The entire source is compressed into:

> one fixed-size vector.

Short input:

```text
5 tokens → c
```

Long input:

```text
500 tokens → same-sized c
```

This becomes an:

# **information bottleneck**

---

# 🧠 20. Why Bigger Context Is Not a Complete Fix

Increasing hidden size may help.

But structurally, the model still requires:

```text
Whole Source
↓
One Vector
↓
Decoder
```

The decoder still cannot directly revisit specific encoder positions.

So:

> larger context ≠ removal of bottleneck.

---

# ⭐ 21. What Leads to Attention?

Instead of using only:

`h_T`

retain:

```text
h1, h2, h3, ..., hT
```

Then allow the decoder to ask:

> Which source states matter for the current output token?

This leads to:

# **Attention**

---

# 📖 22. Tiny Translation Walkthrough

Source:

```text
I am happy
```

Encoder:

```text
I      → h1
am     → h2
happy  → h3
```

Context:

`c = h3`

Decoder:

```text
<SOS>
↓
Je
↓
suis
↓
heureux
↓
<EOS>
```

---

# 🔗 23. Encoder–Decoder vs Seq2Seq

### Encoder–Decoder

Architecture pattern:

```text
Encoder
→ Representation
→ Decoder
```

### Seq2Seq

Usually:

```text
Sequence
→ Sequence
```

using an Encoder–Decoder architecture.

Closely related, but conceptually:

> Encoder–Decoder = architecture
> Seq2Seq = sequence-transformation use/design

---

# 🚫 24. High-Yield Traps

### ❌ Encoder generates target tokens

No. Decoder does.

### ❌ Context stores source exactly

No. It is compressed.

### ❌ Input and output lengths must match

No.

### ❌ Decoder state is already a probability distribution

No. Usually needs linear projection + softmax.

### ❌ Decoder can always be bidirectional

No.

### ❌ `<EOS>` is just padding

No. It signals termination.

### ❌ Encoder and decoder must have equal hidden sizes

No. Use projection.

### ❌ Larger context completely solves long-sequence problems

No.

### ❌ Encoder–Decoder automatically includes attention

No.

---

# 📐 Formula Flash Card

### Encoder

`h_t = f_enc(x_t, h_(t-1))`

### Original Context

# `c = h_T`

### Decoder

`d_t = f_dec(y_(t-1), d_(t-1), c)`

### Decoder Initialization

`d_0 = h_T`

or:

`d_0 = W_c h_T + b_c`

### Output

`z_t = W_o d_t + b_o`

`P(y_t | ...) = softmax(z_t)`

### Full Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

---

# 🎤 30-Second Interview Answer

> **Encoder–Decoder is an architecture for mapping one variable-length sequence to another. The encoder processes the source sequence and produces a learned representation, traditionally the final hidden state. The decoder is conditioned on that representation and generates the target sequence autoregressively from a start token until an end token is produced. Its original limitation is the fixed context vector, because the entire source sequence must be compressed into one representation, which later motivated attention.**

---

# 🧠 10 Things You Must Know Cold

1. **Encoder = source representation**
2. **Decoder = target generation**
3. **Original context often = `h_T`**
4. **`<SOS>` starts generation**
5. **`<EOS>` stops generation**
6. **Input/output lengths can differ**
7. **Decoder is autoregressive**
8. **Encoder can often be bidirectional**
9. **Autoregressive decoder is usually causal**
10. **Fixed context bottleneck motivates attention**

---

# 🧠 Final Mental Model

```text
SOURCE
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
TARGET
```

Think:

# **Encoder = What did I read?**

# **Context = What did I understand?**

# **Decoder = What should I generate next?**

And remember the evolution:

```text
Whole Source
↓
One Fixed Context
↓
Too Much Compression
↓
Need Access to Encoder States
↓
Attention
```

---

# ⭐ Golden Rule

> **Encoder–Decoder solves sequence-to-sequence transformation by separating source understanding from target generation, but the original design's dependence on one fixed context vector eventually becomes its central limitation.**
