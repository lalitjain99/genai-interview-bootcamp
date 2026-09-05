# 💡 Luong / Multiplicative Attention — Assignment

> Goal: Practice dot scoring, general scoring, shape reasoning, softmax normalization, attentional hidden state construction, global vs local attention, and comparison with Bahdanau Attention.

---

# 🧮 1. Compute a Dot Attention Score

Suppose:

```text
s_t = [1, 2, 3]

h_i = [4, 1, 2]
```

Compute:

```text
e_(t,i) = s_t^T h_i
```

### Answer

```text
e_(t,i)
=
1×4
+
2×1
+
3×2
```

```text
=
4 + 2 + 6
```

```text
=
12
```

Therefore:

```text
e_(t,i) = 12
```

---

# 🧠 2. Can Dot Attention Be Used Directly?

Suppose:

```text
s_t ∈ R^256

h_i ∈ R^512
```

Can you compute:

```text
s_t^T h_i
```

directly?

### Answer

No.

A standard dot product requires compatible dimensions.

Here:

```text
256 ≠ 512
```

So direct dot attention is not valid.

A General Luong score can be used instead.

---

# 📐 3. Design the General Attention Matrix

Suppose:

```text
s_t ∈ R^256

h_i ∈ R^512
```

What should be the shape of:

```text
W_a
```

so that:

```text
s_t^T W_a h_i
```

is valid?

### Answer

We want:

```text
W_a h_i
```

to produce:

```text
R^256
```

Therefore:

```text
W_a ∈ R^(256 × 512)
```

Then:

```text
W_a h_i ∈ R^256
```

and:

```text
s_t^T W_a h_i
```

produces a scalar.

---

# 🧮 4. Compute a General Attention Score

Suppose:

```text
s_t =
[1
 2]
```

```text
h_i =
[3
 4
 5]
```

and:

```text
W_a =
[1 0 0
 0 1 0]
```

Compute:

```text
e_(t,i)
=
s_t^T W_a h_i
```

### Answer

First:

```text
W_a h_i
=
[3
 4]
```

Then:

```text
s_t^T W_a h_i

=
[1, 2]
·
[3, 4]
```

```text
=
1×3 + 2×4
```

```text
=
3 + 8
```

```text
=
11
```

Therefore:

```text
e_(t,i) = 11
```

---

# 🧠 5. Dot or General?

Identify which scoring function is more appropriate.

### Case A

```text
s_t ∈ R^128

h_i ∈ R^128
```

and you want the simplest parameter-free scorer.

### Answer

Use:

```text
Dot Attention
```

```text
e_(t,i) = s_t^T h_i
```

---

### Case B

```text
s_t ∈ R^256

h_i ∈ R^512
```

### Answer

Use:

```text
General Attention
```

with:

```text
W_a ∈ R^(256 × 512)
```

---

# 🔍 6. Find the Shape Bug

A developer writes:

```text
s_t ∈ R^128

h_i ∈ R^256

W_a ∈ R^(256 × 128)
```

and tries:

```text
s_t^T W_a h_i
```

What is wrong?

### Answer

For:

```text
W_a h_i
```

to be valid, the second dimension of `W_a` must equal the dimension of `h_i`.

So `W_a` should be:

```text
R^(128 × 256)
```

not:

```text
R^(256 × 128)
```

Then:

```text
W_a h_i ∈ R^128
```

which matches `s_t`.

---

# 🧠 7. What Does `W_a` Actually Learn?

A candidate says:

> "`W_a` only fixes the vector dimensions."

Is this complete?

### Answer

No.

Dimension matching is one role.

More importantly, `W_a` learns:

> how the encoder representation should be transformed so that a multiplicative comparison with the decoder state becomes useful.

So:

```text
W_a
=
dimension bridge
+
learned compatibility transformation
```

---

# 🧮 8. Compute Attention Weights

Suppose the Luong scores for four source positions are:

```text
e_t = [1, 2, 3, 0]
```

Approximate the softmax attention distribution.

Use:

```text
exp(0) = 1
exp(1) ≈ 2.72
exp(2) ≈ 7.39
exp(3) ≈ 20.09
```

### Answer

Total:

```text
2.72 + 7.39 + 20.09 + 1
=
31.20
```

Therefore:

```text
α_1 ≈ 2.72 / 31.20 ≈ 0.087

α_2 ≈ 7.39 / 31.20 ≈ 0.237

α_3 ≈ 20.09 / 31.20 ≈ 0.644

α_4 ≈ 1 / 31.20 ≈ 0.032
```

So:

```text
α_t ≈ [0.087, 0.237, 0.644, 0.032]
```

The third source position receives the strongest attention.

---

# 🧮 9. Compute the Context Vector

Suppose:

```text
h1 = [1, 0]

h2 = [0, 1]

h3 = [1, 1]
```

and:

```text
α_t = [0.2, 0.3, 0.5]
```

Compute:

```text
c_t = Σ_i α_(t,i) h_i
```

### Answer

```text
c_t
=
0.2[1,0]
+
0.3[0,1]
+
0.5[1,1]
```

```text
=
[0.2,0]
+
[0,0.3]
+
[0.5,0.5]
```

```text
=
[0.7,0.8]
```

Therefore:

```text
c_t = [0.7,0.8]
```

---

# 🧠 10. Score vs Weight vs Context

Match each term to its role:

```text
s_t^T h_i
```

```text
α_(t,i)
```

```text
c_t
```

### Answer

```text
s_t^T h_i
→ raw compatibility score
```

```text
α_(t,i)
→ normalized attention weight
```

```text
c_t
→ weighted source context
```

---

# 🔍 11. Is Dot Product the Same as Cosine Similarity?

Suppose:

```text
a = [1, 0]

b = [10, 0]
```

What are:

```text
a^T b
```

and cosine similarity?

### Answer

Dot product:

```text
1×10 + 0×0 = 10
```

Cosine similarity:

```text
cos(θ) = 1
```

So:

```text
dot product = 10
cosine similarity = 1
```

This shows that dot product depends on vector magnitude.

---

# 🧠 12. Why Can Large Dot Products Matter?

Suppose the scores become:

```text
[2, 4, 20]
```

What happens after softmax?

### Answer

The distribution becomes very sharp, with most probability mass going to the largest score.

So large vector magnitudes can produce:

```text
large dot products
↓
sharp softmax
↓
very concentrated attention
```

This intuition becomes important later for:

> scaled dot-product attention.

---

# 🔍 13. Find the Wrong Softmax Dimension

Suppose score tensor shape is:

```text
[batch_size, source_length]
```

A developer runs:

```text
softmax(scores, dim=0)
```

What is wrong?

### Answer

`dim=0` normalizes across different examples in the batch.

Luong attention should typically normalize across:

```text
source positions
```

So for:

```text
[batch, source]
```

the correct dimension is usually:

```text
dim=1
```

---

# 🧠 14. Current Decoder State vs Previous State

A candidate says:

> "Luong Attention always uses `s_t`, while Bahdanau always uses `s_(t-1)`."

Is this safe?

### Answer

No.

A more accurate statement is:

> Luong-style attention is commonly described using the current decoder state, while Bahdanau-style attention is commonly described using the previous decoder state during alignment computation.

Exact notation and ordering can vary by implementation.

---

# 🧮 15. Attentional Hidden State Shape

Suppose:

```text
c_t ∈ R^512

s_t ∈ R^256
```

What is the dimension of:

```text
[c_t ; s_t]
```

### Answer

Concatenation gives:

```text
512 + 256 = 768
```

So:

```text
[c_t ; s_t] ∈ R^768
```

If we want:

```text
h̃_t ∈ R^256
```

then:

```text
W_c ∈ R^(256 × 768)
```

can be used.

---

# 🧠 16. Why Is the Attentional Hidden State Useful?

Why not use only:

```text
c_t
```

for output prediction?

### Answer

Because `c_t` mainly represents:

> retrieved source information

while `s_t` contains:

> decoder / target-history information.

Combining them gives:

```text
source evidence
+
target-generation state
```

which is more informative for predicting the next token.

---

# 🌍 17. Global or Local Attention?

Suppose the source sequence contains 2,000 positions.

At every target timestep, Global Attention scores all 2,000 source positions.

What problem might arise?

### Answer

The attention computation can become expensive.

Global attention performs source-wide comparisons at every decoder step.

This motivates Local Attention, which limits attention to a smaller source region.

---

# 🧠 18. Local Attention Trade-Off

Suppose Local Attention selects:

```text
positions 40–50
```

but the important source detail lies at position:

```text
120
```

What happens?

### Answer

The model may fail to retrieve that information because the relevant position lies outside the attention window.

So Local Attention trades:

```text
lower computation
```

for:

```text
reduced global accessibility
```

---

# 🔍 19. Is Luong Attention Self-Attention?

Architecture:

```text
LSTM Encoder
↓
h1 ... hT

LSTM Decoder
↓
s_t

s_t attends to h1 ... hT
```

Is this self-attention?

### Answer

No.

This is:

> encoder-decoder attention.

The decoder representation attends to encoder representations.

Self-attention instead has positions within a representation sequence attending to other positions in that sequence.

---

# 🧠 20. Compare Bahdanau and Luong

Fill the table mentally:

| Property                 | Bahdanau | Luong |
| ------------------------ | -------- | ----- |
| Interaction              | ?        | ?     |
| Nonlinearity             | ?        | ?     |
| Dot product              | ?        | ?     |
| Extra scoring parameters | ?        | ?     |
| Matrix friendly          | ?        | ?     |

### Answer

| Property                 | Bahdanau | Luong              |
| ------------------------ | -------- | ------------------ |
| Interaction              | Additive | Multiplicative     |
| Nonlinearity             | Yes      | No for Dot/General |
| Dot product              | No       | Yes                |
| Extra scoring parameters | More     | None/Fewer         |
| Matrix friendly          | Less     | More               |

---

# ✅ True / False

## 1.

Luong Dot Attention uses:

```text
s_t^T h_i
```

**Answer:** True.

---

## 2.

Luong Dot Attention can directly compare any two hidden-state dimensions.

**Answer:** False.

---

## 3.

Luong General Attention introduces a learned matrix.

**Answer:** True.

---

## 4.

`W_a` is the dynamically computed source attention distribution.

**Answer:** False.

---

## 5.

Dot-product scoring can be implemented efficiently with matrix multiplication.

**Answer:** True.

---

## 6.

Dot product and cosine similarity are always identical.

**Answer:** False.

---

## 7.

Luong Attention completely removes recurrence.

**Answer:** False.

---

## 8.

Global Attention considers all source positions.

**Answer:** True.

---

## 9.

Local Attention can reduce computation.

**Answer:** True.

---

## 10.

Local Attention can potentially miss relevant distant source information.

**Answer:** True.

---

## 11.

The context vector is usually a weighted sum of encoder states.

**Answer:** True.

---

## 12.

Luong Attention is Transformer self-attention.

**Answer:** False.

---

# ⭐ Staff Engineer Challenge

You have a production sequence-to-sequence system:

```text
BiLSTM Encoder
+
Bahdanau Attention
+
LSTM Decoder
```

The model produces good output, but the attention scoring stage is consuming significant latency.

The encoder and decoder hidden states both have dimension:

```text
512
```

The team proposes replacing Bahdanau scoring with:

```text
e_(t,i) = s_t^T h_i
```

### Questions

1. Is the proposed score dimensionally valid?
2. Why could it reduce compute?
3. What model-quality risk does the change introduce?
4. Would you immediately deploy the change?
5. Does it solve the larger decoding bottleneck?

---

## Strong Answer

### 1. Is it dimensionally valid?

Yes.

Both vectors are:

```text
R^512
```

so direct dot-product scoring is valid.

---

### 2. Why might it reduce compute?

Bahdanau performs:

```text
projections
+
addition
+
tanh
+
scalar readout
```

for compatibility scoring.

Dot scoring performs:

```text
s_t^T h_i
```

which can be efficiently vectorized as matrix multiplication.

---

### 3. What is the quality risk?

Bahdanau has a learned nonlinear compatibility network.

Plain dot attention assumes the existing encoder and decoder representations are already aligned enough for direct dot-product compatibility.

So model quality may change.

---

### 4. Would you immediately deploy it?

No.

Benchmark both.

Evaluate:

* task quality
* long-input behavior
* entity preservation
* alignment quality
* p50/p95/p99 latency
* throughput
* compute utilization
* memory

Then make the trade-off based on measured results.

---

### 5. Does it solve the larger decoding bottleneck?

No.

The decoder remains:

> autoregressive and recurrent.

Each target step still depends on the previous decoder state.

So cheaper attention scoring reduces one part of latency but does not eliminate sequential generation.

---

# 🧠 Final Exercise — Build Luong from Generic Attention

Start with:

```text
e_(t,i)
=
score(s_t, h_i)
```

Fill in two possible score functions.

### Answer

### Dot

```text
score(s_t, h_i)
=
s_t^T h_i
```

### General

```text
score(s_t, h_i)
=
s_t^T W_a h_i
```

Then:

```text
scores
↓
softmax
↓
α_(t,i)
↓
weighted sum
↓
c_t
```

Optionally combine:

```text
[c_t ; s_t]
↓
W_c + tanh
↓
h̃_t
```

for prediction.

---

# 🎯 Assignment Mental Model

You should now be able to reconstruct:

```text
Dot:
s_t^T h_i
```

and:

```text
General:
s_t^T W_a h_i
```

and explain the chain:

```text
Decoder State
+
Encoder States
↓
Multiplicative Scoring
↓
Scalar Scores
↓
Softmax
↓
Attention Weights
↓
Weighted Context
↓
Combine with Decoder State
↓
Prediction
```

The shortest memory version is:

# **Transform if needed → Multiply → Normalize → Aggregate → Combine**

---

# 🔗 Next Topic

We now know how classical attention calculates alignment using both:

```text
Bahdanau
→ Additive Scoring

Luong
→ Multiplicative Scoring
```

The next question is:

> **What do these alignment distributions look like across complete source and target sequences, and what can we actually infer from them?**

That takes us to:

# **05_Attention_Alignment_And_Visualization**
