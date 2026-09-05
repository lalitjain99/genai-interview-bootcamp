# ⚡ Luong / Multiplicative Attention — Cheat Sheet

> Goal: Reconstruct Luong Attention in a few minutes before an interview.

---

# 1. 🎯 One-Line Definition

> **Luong Attention computes compatibility between the decoder state and encoder states using multiplicative scoring such as a direct dot product or a learned bilinear transformation.**

Also commonly called:

# **Multiplicative Attention**

---

# 2. 🧠 Why Luong Attention?

Bahdanau computes compatibility using:

```text
project
+
add
+
tanh
+
scalar projection
```

Luong asks:

> **Can compatibility be computed more directly through multiplication?**

Answer:

```text
Yes
```

using:

```text
dot product
```

or:

```text
general / bilinear scoring
```

---

# 3. 📐 Core Scoring Equations

## Dot

```text
e_(t,i)
=
s_t^T h_i
```

## General

```text
e_(t,i)
=
s_t^T W_a h_i
```

Then:

```text
α_(t,i)
=
softmax_i(e_(t,i))
```

and:

```text
c_t
=
Σ_i α_(t,i) h_i
```

---

# 4. ⭐ Core Mental Model

# **Transform if needed → Multiply → Normalize → Aggregate**

Full flow:

```text
Decoder State
+
Encoder State
↓
Multiplicative Compatibility
↓
Scalar Score
↓
Softmax
↓
Attention Weight
↓
Weighted Context
```

---

# 5. 🧠 Dot Attention

Dot attention uses:

```text
e_(t,i)
=
s_t^T h_i
```

Requirement:

```text
dim(s_t) = dim(h_i)
```

Example:

```text
s_t = [1, 2]

h_i = [3, 4]
```

Then:

```text
e_(t,i)

=
1×3 + 2×4

=
11
```

---

# 6. 🚨 Dot Product Requires Matching Dimensions

Works:

```text
s_t ∈ R^256
h_i ∈ R^256
```

Does not work directly:

```text
s_t ∈ R^256
h_i ∈ R^512
```

For mismatched dimensions, use:

# **General Attention**

---

# 7. 📐 General Attention

```text
e_(t,i)
=
s_t^T W_a h_i
```

If:

```text
s_t ∈ R^(H_d)

h_i ∈ R^(H_e)
```

choose:

```text
W_a ∈ R^(H_d × H_e)
```

Then:

```text
W_a h_i
∈ R^(H_d)
```

and the dot product becomes valid.

---

# 8. 🧠 What Does `W_a` Do?

`W_a` is not just a dimension fixer.

It learns:

> how the encoder representation should be transformed so that multiplication with the decoder state produces a useful compatibility score.

Mental model:

```text
Encoder State
↓
W_a
↓
Decoder-compatible representation
↓
Dot with decoder state
↓
Score
```

---

# 9. ⭐ Dot vs General

## Dot

```text
s_t^T h_i
```

* simplest
* no extra scoring parameters
* requires compatible dimensions
* highly efficient

## General

```text
s_t^T W_a h_i
```

* learned transformation
* handles representation mismatch
* more flexible
* adds parameters

---

# 10. 🧠 Why Is It Called Multiplicative?

Because the core interaction is based on multiplication:

```text
s_t^T h_i
```

or:

```text
s_t^T W_a h_i
```

instead of:

```text
W_s s + W_h h
```

as in additive attention.

---

# 11. 📐 Complete Luong Pipeline

### Score

```text
e_(t,i)
=
s_t^T h_i
```

or:

```text
e_(t,i)
=
s_t^T W_a h_i
```

### Weight

```text
α_(t,i)
=
softmax_i(e_(t,i))
```

### Context

```text
c_t
=
Σ_i α_(t,i) h_i
```

---

# 12. 🧠 One Score Per Source Position

For:

```text
h_1, h_2, ..., h_T
```

Luong computes:

```text
e_(t,1)
e_(t,2)
...
e_(t,T)
```

So:

```text
e_t ∈ R^T
```

Then:

```text
softmax(e_t)
```

gives:

```text
α_t ∈ R^T
```

---

# 13. ⚙️ Why Luong Is Efficient

If encoder states are stacked:

```text
H_enc ∈ R^(T × H)
```

and:

```text
s_t ∈ R^H
```

then all dot scores can be computed together:

```text
e_t
=
H_enc s_t
```

giving:

```text
e_t ∈ R^T
```

This maps naturally to:

> matrix multiplication

which is highly efficient on modern hardware.

---

# 14. ⭐ Bahdanau vs Luong

## Bahdanau

```text
e_(t,i)
=
v_a^T tanh(
W_s s
+
W_h h_i
+
b_a
)
```

### Characteristics

* additive interaction
* nonlinear scorer
* more parameters
* explicit attention hidden space

---

## Luong

```text
e_(t,i)
=
s_t^T h_i
```

or:

```text
e_(t,i)
=
s_t^T W_a h_i
```

### Characteristics

* multiplicative interaction
* simpler scorer
* fewer parameters
* matrix-friendly

---

# 15. 📊 Quick Comparison

| Aspect                    | Bahdanau            | Luong              |
| ------------------------- | ------------------- | ------------------ |
| Interaction               | Additive            | Multiplicative     |
| Nonlinearity in score     | Yes                 | No for Dot/General |
| Score parameters          | More                | Few / none         |
| Attention hidden space    | Explicit            | Not necessarily    |
| Matrix efficiency         | Lower               | Higher             |
| Dimension mismatch        | Projection matrices | General form       |
| Classical recurrent setup | Yes                 | Yes                |

---

# 16. 🧠 Decoder-State Timing

Common historical pattern:

### Bahdanau

```text
s_(t-1)
```

often used for alignment.

### Luong

```text
s_t
```

often used for alignment.

Luong conceptual flow:

```text
previous target info
↓
compute s_t
↓
attend to source
↓
compute c_t
↓
combine s_t and c_t
↓
predict
```

Do not treat this as an absolute rule across every implementation.

---

# 17. 🧠 Attentional Hidden State

Luong commonly combines:

```text
s_t
```

and:

```text
c_t
```

using:

```text
[c_t ; s_t]
```

Then:

```text
h̃_t
=
tanh(
W_c [c_t ; s_t]
)
```

This is called the:

# **Attentional Hidden State**

---

# 18. 🧠 Why Combine `c_t` and `s_t`?

Because:

```text
s_t
→ target-side decoding information
```

while:

```text
c_t
→ source-side retrieved information
```

Together:

```text
target history
+
source evidence
```

give a richer representation for output prediction.

---

# 19. 🚨 Important Role Separation

```text
s_t
→ current decoder state

e_(t,i)
→ raw compatibility score

α_(t,i)
→ normalized source relevance

c_t
→ weighted source context

h̃_t
→ combined decoder + source representation
```

---

# 20. 🌍 Global vs Local Attention

## Global

```text
compare against all encoder states
```

Pros:

* full source access

Cons:

* more compute for long sequences

---

## Local

```text
predict where to look
↓
attend within smaller source window
```

Pros:

* lower compute

Cons:

* may miss relevant distant information

---

# 21. 🚨 Dot Product ≠ Cosine Similarity

Dot product:

```text
s_t^T h_i
```

depends on:

* direction
* magnitude

Cosine similarity normalizes magnitudes.

So:

```text
dot product ≠ cosine similarity
```

unless vectors are normalized.

---

# 22. 🧠 Why Vector Magnitude Matters

Large vector norms can produce large dot-product scores.

That can lead to:

```text
large scores
↓
sharp softmax
↓
very concentrated attention
```

This becomes important later for:

# **Scaled Dot-Product Attention**

---

# 23. 📐 Parameter Count

## Dot

Additional scoring parameters:

```text
0
```

if dimensions already match.

## General

```text
W_a ∈ R^(H_d × H_e)
```

Parameter count:

```text
H_d × H_e
```

---

# 24. 🔗 Relation to Transformers

Luong introduces an important idea:

```text
compatibility through dot products
```

This later evolves into:

```text
Scaled Dot-Product Attention
```

But:

> Luong Attention is still classical recurrent encoder-decoder attention.

It is not Transformer self-attention.

---

# ⚡ 10 Things to Know Cold

1. Luong = Multiplicative Attention.
2. Dot score = `s_t^T h_i`.
3. General score = `s_t^T W_a h_i`.
4. Dot attention requires matching dimensions.
5. `W_a` is a learned compatibility transformation.
6. Softmax runs across source-position scores.
7. Context is the weighted sum of encoder states.
8. Luong commonly uses the current decoder state.
9. Multiplicative scoring is matrix-multiplication friendly.
10. Luong is not Transformer self-attention.

---

# 🚫 High-Yield Traps

### ❌ Luong means only Dot Attention

No.

General is also important.

---

### ❌ `W_a` is the attention weight matrix

No.

It is a learned scoring parameter.

---

### ❌ Equal dimensions mean Dot Attention must be best

No.

Equal dimensions only make direct dot product possible.

---

### ❌ Dot product is cosine similarity

No.

Magnitude matters.

---

### ❌ Luong removes recurrence

No.

Classic Luong Attention is used with recurrent encoder-decoder models.

---

### ❌ Local Attention always preserves full-source quality

No.

It can miss useful information outside the selected window.

---

### ❌ Luong Attention is self-attention

No.

It is encoder-decoder attention.

---

# 🎤 30-Second Interview Answer

> **Luong Attention is a classical encoder-decoder attention mechanism that uses multiplicative compatibility scoring. Dot attention computes `s_t^T h_i`, while General attention computes `s_t^T W_a h_i`. The resulting scores are softmax-normalized across source positions and used to form a weighted context vector. Compared with Bahdanau's nonlinear additive scorer, Luong's main variants are simpler and typically more efficient for matrix computation.**

---

# 🎤 Dot vs General Short Answer

> **Dot attention directly compares decoder and encoder states and therefore requires compatible dimensions. General attention introduces a learned matrix `W_a` before the dot product, allowing the model to transform encoder representations into a space suitable for comparison with the decoder state.**

---

# 🎤 Bahdanau vs Luong Short Answer

> **Bahdanau uses a nonlinear additive scoring network, while Luong Dot and General attention use multiplicative interactions. Luong is generally simpler and more matrix-friendly, while Bahdanau provides a richer learned scoring transformation.**

---

# 🧠 Final Mental Model

```text
Decoder State
       │
       │
       ▼

Encoder State
       │
   transform if needed
       │
      W_a
       ▼

Multiplicative Interaction
       ↓
Scalar Score
       ↓
Softmax
       ↓
Attention Weight
       ↓
Weighted Context
       ↓
Combine with Decoder State
       ↓
Prediction
```

Shortest memory version:

# **Transform if needed → Multiply → Normalize → Aggregate → Combine**

---

# 🔗 Next Topic

We now understand both major classical scoring families:

```text
Bahdanau
→ Additive

Luong
→ Multiplicative
```

The next question is:

> **What do these attention distributions actually look like over a full source-target sequence, and how should we interpret them?**

That leads to:

# **05_Attention_Alignment_And_Visualization**
