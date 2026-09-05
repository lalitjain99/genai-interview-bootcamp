# 🎤 Luong / Multiplicative Attention — Interview Questions

> Goal: Be able to explain Luong Attention from intuition, equations, tensor shapes, implementation details, comparison with Bahdanau Attention, and architectural trade-offs.

---

# 🟢 Beginner

## 1. What is Luong Attention?

**Answer:**

Luong Attention is a classical encoder-decoder attention mechanism that computes compatibility between the decoder state and encoder states using multiplicative interactions.

The two most important scoring variants are:

```text
Dot:
e_(t,i) = s_t^T h_i
```

and:

```text
General:
e_(t,i) = s_t^T W_a h_i
```

The scores are then normalized with softmax:

```text
α_(t,i) = softmax_i(e_(t,i))
```

and used to compute the context vector:

```text
c_t = Σ_i α_(t,i) h_i
```

---

## 2. Why is Luong Attention called Multiplicative Attention?

**Answer:**

Because the decoder and encoder representations interact through multiplication.

For example:

```text
s_t^T h_i
```

or:

```text
s_t^T W_a h_i
```

This differs from Bahdanau Attention, which combines projected representations additively before applying a nonlinear scoring function.

---

## 3. What is the difference between Luong Dot and Luong General Attention?

**Answer:**

Dot attention uses:

```text
e_(t,i) = s_t^T h_i
```

It directly compares decoder and encoder states.

General attention uses:

```text
e_(t,i) = s_t^T W_a h_i
```

where:

```text
W_a
```

is a learned matrix.

So:

```text
Dot
→ direct multiplication

General
→ learned transformation + multiplication
```

---

## 4. What requirement does Dot Attention have?

**Answer:**

The decoder and encoder state dimensions must be compatible.

For example:

```text
s_t ∈ R^256
h_i ∈ R^256
```

works.

But:

```text
s_t ∈ R^256
h_i ∈ R^512
```

cannot be directly used with:

```text
s_t^T h_i
```

---

## 5. What does `W_a` do in General Attention?

**Answer:**

`W_a` transforms the encoder representation before comparing it with the decoder state.

If:

```text
s_t ∈ R^(H_d)
h_i ∈ R^(H_e)
```

then choose:

```text
W_a ∈ R^(H_d × H_e)
```

so:

```text
W_a h_i ∈ R^(H_d)
```

Now the score:

```text
s_t^T W_a h_i
```

is valid.

---

# 🟡 Intermediate

## 6. Walk through one complete Luong Attention timestep.

**Answer:**

At decoder timestep `t`:

### Step 1 — Compute decoder state

```text
s_t
```

### Step 2 — Score each encoder state

For Dot:

```text
e_(t,i) = s_t^T h_i
```

or General:

```text
e_(t,i) = s_t^T W_a h_i
```

### Step 3 — Normalize

```text
α_(t,i) = softmax_i(e_(t,i))
```

### Step 4 — Build context

```text
c_t = Σ_i α_(t,i) h_i
```

### Step 5 — Combine context and decoder state

Luong commonly forms:

```text
h̃_t
=
tanh(
W_c [c_t ; s_t]
)
```

### Step 6 — Predict output

The attentional hidden state is then used for the target prediction.

---

## 7. Why is Luong Attention usually more efficient than Bahdanau Attention?

**Answer:**

Luong's dot/general forms use matrix multiplication directly.

For example, if encoder states are stacked:

```text
H_enc ∈ R^(T × H)
```

and:

```text
s_t ∈ R^H
```

then all dot scores can be computed as:

```text
e_t = H_enc s_t
```

This is efficient because modern hardware is heavily optimized for matrix operations.

Bahdanau instead performs:

```text
projection
+
addition
+
tanh
+
scalar readout
```

for compatibility scoring.

So Luong scoring is typically simpler and more matrix-friendly.

---

## 8. What is the attentional hidden state in Luong Attention?

**Answer:**

Luong commonly combines the current decoder state:

```text
s_t
```

with the retrieved source context:

```text
c_t
```

using concatenation:

```text
[c_t ; s_t]
```

and then computes:

```text
h̃_t
=
tanh(
W_c [c_t ; s_t]
)
```

This gives a representation containing:

```text
target-side decoder information
+
source-side attended information
```

---

## 9. Why combine `s_t` and `c_t`?

**Answer:**

Because they contain complementary information.

```text
s_t
→ what the decoder knows from target history

c_t
→ what the decoder retrieved from the source
```

Combining them gives the model both:

> what it has generated so far

and:

> what source information is relevant now.

---

## 10. What is the common timing difference between Bahdanau and Luong Attention?

**Answer:**

A commonly taught historical distinction is:

Bahdanau often uses:

```text
s_(t-1)
```

for alignment.

Luong often uses:

```text
s_t
```

the current decoder state.

Conceptually for Luong:

```text
compute s_t
↓
attend to encoder states
↓
build c_t
↓
combine s_t and c_t
↓
predict
```

But this should not be treated as a universal law across all implementations.

---

# 🔴 Advanced

## 11. Why is Dot Attention parameter-free for scoring?

**Answer:**

Dot attention computes:

```text
e_(t,i) = s_t^T h_i
```

There is no additional learned matrix inside the score function.

The encoder and decoder states themselves are already learned representations.

So the scoring function adds:

```text
0
```

extra parameters, assuming dimensions already match.

---

## 12. Why might plain Dot Attention still be too restrictive?

**Answer:**

Dot attention assumes that encoder and decoder states already live in representation spaces where direct dot-product compatibility is meaningful.

That may be restrictive because:

* the representation spaces may differ;
* the features may not be aligned;
* equal dimensions do not guarantee useful direct comparison.

General attention relaxes this by learning:

```text
W_a
```

before the dot product.

---

## 13. How is General Attention different from Bahdanau Attention?

**Answer:**

General Luong:

```text
e_(t,i)
=
s_t^T W_a h_i
```

is a bilinear multiplicative interaction.

Bahdanau:

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

uses:

* separate projections;
* additive combination;
* nonlinearity;
* scalar projection.

So the structural difference is:

```text
Bahdanau
→ nonlinear additive scorer

Luong General
→ bilinear multiplicative scorer
```

---

## 14. Is Dot Product the same as Cosine Similarity?

**Answer:**

No.

Dot product:

```text
s_t^T h_i
```

depends on:

* direction;
* magnitude.

Cosine similarity normalizes by vector norms:

```text
cos(θ)
=
(s_t^T h_i)
/
(||s_t|| ||h_i||)
```

So two vectors with the same direction but larger magnitudes can produce a larger dot product while having the same cosine similarity.

---

## 15. Why does vector magnitude matter in Dot Attention?

**Answer:**

Because:

```text
s_t^T h_i
```

can become large when vector components or norms become large.

Larger raw scores can lead to a sharper softmax distribution.

This becomes especially important later in Transformer attention, where scaled dot-product attention divides scores by a dimension-dependent factor.

---

# ⭐ Staff Engineer Challenge

## Scenario

You are reviewing two attention implementations for a recurrent translation service.

### Model A

```text
BiLSTM Encoder
+
Bahdanau Attention
+
LSTM Decoder
```

### Model B

```text
BiLSTM Encoder
+
Luong General Attention
+
LSTM Decoder
```

Both have similar translation quality.

However, Model A has noticeably higher decoding latency.

### Questions

1. Why might Model B be faster?
2. Is Model B always expected to produce equal or better quality?
3. What dimensions must `W_a` have?
4. What would you measure before replacing Model A?
5. Does switching to Luong solve autoregressive decoding latency completely?

---

## Strong Answer

### 1. Why might Model B be faster?

Luong General uses:

```text
s_t^T W_a h_i
```

which maps naturally to matrix multiplication.

Bahdanau uses:

```text
projection
+
addition
+
tanh
+
scalar readout
```

for every compatibility computation.

Therefore Luong can often be implemented more efficiently on matrix hardware.

---

### 2. Is quality guaranteed to be equal or better?

No.

Bahdanau's nonlinear scoring may model compatibility differently and sometimes more flexibly.

The scoring choice should be evaluated empirically.

---

### 3. What dimensions should `W_a` have?

If:

```text
s_t ∈ R^(H_d)

h_i ∈ R^(H_e)
```

then:

```text
W_a ∈ R^(H_d × H_e)
```

so:

```text
W_a h_i ∈ R^(H_d)
```

and:

```text
s_t^T W_a h_i
```

produces a scalar.

---

### 4. What would you measure?

Model quality:

* translation quality
* entity preservation
* long-sequence quality
* alignment behavior

System metrics:

* p50 / p95 / p99 latency
* throughput
* memory
* compute utilization
* source-length sensitivity
* target-length sensitivity

---

### 5. Does Luong solve autoregressive latency completely?

No.

The decoder is still recurrent and autoregressive.

Each target token depends on previous decoding state.

So even if attention scoring becomes cheaper:

```text
target generation remains sequential
```

That larger architectural limitation remains.

---

# ⚡ Rapid-Fire Interview Questions

## What is the Dot score?

```text
s_t^T h_i
```

---

## What is the General score?

```text
s_t^T W_a h_i
```

---

## Does Dot Attention require equal dimensions?

Yes, for direct dot product.

---

## Does General Attention add learned parameters?

Yes.

---

## What does `W_a` represent?

A learned transformation used before multiplicative compatibility scoring.

---

## Is `W_a` the attention matrix over source tokens?

No.

---

## Is Luong Attention self-attention?

No.

---

## Is Luong Attention recurrent?

The classic architecture is recurrent.

---

## What does softmax normalize over?

Source-position scores.

---

## What is `c_t`?

The weighted sum of encoder states.

---

## What is `h̃_t`?

The attentional hidden state combining decoder state and attended source context.

---

## Is dot product cosine similarity?

No.

---

# 🚨 High-Yield Interview Traps

### ❌ Luong Attention means only Dot Attention

No.

Luong discusses multiple score forms, including Dot and General.

---

### ❌ General Attention and Dot Attention are identical

No.

General introduces:

```text
W_a
```

---

### ❌ Equal vector dimensions mean direct dot product is automatically ideal

No.

They may still live in poorly aligned representation spaces.

---

### ❌ `W_a` contains attention weights over the sequence

No.

It is a learned scoring matrix.

---

### ❌ Dot product ignores vector magnitude

No.

Magnitude directly affects the score.

---

### ❌ Luong eliminates the need for softmax

No.

The multiplicative interaction creates raw scores.

Softmax still converts them into attention weights.

---

### ❌ Luong eliminates recurrent sequential decoding

No.

---

### ❌ Local attention is guaranteed to be better than global attention

No.

It may be faster, but it can miss relevant source information.

---

# 📐 Formula Card

## Dot Score

```text
e_(t,i)
=
s_t^T h_i
```

## General Score

```text
e_(t,i)
=
s_t^T W_a h_i
```

## Attention Weight

```text
α_(t,i)
=
softmax_i(e_(t,i))
```

## Context

```text
c_t
=
Σ_i α_(t,i) h_i
```

## Attentional Hidden State

```text
h̃_t
=
tanh(
W_c [c_t ; s_t]
)
```

---

# 🧠 Bahdanau vs Luong — Interview Table

| Aspect                          | Bahdanau                  | Luong                  |
| ------------------------------- | ------------------------- | ---------------------- |
| Main score type                 | Additive                  | Multiplicative         |
| Core equation                   | `v^T tanh(W_s s + W_h h)` | `s^T h` or `s^T W_a h` |
| Score nonlinearity              | Yes                       | No for Dot/General     |
| Separate attention hidden space | Yes                       | Not necessarily        |
| Score parameters                | More                      | None / fewer           |
| Matrix friendliness             | Lower                     | Higher                 |
| State often used                | Previous decoder state    | Current decoder state  |
| Recurrent architecture          | Yes                       | Yes                    |

---

# 🎤 30-Second Interview Answer

> **Luong Attention is a classical encoder-decoder attention mechanism that scores compatibility using multiplicative interactions. Dot attention computes `s_t^T h_i`, while General attention computes `s_t^T W_a h_i`. These raw scores are softmax-normalized across source positions and used to create a weighted context vector. Compared with Bahdanau's nonlinear additive scorer, Luong's main variants are simpler and typically more matrix-multiplication friendly.**

---

# 🎤 Bahdanau vs Luong — Strong Short Answer

> **Bahdanau uses a learned nonlinear additive compatibility network, whereas Luong Dot and General attention use multiplicative interactions. Bahdanau is more structurally expressive in its scorer, while Luong is simpler and often more computationally efficient. Both still belong to classical recurrent encoder-decoder attention.**

---

# 🧠 Final Interview Mental Model

When asked about Luong Attention, reconstruct it like this:

```text
Why?
Bahdanau scoring works but is relatively complex

↓

Can compatibility be simpler?
Yes — use multiplication

↓

Simplest form?
s_t^T h_i

↓

What if dimensions or spaces differ?
Insert W_a

↓

Then?
Softmax across source positions

↓

Then?
Weighted context c_t

↓

Then?
Combine c_t with s_t

↓

Result?
Attentional hidden state for prediction
```

Shortest memory version:

# **Transform if needed → Multiply → Normalize → Aggregate → Combine**

---

# 🔗 Next Topic

We now understand two major classical attention scoring families:

```text
Bahdanau
→ Additive

Luong
→ Multiplicative
```

The next question is:

> **What do these learned source-target alignments actually look like, and what can we safely infer from them?**

That leads to:

# **05_Attention_Alignment_And_Visualization**
