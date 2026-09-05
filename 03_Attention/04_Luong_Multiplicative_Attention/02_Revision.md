# 📝 Luong / Multiplicative Attention — Revision Notes

> Goal: Reconstruct Luong Attention quickly from first principles, understand its main scoring variants, how it differs from Bahdanau Attention, and why multiplicative scoring became important.

---

# 🎯 One-Line Definition

> **Luong Attention computes compatibility between the decoder state and encoder states using multiplicative interactions such as a dot product or a learned bilinear transformation.**

It is commonly referred to as:

# **Multiplicative Attention**

---

# 🧠 Core Motivation

Bahdanau Attention computes:

```text
e_(t,i)
=
v_a^T tanh(
    W_s s_(t-1)
    +
    W_h h_i
    +
    b_a
)
```

This is flexible, but it requires:

* separate projections
* addition
* nonlinearity
* scalar readout

Luong asks:

> **Can we calculate compatibility more directly using multiplication?**

---

# ⭐ Core Mental Model

Luong scoring:

# **Transform if needed → Multiply**

Then generic attention:

# **Normalize → Aggregate**

Full flow:

```text
Decoder State
+
Encoder State
↓
Multiplicative Compatibility
↓
Scalar Scores
↓
Softmax
↓
Attention Weights
↓
Weighted Context
```

---

# 📐 Main Luong Scoring Variants

The two most important multiplicative forms are:

## 1. Dot Attention

```text
e_(t,i)
=
s_t^T h_i
```

## 2. General Attention

```text
e_(t,i)
=
s_t^T W_a h_i
```

---

# 🧠 Dot Attention

Dot attention directly compares:

```text
s_t
```

with:

```text
h_i
```

using:

```text
s_t^T h_i
```

Requirement:

```text
dim(s_t) = dim(h_i)
```

because a standard dot product requires equal vector dimensions.

---

# 🧮 Small Dot-Product Example

Suppose:

```text
s_t = [1, 2]

h_i = [3, 4]
```

Then:

```text
e_(t,i)
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

So:

```text
e_(t,i) = 11
```

The result is already a scalar.

---

# 🧠 What Does the Dot Product Represent?

Conceptually:

```text
decoder state
→ what the decoder currently needs

encoder state
→ what source position i contains
```

The dot product measures how compatible those representations are.

A higher value generally indicates stronger alignment.

---

# 🚨 Dot Product ≠ Cosine Similarity

Do not confuse:

```text
s_t^T h_i
```

with cosine similarity.

Dot product depends on:

* direction
* vector magnitude

Cosine similarity explicitly normalizes vector magnitudes.

So:

```text
dot product ≠ cosine similarity
```

unless normalization is applied.

---

# ⚠️ Plain Dot Attention Limitation

Suppose:

```text
s_t ∈ R^256

h_i ∈ R^512
```

Then:

```text
s_t^T h_i
```

is not defined.

That leads to:

# **General Luong Attention**

---

# 📐 General Scoring

General scoring introduces:

```text
W_a
```

and computes:

```text
e_(t,i)
=
s_t^T W_a h_i
```

The matrix transforms the encoder state before the multiplicative comparison.

---

# 🧠 Role of `W_a`

Suppose:

```text
s_t ∈ R^(H_d)

h_i ∈ R^(H_e)
```

Choose:

```text
W_a ∈ R^(H_d × H_e)
```

Then:

```text
W_a h_i ∈ R^(H_d)
```

Now:

```text
s_t^T W_a h_i
```

is valid.

So:

> `W_a` acts as a learned bridge between encoder and decoder representation spaces.

---

# 🧮 Shape Example

Suppose:

```text
decoder hidden size = 256
encoder hidden size = 512
```

Then:

```text
s_t ∈ R^256
h_i ∈ R^512
```

Choose:

```text
W_a ∈ R^(256 × 512)
```

Then:

```text
W_a h_i
→ R^256
```

and:

```text
s_t^T W_a h_i
→ scalar
```

So:

```text
e_(t,i) ∈ R
```

---

# ⭐ Dot vs General

## Dot

```text
e_(t,i) = s_t^T h_i
```

Advantages:

* simplest
* no extra scoring parameters
* highly efficient

Limitation:

* dimensions must match

## General

```text
e_(t,i) = s_t^T W_a h_i
```

Advantages:

* learned transformation
* handles different representation dimensions
* more flexible than plain dot

Cost:

* additional parameters

---

# 🧠 Why Is It Called Multiplicative Attention?

Because the core compatibility interaction is based on:

```text
multiplication
```

For example:

```text
s_t^T h_i
```

or:

```text
s_t^T W_a h_i
```

Unlike Bahdanau, which uses:

```text
project
+
add
+
tanh
```

Luong uses a more direct multiplicative interaction.

---

# 📐 Complete Luong Pipeline

### Score

Dot:

```text
e_(t,i) = s_t^T h_i
```

or General:

```text
e_(t,i) = s_t^T W_a h_i
```

### Normalize

```text
α_(t,i)
=
exp(e_(t,i))
/
Σ_j exp(e_(t,j))
```

### Context

```text
c_t
=
Σ_i α_(t,i) h_i
```

---

# 🧠 One Score Per Source Position

For:

```text
h_1, h_2, ..., h_T
```

the decoder computes:

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

Softmax gives:

```text
α_t ∈ R^T
```

and then:

```text
c_t
```

is formed as a weighted sum.

---

# ⚙️ Why Luong Is Efficient

Multiplicative attention maps naturally to matrix multiplication.

Suppose:

```text
H_enc ∈ R^(T × H)
```

and:

```text
s_t ∈ R^H
```

Then all dot scores can be computed as:

```text
e_t
=
H_enc s_t
```

giving:

```text
e_t ∈ R^T
```

So all source positions can be scored in one efficient linear algebra operation.

---

# ⭐ Why Matrix Efficiency Matters

Modern hardware is optimized for:

* matrix multiplication
* batching
* vectorized computation

Luong-style scoring therefore tends to be:

> simpler and more hardware-friendly than running a nonlinear scoring network separately for every source position.

---

# 🧠 Bahdanau vs Luong

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

Uses:

* separate projections
* addition
* nonlinearity
* learned scalar readout

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

Uses:

* direct multiplicative interaction

---

# 📊 Comparison Table

| Aspect                        | Bahdanau             | Luong                   |
| ----------------------------- | -------------------- | ----------------------- |
| Main interaction              | Additive             | Multiplicative          |
| Score nonlinearity            | Yes                  | No for dot/general      |
| Extra score parameters        | More                 | Few / none              |
| Shared attention hidden space | Explicit             | Not necessarily         |
| Matrix efficiency             | Lower                | Higher                  |
| Dimension mismatch            | Separate projections | General form handles it |
| Classical recurrent setting   | Yes                  | Yes                     |

---

# 🧠 Decoder-State Timing

A commonly taught distinction is:

### Bahdanau

often uses:

```text
s_(t-1)
```

for alignment.

### Luong

often uses:

```text
s_t
```

the current decoder state.

Conceptually:

```text
previous target information
↓
compute current decoder state s_t
↓
use s_t to attend to source
↓
build c_t
↓
predict
```

---

# ⚠️ Timing Is Not a Universal Law

Do not memorize:

> Bahdanau always uses previous state and Luong always uses current state.

Notation and implementation details vary.

The safe conceptual distinction is:

> Luong-style attention is commonly described using the current decoder state, while Bahdanau-style attention is commonly described using the previous decoder state during alignment.

---

# 🧠 Attentional Hidden State

Luong commonly combines:

```text
s_t
```

with:

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

This creates the:

# **attentional hidden state**

---

# 🧠 Why Combine `s_t` and `c_t`?

Because:

```text
s_t
```

contains:

> target-side generation information

while:

```text
c_t
```

contains:

> source information retrieved through attention

So:

```text
[c_t ; s_t]
```

combines:

```text
target history
+
retrieved source context
```

The combined representation is then used for output prediction.

---

# 🚨 Important Role Separation

Keep these distinct:

```text
s_t
→ decoder state

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

# 🌍 Global Luong Attention

Global attention considers:

```text
all encoder states
```

at each decoder timestep.

Pipeline:

```text
s_t
↓
compare with all h_i
↓
all scores
↓
softmax
↓
context
```

Conceptual scoring work across decoding:

```text
T_src × T_tgt
```

---

# 📍 Local Luong Attention

Local attention restricts attention to a smaller source region.

Conceptually:

```text
predict where to look
↓
choose local source window
↓
attend within that window
```

Example:

```text
[p_t - D, ..., p_t + D]
```

instead of:

```text
[1, ..., T]
```

---

# ⭐ Global vs Local

## Global

Pros:

* full source access
* simple concept

Cons:

* more expensive for long sources

## Local

Pros:

* smaller attention region
* potentially lower compute

Cons:

* may miss relevant source information
* introduces position/window-selection complexity

---

# 🧠 Luong Scoring Variants

Luong discussed:

```text
dot
general
concat
```

For our multiplicative-attention progression, focus mainly on:

```text
dot
general
```

because these are the core multiplicative forms.

---

# ⚠️ Concat Variant

A concat-style score may look like:

```text
v_a^T tanh(
W_a [s_t ; h_i]
)
```

This uses a neural nonlinear scorer.

So do not use it as the main explanation for why Luong is called:

> multiplicative attention.

The central multiplicative variants are:

* dot
* general

---

# 🔗 Relation to Transformers

Luong dot attention introduces an important idea:

```text
compatibility through dot products
```

This will later evolve into:

```text
scaled dot-product attention
```

in Transformers.

But Luong Attention is still:

> classical encoder-decoder attention in a recurrent architecture.

---

# 🚨 Luong Attention ≠ Self-Attention

Luong:

```text
decoder state
attends to
encoder states
```

Self-attention:

```text
positions within a sequence
attend to
other positions
```

Both may use dot products, but they are architecturally different.

---

# 🧠 Scale Matters

Dot-product score:

```text
s_t^T h_i
```

depends on vector magnitude.

Larger magnitudes can lead to larger raw scores.

This idea becomes very important later when we study:

> scaled dot-product attention.

---

# 📐 Parameter Count Intuition

## Dot

Scoring parameters:

```text
0
```

if hidden dimensions already match.

## General

Parameters:

```text
W_a ∈ R^(H_d × H_e)
```

So approximately:

```text
H_d × H_e
```

learned scoring parameters.

---

# 🚨 High-Yield Mistakes

### ❌ Dot attention works with any two vector dimensions

No.

They must match.

---

### ❌ `W_a` is the attention weight distribution

No.

It is a learned scoring parameter.

---

### ❌ Dot product is cosine similarity

No.

Magnitude affects dot product.

---

### ❌ Luong Attention removes recurrence

No.

The classical architecture is still recurrent.

---

### ❌ Luong Attention is Transformer attention

No.

It is classical encoder-decoder attention.

---

### ❌ General attention and Bahdanau are equivalent

No.

General Luong uses a bilinear multiplicative interaction.

Bahdanau uses a nonlinear additive network.

---

### ❌ Local attention is always better because it is cheaper

No.

It may miss useful source positions.

---

# 🎤 30-Second Interview Answer

> **Luong Attention is a classical encoder-decoder attention mechanism that uses multiplicative compatibility scoring. Dot attention computes `s_t^T h_i`, while general attention computes `s_t^T W_a h_i`. The resulting scores are normalized with softmax across source positions and used to form the context vector. Compared with Bahdanau's nonlinear additive scorer, Luong's multiplicative forms are simpler and more matrix-multiplication friendly.**

---

# 🎤 Dot vs General Interview Answer

> **Dot attention directly compares decoder and encoder states, so their dimensions must match. General attention introduces a learned matrix `W_a` that transforms the encoder state before the dot-product interaction, which allows more flexible compatibility modeling and can bridge different representation dimensions.**

---

# 🎤 Bahdanau vs Luong Interview Answer

> **Bahdanau uses a learned nonlinear additive scoring network, whereas Luong's main variants use multiplicative interactions. Bahdanau explicitly projects both states into an attention hidden space and applies `tanh`, while Luong dot/general scoring is simpler and generally more efficient for matrix computation.**

---

# 📐 Formula Card

### Dot

```text
e_(t,i)
=
s_t^T h_i
```

### General

```text
e_(t,i)
=
s_t^T W_a h_i
```

### Attention Weight

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

### Attentional Hidden State

```text
h̃_t
=
tanh(
W_c [c_t ; s_t]
)
```

---

# ⚡ 10 Things to Know Cold

1. Luong is commonly called Multiplicative Attention.
2. Dot score = `s_t^T h_i`.
3. General score = `s_t^T W_a h_i`.
4. Dot attention requires compatible dimensions.
5. General attention can transform encoder representations before comparison.
6. Scores are normalized across source positions.
7. Context is the weighted sum of encoder states.
8. Luong commonly uses the current decoder state.
9. Multiplicative scoring is efficient for matrix computation.
10. Luong is still recurrent encoder-decoder attention, not self-attention.

---

# 🧠 Final Mental Model

Bahdanau asks:

```text
Can a small neural network learn compatibility?
```

Luong asks:

```text
Can compatibility be computed directly
through multiplication?
```

Core Luong flow:

```text
Decoder State
        ×
Encoder Representation
        ↓
Scalar Compatibility Score
        ↓
Softmax
        ↓
Attention Weight
        ↓
Weighted Context
```

If dimensions do not match:

```text
Encoder State
↓ W_a
Transformed Encoder State
        ×
Decoder State
        ↓
Score
```

Shortest memory version:

# **Transform if needed → Multiply → Normalize → Aggregate**

---

# 🔗 Next Topic

Now that we understand:

```text
Bahdanau Additive Attention
+
Luong Multiplicative Attention
```

the next question is:

> **What do these attention weights actually look like across a full source-target sequence, and how should we interpret them?**

That leads to:

# **05_Attention_Alignment_And_Visualization**
