# 🎤 Bahdanau / Additive Attention — Interview Questions

> Goal: Be able to explain Bahdanau Attention from intuition, equations, tensor shapes, implementation details, and architectural trade-offs.

---

# 🟢 Beginner

## 1. What is Bahdanau Attention?

**Answer:**

Bahdanau Attention is an attention scoring mechanism that uses a small learned neural network to calculate compatibility between:

```text
decoder state
```

and:

```text
encoder state
```

The score is:

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

Then:

```text
scores
↓
softmax
↓
attention weights
↓
weighted sum of encoder states
↓
context vector
```

It is also called:

# **Additive Attention**

---

## 2. Why is Bahdanau Attention called Additive Attention?

**Answer:**

Because the projected encoder and decoder representations are combined using addition:

```text
W_s s_(t-1)
+
W_h h_i
+
b_a
```

before passing through:

```text
tanh
```

This is different from multiplicative attention, where the interaction is based on multiplication or dot products.

---

## 3. What problem does Bahdanau Attention solve?

**Answer:**

Generic attention tells us:

```text
e_(t,i) = score(s_(t-1), h_i)
```

but does not specify how to compute the score.

Bahdanau provides a learned nonlinear scoring function.

It also helps when encoder and decoder states:

* have different dimensions;
* encode information differently;
* need a richer compatibility function than a simple dot product.

---

## 4. What are the main learned parameters in Bahdanau Attention?

**Answer:**

Typically:

```text
W_s
W_h
b_a
v_a
```

where:

* `W_s` projects the decoder state;
* `W_h` projects the encoder state;
* `b_a` is a bias;
* `v_a` maps the nonlinear compatibility representation to a scalar score.

---

## 5. What does `v_a` do?

**Answer:**

After:

```text
tanh(
W_s s_(t-1)
+
W_h h_i
+
b_a
)
```

we still have a vector in the attention hidden space.

But attention needs one scalar score per source position.

So:

```text
v_a^T
```

acts as a learned vector-to-scalar projection:

```text
attention representation
↓
v_a^T
↓
scalar score e_(t,i)
```

---

# 🟡 Intermediate

## 6. Why are `W_s` and `W_h` needed?

**Answer:**

Encoder and decoder states may have different dimensions.

For example:

```text
s_(t-1) ∈ R^256
h_i     ∈ R^512
```

They cannot be directly added.

So Bahdanau maps both into a shared attention space:

```text
W_s ∈ R^(128 × 256)
W_h ∈ R^(128 × 512)
```

Then:

```text
W_s s_(t-1) ∈ R^128
W_h h_i     ∈ R^128
```

Now they can be combined.

More importantly:

> `W_s` and `W_h` are learned transformations, not merely resizing operations.

They learn how to represent encoder and decoder information in a space useful for compatibility scoring.

---

## 7. What is meant by a common attention space?

**Answer:**

A common attention space is a learned representation space where both encoder and decoder states are transformed into the same dimensionality and representation basis.

Conceptually:

```text
Decoder Representation
↓ W_s
Attention Space

Encoder Representation
↓ W_h
Attention Space
```

This allows the model to compare them meaningfully.

A useful mental model is:

> encoder and decoder speak different representation languages; `W_s` and `W_h` translate both into the same attention language.

---

## 8. Walk through the complete Bahdanau scoring process.

**Answer:**

For source position `i` and decoder step `t`:

### Step 1 — Project decoder state

```text
W_s s_(t-1)
```

### Step 2 — Project encoder state

```text
W_h h_i
```

### Step 3 — Combine them

```text
z_(t,i)
=
W_s s_(t-1)
+
W_h h_i
+
b_a
```

### Step 4 — Apply nonlinearity

```text
u_(t,i)
=
tanh(z_(t,i))
```

### Step 5 — Produce scalar score

```text
e_(t,i)
=
v_a^T u_(t,i)
```

Then generic attention continues:

```text
e_(t,i)
↓
softmax
↓
α_(t,i)
↓
weighted sum
↓
c_t
```

---

## 9. Why is `tanh` used?

**Answer:**

The main purpose is:

> introducing nonlinearity into the compatibility function.

Without the nonlinearity, stacked linear transformations would still reduce to an overall linear transformation.

`tanh` lets the attention scoring network model richer interactions between the decoder and encoder representations.

Historically, `tanh` was also common in recurrent architectures.

---

## 10. Do the encoder hidden size and decoder hidden size need to be equal?

**Answer:**

No.

That is one useful property of Bahdanau Attention.

If:

```text
H_d ≠ H_e
```

we can choose an attention dimension:

```text
H_a
```

and learn:

```text
W_s ∈ R^(H_a × H_d)

W_h ∈ R^(H_a × H_e)
```

Both projected vectors then become:

```text
R^(H_a)
```

---

# 🔴 Advanced

## 11. Why is Bahdanau Attention more expressive than a plain dot product?

**Answer:**

A plain dot product computes something like:

```text
s^T h_i
```

This is a relatively simple interaction.

Bahdanau computes:

```text
v_a^T tanh(
W_s s
+
W_h h_i
+
b_a
)
```

So it introduces:

* separate learned projections;
* an intermediate attention space;
* nonlinear interaction;
* a learned scalar readout.

Therefore it can model more complex compatibility patterns.

---

## 12. Are `W_s`, `W_h`, `v_a`, and `b_a` different for each timestep?

**Answer:**

No.

They are shared learned parameters.

The same scoring network is reused across:

* all decoder timesteps;
* all source positions;
* all examples.

What changes dynamically are:

```text
s_(t-1)
h_i
e_(t,i)
α_(t,i)
c_t
```

---

## 13. What can be precomputed in Bahdanau Attention?

**Answer:**

For a fixed encoded source:

```text
h_1, ..., h_T
```

the encoder states do not change during decoding.

Therefore:

```text
W_h h_i
```

can be precomputed for every source position.

At each decoder step, only:

```text
W_s s_(t-1)
```

changes.

So an efficient implementation can do:

```text
Encode source
↓
Precompute W_h h_i
↓
For each decoder step:
    compute W_s s_(t-1)
    combine with cached source projections
    score
```

This avoids redundant source-side matrix multiplications.

---

## 14. Why is softmax not applied immediately after `tanh`?

**Answer:**

The `tanh` output is:

```text
u_(t,i) ∈ R^(H_a)
```

which is an internal compatibility feature vector.

Attention needs:

> one scalar per source position.

So first:

```text
v_a^T u_(t,i)
```

produces:

```text
e_(t,i) ∈ R
```

Then softmax is applied across:

```text
e_(t,1), ..., e_(t,T)
```

That gives a distribution over source positions.

---

## 15. Does Bahdanau Attention use the projected encoder states to build the context vector?

**Answer:**

In the standard formulation, no.

The projections:

```text
W_h h_i
```

are mainly used for:

> compatibility scoring.

The context is normally:

```text
c_t
=
Σ_i α_(t,i) h_i
```

using the original encoder states.

So:

```text
projected states
→ scoring

original encoder states
→ weighted context
```

---

# ⭐ Staff Engineer Challenge

## Scenario

You are building a translation system with:

```text
BiLSTM Encoder
+
LSTM Decoder
```

The encoder produces:

```text
h_i ∈ R^512
```

and the decoder state is:

```text
s_t ∈ R^256
```

A developer proposes this scoring function:

```text
score = s_t^T h_i
```

### Questions

1. Will this work directly?
2. How would Bahdanau Attention solve the problem?
3. What trade-offs does it introduce?
4. What optimization would you consider in implementation?

---

## Strong Answer

### 1. Will direct dot product work?

No.

The dimensions differ:

```text
s_t ∈ R^256
h_i ∈ R^512
```

So:

```text
s_t^T h_i
```

is not defined directly.

---

### 2. How does Bahdanau solve it?

Choose a shared attention dimension, for example:

```text
H_a = 128
```

Then learn:

```text
W_s ∈ R^(128 × 256)

W_h ∈ R^(128 × 512)
```

This gives:

```text
W_s s_t ∈ R^128

W_h h_i ∈ R^128
```

Now compute:

```text
e_(t,i)
=
v_a^T tanh(
W_s s_t
+
W_h h_i
+
b_a
)
```

This gives one scalar score per source position.

---

### 3. Trade-offs

Advantages:

* handles mismatched dimensions;
* learns flexible compatibility;
* nonlinear scoring;
* useful alignment capacity.

Costs:

* extra learned parameters;
* extra matrix operations;
* nonlinear computation;
* potentially higher latency than simpler multiplicative methods.

---

### 4. Implementation optimization

Precompute:

```text
W_h h_i
```

for all encoder states once.

Reuse those projections across decoder timesteps.

Only recompute:

```text
W_s s_t
```

for each decoder step.

---

# ⚡ Rapid-Fire Questions

## Bahdanau Attention is also called?

Additive Attention.

---

## What is added?

```text
W_s s_(t-1)
+
W_h h_i
+
b_a
```

---

## What is the output of `tanh(...)`?

A vector in the attention hidden space:

```text
R^(H_a)
```

---

## What is the output of `v_a^T tanh(...)`?

A scalar compatibility score.

---

## Is `v_a` dynamic?

No.

It is a learned parameter.

---

## Are attention weights learned parameters?

No.

They are dynamically computed.

---

## Can encoder and decoder hidden dimensions differ?

Yes.

---

## What handles that mismatch?

`W_s` and `W_h`.

---

## Is `W_h h_i` usually the final context vector?

No.

---

## Where is softmax applied?

Across source-position scalar scores.

---

## Does Bahdanau remove recurrence?

No.

---

## Can source projections be cached?

Yes.

---

# 📐 Formula Card

## Bahdanau Score

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

## Shapes

```text
s_(t-1) ∈ R^(H_d)

h_i ∈ R^(H_e)

W_s ∈ R^(H_a × H_d)

W_h ∈ R^(H_a × H_e)

b_a ∈ R^(H_a)

v_a ∈ R^(H_a)

e_(t,i) ∈ R
```

---

# 🚨 High-Yield Interview Traps

### ❌ Bahdanau Attention means softmax itself

No.

Its defining feature is the additive nonlinear scoring function.

---

### ❌ `W_s` and `W_h` are just dimension converters

Incomplete.

They are learned attention-specific representation transformations.

---

### ❌ `v_a` is the attention weight vector over source tokens

No.

It is a learned readout parameter.

---

### ❌ `tanh` output is already the attention score

No.

It is still a vector.

`v_a^T` converts it to a scalar score.

---

### ❌ Softmax is applied across the `H_a` features

No.

Softmax is applied across source positions.

---

### ❌ Every source word has its own `W_h`

No.

The same `W_h` is shared.

---

### ❌ Every timestep has a new `W_s`

No.

The same `W_s` is reused.

---

### ❌ Bahdanau Attention automatically makes the system parallel like a Transformer

No.

The classic decoder is still recurrent and autoregressive.

---

# 🎤 30-Second Interview Answer

> **Bahdanau Attention, also called additive attention, computes compatibility between the decoder state and each encoder state using a learned nonlinear scoring network. The decoder and encoder states are separately projected into a shared attention space using `W_s` and `W_h`, added together, passed through `tanh`, and then projected to a scalar using `v_a`. These scores are softmax-normalized across source positions and used to build the dynamic context vector.**

---

# 🧠 Final Interview Mental Model

When asked about Bahdanau Attention, reconstruct it in this order:

```text
Why?
Generic attention needs a score function

↓

Problem?
Encoder and decoder representations may differ

↓

Solution?
Project both into a shared attention space

↓

How?
W_s s + W_h h_i

↓

Why tanh?
Nonlinear compatibility modeling

↓

Why v_a?
Convert compatibility vector to scalar

↓

Then?
Softmax across source positions

↓

Finally?
Weighted sum of encoder states
```

Shortest memory version:

# **Project → Add → Nonlinearity → Score**

followed by:

# **Normalize → Aggregate**

---

# 🔗 Next Topic

The natural next interview question is:

> **Can we compute compatibility more simply and efficiently than a small nonlinear neural network?**

That leads directly to:

# **Luong / Multiplicative Attention**
