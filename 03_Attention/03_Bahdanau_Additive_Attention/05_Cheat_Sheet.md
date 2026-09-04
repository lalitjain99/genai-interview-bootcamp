# ⚡ Bahdanau / Additive Attention — Cheat Sheet

> Goal: Reconstruct Bahdanau Attention in a few minutes before an interview.

---

# 1. 🎯 One-Line Definition

> **Bahdanau Attention computes compatibility between the decoder state and each encoder state using a learned nonlinear additive scoring network.**

Also called:

# **Additive Attention**

---

# 2. 🧠 Why Bahdanau Attention?

Generic attention gives:

```text
e_(t,i) = score(s_(t-1), h_i)
```

But leaves one question open:

> **How should `score(...)` be computed?**

Bahdanau answers with:

```text
Project
↓
Add
↓
Nonlinearity
↓
Scalar Score
```

---

# 3. 📐 Core Score Equation

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
α_(t,i) = softmax_i(e_(t,i))
```

and:

```text
c_t = Σ_i α_(t,i) h_i
```

---

# 4. ⭐ Core Mental Model

# **Project → Add → Nonlinearity → Score**

Then:

# **Normalize → Aggregate**

Full flow:

```text
Decoder State
↓ W_s

Encoder State
↓ W_h

        ↓

Common Attention Space

        ↓

Add + Bias

        ↓

tanh

        ↓

v_a^T

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

# 5. 🧠 Why `W_s` and `W_h`?

Because encoder and decoder states may have:

* different dimensions
* different semantic roles
* different learned representation spaces

Example:

```text
s_(t-1) ∈ R^256
h_i     ∈ R^512
```

Choose:

```text
H_a = 128
```

Then:

```text
W_s ∈ R^(128 × 256)
W_h ∈ R^(128 × 512)
```

So:

```text
W_s s_(t-1) ∈ R^128
W_h h_i     ∈ R^128
```

Now both live in the same attention space.

---

# 6. ⭐ Common Attention Space

Do not think:

```text
W_s and W_h
→ just resize vectors
```

Better:

```text
W_s
→ learned decoder representation for attention

W_h
→ learned encoder representation for attention
```

Mental model:

```text
Decoder speaks one representation language
Encoder speaks another representation language

        ↓

W_s and W_h

        ↓

Both translated into
the same attention language
```

---

# 7. 📐 Shape Card

Suppose:

```text
decoder size   = H_d
encoder size   = H_e
attention size = H_a
```

Then:

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

# 8. 🧠 Why Is It Called Additive?

Because the projected representations are combined using:

```text
W_s s_(t-1)
+
W_h h_i
+
b_a
```

So the defining interaction is:

> **addition after projection**

---

# 9. 🧠 Why `tanh`?

Main reason:

> introduce nonlinearity into the compatibility function.

Without nonlinearity, stacked linear transformations would remain equivalent to a linear mapping.

So:

```text
projected decoder
+
projected encoder
↓
tanh
↓
nonlinear compatibility representation
```

---

# 10. 🧠 Why `v_a`?

After `tanh`, we still have:

```text
R^(H_a)
```

But attention needs:

> one scalar score per source position.

So:

```text
v_a^T
```

does:

```text
R^(H_a)
↓
R
```

Hence:

```text
e_(t,i) ∈ R
```

---

# 11. 🚨 `v_a` vs `α_(t,i)`

Do not confuse these.

### `v_a`

```text
learned model parameter
```

### `α_(t,i)`

```text
dynamic attention weight
```

So:

```text
v_a ≠ attention distribution
```

---

# 12. 📐 Complete Bahdanau Pipeline

### Score

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

# 13. 🧠 What Gets Projected vs What Gets Aggregated?

For scoring:

```text
W_h h_i
```

is used.

For context:

```text
h_i
```

is usually used.

So:

```text
projected encoder state
→ scoring

original encoder state
→ weighted context
```

---

# 14. ⭐ Learned Parameters vs Dynamic Values

## Learned

```text
W_s
W_h
b_a
v_a
```

## Dynamic

```text
e_(t,i)
α_(t,i)
c_t
```

Parameters persist.

Scores and weights change with the input and decoder timestep.

---

# 15. 🔄 Parameter Sharing

The same:

```text
W_s
W_h
b_a
v_a
```

are reused across:

* all source positions
* all decoder timesteps
* all training examples

Sequence length affects compute, not parameter count.

---

# 16. ⚙️ What Can Be Precomputed?

For a fixed source:

```text
h_1, ..., h_T
```

we can precompute:

```text
W_h h_1
W_h h_2
...
W_h h_T
```

once.

At each decoder step, recompute only:

```text
W_s s_(t-1)
```

because the decoder state changes.

---

# 17. 📐 Parameter Count

Approximate scoring-network parameter count:

```text
H_a × H_d
+
H_a × H_e
+
H_a
+
H_a
```

or:

```text
H_a(H_d + H_e + 2)
```

Increasing `H_a` means:

* more capacity
* more parameters
* more compute

---

# 18. 🔗 Bahdanau vs Generic Attention

Generic:

```text
e_(t,i) = score(s, h_i)
```

Bahdanau:

```text
score(s, h_i)
=
v_a^T tanh(
W_s s
+
W_h h_i
+
b_a
)
```

Everything afterward remains the generic mechanism:

```text
score
↓
softmax
↓
weights
↓
weighted sum
```

---

# 19. 🔗 Bahdanau vs Dot Product

Plain dot product:

```text
s^T h_i
```

requires compatible dimensions and gives a simpler interaction.

Bahdanau:

```text
v_a^T tanh(
W_s s
+
W_h h_i
+
b_a
)
```

provides:

* separate learned projections
* shared attention space
* nonlinear compatibility
* more flexibility

Trade-off:

> more computation and parameters.

---

# 20. 🔗 Bahdanau vs Luong

### Bahdanau

```text
additive
+
nonlinear scorer
```

### Luong

```text
multiplicative / dot-product-style scorer
```

Mental progression:

```text
Bahdanau
→ more expressive scoring

Luong
→ simpler multiplicative scoring
```

---

# ⚡ 10 Things to Know Cold

1. Bahdanau = Additive Attention.
2. It implements the generic `score(...)` function.
3. `W_s` projects the decoder state.
4. `W_h` projects the encoder state.
5. Both projections enter a shared attention space.
6. The projected states are added.
7. `tanh` provides nonlinearity.
8. `v_a^T` converts the compatibility vector to a scalar.
9. Softmax is applied across source-position scores.
10. The original encoder states are typically used to form the context vector.

---

# 🚫 High-Yield Traps

### ❌ `W_s` and `W_h` only resize vectors

No.

They are learned representation transformations.

---

### ❌ Encoder and decoder dimensions must match

No.

Projection matrices can map both to `H_a`.

---

### ❌ `v_a` is an attention weight

No.

It is a learned parameter.

---

### ❌ `tanh(...)` directly gives the scalar score

No.

It gives an `H_a`-dimensional vector.

---

### ❌ Softmax is applied across `H_a`

No.

Softmax is across source positions.

---

### ❌ `W_h h_i` must be recomputed every decoder step

No.

It can be cached for a fixed source.

---

### ❌ Bahdanau removes recurrence

No.

Classic Bahdanau Attention is used with recurrent encoder-decoder models.

---

### ❌ Attention weights are learned directly

No.

They are dynamically computed from learned parameters and current states.

---

# 🎤 30-Second Interview Answer

> **Bahdanau Attention, also called additive attention, uses a learned nonlinear scoring network to measure compatibility between the decoder state and each encoder state. Both are projected into a shared attention space using `W_s` and `W_h`, added together, passed through `tanh`, and then projected to a scalar using `v_a`. The resulting source-position scores are softmax-normalized and used to build the dynamic context vector.**

---

# 🧠 Final Mental Model

```text
Decoder State
↓ W_s

Encoder State
↓ W_h

        ↓

Shared Attention Space

        ↓

Add

        ↓

tanh

        ↓

v_a^T

        ↓

Score

        ↓

Softmax

        ↓

Weight

        ↓

Context
```

Shortest version:

# **Project → Add → Nonlinearity → Score**

then:

# **Normalize → Aggregate**

---

# 🔗 Next Topic

The next question is:

> **Can compatibility be computed with a simpler and more multiplication-friendly scoring function?**

That leads to:

# **Luong / Multiplicative Attention**
