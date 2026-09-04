# 💡 Bahdanau / Additive Attention — Assignment

> Goal: Practice the actual reasoning behind Bahdanau Attention: projection into a common attention space, nonlinear scoring, scalar readout, softmax normalization, tensor shapes, implementation optimization, and architectural trade-offs.

---

# 🧮 1. Shape Reasoning — Can These States Be Added Directly?

Suppose:

```text
decoder state:
s_(t-1) ∈ R^256
```

and:

```text
encoder state:
h_i ∈ R^512
```

Can we compute:

```text
s_(t-1) + h_i
```

directly?

### Answer

No.

The dimensions are different:

```text
R^256
+
R^512
```

is not defined.

Bahdanau Attention solves this by projecting both into a shared attention space.

For example:

```text
H_a = 128
```

then:

```text
W_s ∈ R^(128 × 256)

W_h ∈ R^(128 × 512)
```

so:

```text
W_s s_(t-1) ∈ R^128

W_h h_i ∈ R^128
```

Now they can be added.

---

# 🧠 2. What Is the Common Attention Space?

Explain in your own words what this means:

```text
W_s s_(t-1)
```

and:

```text
W_h h_i
```

both belong to:

```text
R^(H_a)
```

### Answer

It means both the decoder and encoder representations have been transformed into the same learned representation space.

This space is designed specifically for:

> compatibility scoring.

The projections do more than make dimensions equal.

They learn how to re-express:

```text
decoder information
```

and:

```text
encoder information
```

in a form useful for deciding relevance.

---

# 🧮 3. Small Projection Example

Suppose:

```text
s_(t-1) =
[1
 2]
```

and:

```text
W_s =
[2 0
 0 3]
```

Compute:

```text
W_s s_(t-1)
```

### Answer

```text
[2 0      [1
 0 3]  ×   2]
```

gives:

```text
[2
 6]
```

Therefore:

```text
W_s s_(t-1) = [2, 6]
```

---

# 🧮 4. Encoder Projection

Suppose:

```text
h_i =
[1
 2
 3]
```

and:

```text
W_h =
[1 0 1
 0 2 0]
```

Compute:

```text
W_h h_i
```

### Answer

First component:

```text
1×1 + 0×2 + 1×3 = 4
```

Second component:

```text
0×1 + 2×2 + 0×3 = 4
```

Therefore:

```text
W_h h_i = [4, 4]
```

---

# 🧮 5. Combine Both Projections

Using:

```text
W_s s_(t-1) = [2, 6]
```

and:

```text
W_h h_i = [4, 4]
```

with:

```text
b_a = [1, -1]
```

compute:

```text
z_(t,i)
=
W_s s_(t-1)
+
W_h h_i
+
b_a
```

### Answer

```text
[2, 6]
+
[4, 4]
+
[1, -1]
```

gives:

```text
[7, 9]
```

So:

```text
z_(t,i) = [7, 9]
```

---

# 🧮 6. Apply the Nonlinearity

Suppose:

```text
z_(t,i) = [0.5, -1.0]
```

and approximately:

```text
tanh(0.5) ≈ 0.46

tanh(-1.0) ≈ -0.76
```

What is:

```text
u_(t,i)
=
tanh(z_(t,i))
```

### Answer

```text
u_(t,i)
≈
[0.46, -0.76]
```

---

# 🧮 7. Convert the Compatibility Vector into a Scalar Score

Suppose:

```text
u_(t,i)
=
[0.46, -0.76]
```

and:

```text
v_a =
[2, 1]
```

Compute:

```text
e_(t,i)
=
v_a^T u_(t,i)
```

### Answer

```text
e_(t,i)
=
2(0.46)
+
1(-0.76)
```

```text
=
0.92 - 0.76
```

```text
=
0.16
```

Therefore:

```text
e_(t,i) = 0.16
```

This is one scalar compatibility score for source position `i`.

---

# 🧠 8. Why Is `v_a` Needed?

A developer asks:

> "Why can't we just use the `tanh(...)` output as the score?"

### Answer

Because:

```text
tanh(...)
```

produces a vector in the attention hidden space:

```text
R^(H_a)
```

But attention needs:

> one scalar score per source position.

So:

```text
v_a^T
```

reduces the compatibility vector to a scalar:

```text
R^(H_a)
↓
R
```

---

# 🔍 9. Find the Bug — Softmax on the Wrong Tensor

Suppose:

```text
u_(t,i) ∈ R^128
```

for each source position.

A developer directly applies softmax to those 128 values.

What is wrong?

### Answer

Those 128 values are:

> internal compatibility features.

They are not source-position scores.

First:

```text
v_a^T u_(t,i)
```

must produce one scalar:

```text
e_(t,i)
```

for every source position.

Only then should softmax be applied across:

```text
e_(t,1), ..., e_(t,T)
```

---

# 🧠 10. Learned Parameters or Dynamic Values?

Classify each quantity as:

* learned model parameter
* dynamically computed value

### A

```text
W_s
```

### B

```text
W_h
```

### C

```text
v_a
```

### D

```text
e_(t,i)
```

### E

```text
α_(t,i)
```

### F

```text
c_t
```

### Answer

Learned parameters:

```text
W_s
W_h
v_a
```

and usually:

```text
b_a
```

Dynamic values:

```text
e_(t,i)
α_(t,i)
c_t
```

---

# 🧠 11. Why Is Bahdanau Attention Called Additive?

Consider:

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

Why is the term:

> Additive Attention

used?

### Answer

Because the projected decoder and encoder representations are combined through addition:

```text
W_s s_(t-1)
+
W_h h_i
```

before the nonlinear transformation.

---

# 🔍 12. Is This Explanation Correct?

A candidate says:

> "`W_s` and `W_h` only exist because matrix dimensions need to match."

Is this complete?

### Answer

No.

Dimension matching is one role.

But the deeper purpose is:

> learned representation transformation.

`W_s` and `W_h` learn how to express decoder and encoder states in a shared attention space that is useful for compatibility scoring.

So:

```text
dimension compatibility
+
representation learning
```

are both important.

---

# 🧠 13. Encoder State vs Projected Encoder State

Bahdanau computes:

```text
W_h h_i
```

for scoring.

Should the standard context vector be:

```text
c_t = Σ_i α_(t,i) W_h h_i
```

or:

```text
c_t = Σ_i α_(t,i) h_i
```

### Answer

In the standard formulation:

```text
c_t = Σ_i α_(t,i) h_i
```

The projected state:

```text
W_h h_i
```

is used for compatibility calculation.

The original encoder state:

```text
h_i
```

is used as the value being aggregated into the context.

---

# 🧠 14. Parameter Sharing

Suppose the source has 20 tokens and the target has 15 tokens.

How many different `W_h` matrices are learned?

### Answer

One.

The same:

```text
W_h
```

is shared across all source positions and decoder timesteps.

Similarly:

```text
W_s
v_a
b_a
```

are also shared.

Sequence length changes compute, not parameter count.

---

# 🔍 15. What Can Be Precomputed?

For a fixed source sequence:

```text
h_1, h_2, ..., h_T
```

which part of Bahdanau scoring can be precomputed before autoregressive decoding starts?

### Answer

The encoder-side projections:

```text
W_h h_1
W_h h_2
...
W_h h_T
```

can be computed once.

During decoding, the decoder-side term:

```text
W_s s_(t-1)
```

must be recomputed because decoder state changes at each target step.

---

# 🧠 16. Attention-Space Dimension Trade-Off

Suppose you increase:

```text
H_a = 128
```

to:

```text
H_a = 1024
```

What changes?

### Answer

Potential benefits:

* larger scoring-network capacity
* potentially richer compatibility features

Costs:

* more parameters
* more memory
* more computation

So:

> larger attention dimension does not automatically mean better attention.

---

# 📐 17. Parameter Count

Suppose:

```text
H_d = 256
H_e = 512
H_a = 128
```

Ignoring any additional implementation-specific parameters, estimate the number of Bahdanau scoring parameters.

Use:

```text
W_s ∈ R^(H_a × H_d)
W_h ∈ R^(H_a × H_e)
b_a ∈ R^(H_a)
v_a ∈ R^(H_a)
```

### Answer

`W_s`:

```text
128 × 256 = 32,768
```

`W_h`:

```text
128 × 512 = 65,536
```

`b_a`:

```text
128
```

`v_a`:

```text
128
```

Total:

```text
32,768
+
65,536
+
128
+
128
=
98,560
```

So approximately:

```text
98.6K parameters
```

---

# 🧠 18. Why Not Use a Direct Dot Product?

Suppose:

```text
s_t ∈ R^256
h_i ∈ R^512
```

Can we compute:

```text
s_t^T h_i
```

directly?

### Answer

No.

The dimensions do not match.

Bahdanau solves this through separate learned projections into:

```text
R^(H_a)
```

before combining the representations.

Even when dimensions match, Bahdanau additionally provides:

> nonlinear learned compatibility.

---

# ✅ True / False

## 1.

Bahdanau Attention is also called Additive Attention.

**Answer:** True.

---

## 2.

`W_s` and `W_h` must have the same original input dimension.

**Answer:** False.

Their output dimensions need to match the attention space.

---

## 3.

`v_a` produces one scalar score from the attention hidden representation.

**Answer:** True.

---

## 4.

`α_(t,i)` is a permanent learned parameter.

**Answer:** False.

It is dynamically computed.

---

## 5.

The same attention parameters are reused across source positions.

**Answer:** True.

---

## 6.

Softmax should normally operate over the `H_a` features.

**Answer:** False.

It operates over source-position scalar scores.

---

## 7.

Bahdanau Attention can handle encoder and decoder states with different dimensions.

**Answer:** True.

---

## 8.

`W_h h_i` usually changes at every decoder timestep.

**Answer:** False.

For a fixed source, it can be reused.

---

## 9.

Bahdanau Attention removes recurrence from the decoder.

**Answer:** False.

---

## 10.

The final context vector is typically built from weighted original encoder states.

**Answer:** True.

---

# ⭐ Staff Engineer Challenge

You are reviewing an implementation of Bahdanau Attention in a production Seq2Seq model.

The architecture is:

```text
BiLSTM Encoder
↓
Bahdanau Attention
↓
LSTM Decoder
```

Shapes:

```text
encoder state = 1024
decoder state = 512
attention dimension = 256
```

The implementation currently does this at every decoder timestep:

```text
for every source token:
    compute W_h h_i

compute W_s s_t

compute tanh(...)

compute score
```

The model is accurate, but decoding latency is high.

### Questions

1. What obvious redundant computation do you see?
2. What can be cached?
3. Does caching change the model mathematically?
4. What other bottleneck remains even after caching?
5. Would replacing Bahdanau with a simpler scoring method necessarily preserve quality?

---

## Strong Answer

### 1. Redundant computation

The implementation repeatedly recomputes:

```text
W_h h_i
```

for the same encoder states at every decoder step.

That is unnecessary.

---

### 2. What can be cached?

After encoding:

```text
h_1, ..., h_T
```

compute:

```text
W_h h_1, ..., W_h h_T
```

once.

Reuse them for every target timestep.

---

### 3. Does caching change the mathematics?

No.

It is only an implementation optimization.

The same values are being reused instead of recomputed.

---

### 4. What bottleneck remains?

The decoder is still autoregressive.

At every target step it must:

```text
compute decoder state
↓
score source positions
↓
softmax
↓
build context
↓
predict next token
```

So target generation remains sequential.

Also, attention still interacts with all source positions per target step.

---

### 5. Would simpler scoring preserve quality?

Not guaranteed.

A multiplicative score may be:

* cheaper
* easier to optimize
* more matrix-multiplication friendly

but Bahdanau's nonlinear scorer may model compatibility differently.

The correct decision should be based on:

* quality
* latency
* throughput
* memory
* source-length behavior
* deployment constraints

rather than assuming one scoring method is universally superior.

---

# 🧠 Final Exercise — Reconstruct the Formula From Roles

Fill the blanks:

```text
Decoder State
↓
__________

Encoder State
↓
__________

Both enter:
__________

Then:
__________

Then:
__________

Result:
scalar compatibility score
```

### Answer

```text
Decoder State
↓
W_s projection

Encoder State
↓
W_h projection

Both enter:
shared attention space

Then:
addition + bias

Then:
tanh + v_a^T

Result:
scalar compatibility score
```

Mathematically:

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

---

# 🎯 Assignment Mental Model

You should be able to explain each transformation independently:

```text
s_(t-1)
↓ W_s
decoder representation for attention

h_i
↓ W_h
source representation for attention

both
↓
shared attention space

add + tanh
↓
nonlinear compatibility representation

v_a^T
↓
scalar compatibility score

softmax
↓
attention weight

weighted h_i
↓
context
```

The shortest memory chain is:

# **Project → Add → Nonlinearity → Score → Normalize → Aggregate**

---

# 🔗 Next Topic

After Bahdanau, the natural engineering question is:

> **Can we calculate compatibility with fewer operations and a simpler interaction?**

That leads directly to:

# **Luong / Multiplicative Attention**
