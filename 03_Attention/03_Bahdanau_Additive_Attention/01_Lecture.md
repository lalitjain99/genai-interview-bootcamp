# 📘 Bahdanau / Additive Attention

## 🌟 Introduction

In the previous lecture, we built the **generic Attention Mechanism**.

We reached this pipeline:

```text
Decoder State
+
Encoder States
↓
Compatibility Scores
↓
Softmax
↓
Attention Weights
↓
Weighted Sum
↓
Dynamic Context
```

Mathematically:

```text
e_(t,i) = score(s_(t-1), h_i)

α_(t,i) = softmax(e_(t,i))

c_t = Σ_i α_(t,i) h_i
```

But one question was intentionally left open:

> **How should `score(s_(t-1), h_i)` actually be computed?**

That is exactly where **Bahdanau Attention** enters.

Bahdanau attention gives us a learned, nonlinear scoring function.

Instead of asking:

> “Can we just take a dot product?”

it asks:

> “Can a small neural network learn what compatibility means?”

That is the central idea of this lecture.

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to:

* explain why a learned scoring function is needed;
* derive the Bahdanau score from first principles;
* understand why it is called **additive attention**;
* understand the role of:

  * `W_s`
  * `W_h`
  * `b_a`
  * `v_a`
* reason about tensor shapes;
* explain how scores become attention weights;
* connect Bahdanau attention to encoder-decoder alignment;
* understand how it differs conceptually from the generic mechanism;
* understand why it is more flexible than a simple dot product;
* explain the computational trade-offs;
* prepare for Luong / Multiplicative Attention.

---

# 📖 Part 1 — Where We Left Off

Generic attention gives us:

```text
e_(t,i) = score(s_(t-1), h_i)
```

But `score(...)` is still abstract.

Suppose the decoder state is:

```text
s_(t-1)
```

and one encoder state is:

```text
h_i
```

We need a scalar:

```text
e_(t,i)
```

that tells us:

> how compatible these two representations are.

---

# 🧠 Part 2 — Why Not Use Raw Vector Comparison?

A naive idea could be:

```text
score = s_(t-1) · h_i
```

But this immediately creates constraints.

What if:

```text
s_(t-1) ∈ R^256
```

and:

```text
h_i ∈ R^512
```

Then direct dot product is not even defined.

And even if dimensions match, a plain dot product gives us only:

> a simple linear similarity.

Bahdanau's idea is:

> first transform both vectors into a shared learned space, combine them, apply nonlinearity, and then convert the result into a scalar score.

---

# ⭐ Part 3 — The Core Idea

Instead of:

```text
decoder state
×
encoder state
```

Bahdanau does:

```text
decoder state
↓ learned projection

encoder state
↓ learned projection

combine them
↓
nonlinearity
↓
learned scalar projection
↓
compatibility score
```

This creates a small neural network for alignment.

---

# 📐 Part 4 — Bahdanau Score Equation

A common formulation is:

```text
e_(t,i) =
v_a^T tanh(
    W_s s_(t-1)
    +
    W_h h_i
    +
    b_a
)
```

where:

* `s_(t-1)` = decoder state
* `h_i` = encoder hidden state
* `W_s` = decoder-state projection matrix
* `W_h` = encoder-state projection matrix
* `b_a` = attention bias
* `tanh` = nonlinearity
* `v_a` = learned vector that converts the intermediate representation to a scalar
* `e_(t,i)` = compatibility score

This is the core Bahdanau scoring function.

---

# 🧠 Part 5 — Why Is It Called Additive Attention?

Look at the middle of the equation:

```text
W_s s_(t-1)
+
W_h h_i
+
b_a
```

The projected decoder and encoder representations are:

> **added together**

before applying `tanh`.

That is why this family is called:

# **Additive Attention**

It is also commonly associated with:

> **Bahdanau Attention**

---

# 📖 Part 6 — Build the Equation Slowly

Let's derive it step by step.

We start with:

```text
s_(t-1)
```

and:

```text
h_i
```

These may live in different dimensional spaces.

So first:

```text
W_s s_(t-1)
```

projects the decoder state.

And:

```text
W_h h_i
```

projects the encoder state.

Now both can live in a common:

> attention space.

---

# 📐 Part 7 — Attention Space

Suppose:

```text
decoder hidden size = H_d
encoder hidden size = H_e
attention size = H_a
```

Then:

```text
s_(t-1) ∈ R^(H_d)
h_i     ∈ R^(H_e)
```

Choose:

```text
W_s ∈ R^(H_a × H_d)
W_h ∈ R^(H_a × H_e)
```

Then:

```text
W_s s_(t-1) ∈ R^(H_a)
```

and:

```text
W_h h_i ∈ R^(H_a)
```

Now they can be added.

---

# ⭐ Part 8 — Why This Shared Attention Space Matters

This means encoder and decoder states do not need to have the same original dimension.

Bahdanau attention learns:

> how to map both representations into a common comparison space.

That is an important practical advantage.

---

# 📐 Part 9 — Combine the Representations

Now compute:

```text
z_(t,i)
=
W_s s_(t-1)
+
W_h h_i
+
b_a
```

where:

```text
z_(t,i) ∈ R^(H_a)
```

At this stage we have:

> a joint representation of decoder need + source information.

---

# 🧠 Part 10 — Why Add Them?

Conceptually:

```text
decoder state
→ what am I looking for?

encoder state
→ what does this source position contain?
```

After projection:

```text
projected decoder need
+
projected source representation
```

creates a joint compatibility representation.

The model learns the projections.

---

# 📐 Part 11 — Add Nonlinearity

Next:

```text
u_(t,i) = tanh(z_(t,i))
```

or:

```text
u_(t,i)
=
tanh(
W_s s_(t-1)
+
W_h h_i
+
b_a
)
```

This gives the scoring network nonlinear capacity.

Without `tanh`, multiple linear transformations would still collapse into a linear mapping.

---

# 🧠 Part 12 — Why Tanh?

Historically, tanh was a natural choice because:

* it introduces nonlinearity;
* outputs are bounded;
* it was standard in recurrent architectures of that era.

The most important conceptual point is:

> the scoring function is nonlinear.

Do not over-focus on tanh specifically as if it were the only possible activation.

---

# 📐 Part 13 — We Still Need a Scalar

After tanh:

```text
u_(t,i) ∈ R^(H_a)
```

But attention needs:

```text
e_(t,i) ∈ R
```

one scalar per source position.

So we introduce:

```text
v_a ∈ R^(H_a)
```

Then:

```text
e_(t,i) = v_a^T u_(t,i)
```

So the full equation becomes:

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

# 🧠 Part 14 — What Does `v_a` Do?

`v_a` acts like:

> a learned readout direction.

The nonlinear hidden representation says:

> what kind of interaction exists between decoder state and encoder state?

Then `v_a` converts that interaction into:

> one scalar compatibility score.

---

# ⭐ Part 15 — Bahdanau Scoring Network Is a Tiny Neural Network

You can think of it as:

```text
[s_(t-1), h_i]
↓
Learned transformations
↓
Nonlinear hidden representation
↓
Scalar output
```

So Bahdanau attention is essentially:

> a small feed-forward neural network used to score alignment.

---

# 📐 Part 16 — Full Attention Pipeline

Bahdanau defines the score.

Everything after that is the same generic mechanism.

### Step 1 — Score

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

### Step 2 — Normalize

```text
α_(t,i)
=
exp(e_(t,i))
/
Σ_j exp(e_(t,j))
```

### Step 3 — Context

```text
c_t
=
Σ_i α_(t,i) h_i
```

---

# ⭐ Part 17 — Important Distinction

Bahdanau Attention is not a completely different attention pipeline.

It is:

> **a specific implementation of the generic score function.**

Generic:

```text
score(...)
```

Bahdanau:

```text
v_a^T tanh(...)
```

That distinction is important.

---

# 🧮 Part 18 — Worked Shape Example

Suppose:

```text
H_d = 256
H_e = 512
H_a = 128
```

Then:

```text
s_(t-1) ∈ R^256
h_i     ∈ R^512
```

Parameters:

```text
W_s ∈ R^(128 × 256)
W_h ∈ R^(128 × 512)
b_a ∈ R^128
v_a ∈ R^128
```

Now:

```text
W_s s_(t-1)
→ R^128
```

```text
W_h h_i
→ R^128
```

Add:

```text
R^128 + R^128
→ R^128
```

After tanh:

```text
u_(t,i) ∈ R^128
```

Then:

```text
v_a^T u_(t,i)
→ scalar
```

So:

```text
e_(t,i) ∈ R
```

Perfect.

---

# 🧠 Part 19 — One Score Per Source Position

For a source sequence:

```text
h_1, h_2, ..., h_T
```

the decoder state is compared against every source state:

```text
e_(t,1)
e_(t,2)
...
e_(t,T)
```

So at decoder step `t`:

```text
e_t ∈ R^T
```

Then softmax gives:

```text
α_t ∈ R^T
```

---

# 📖 Part 20 — Example Story

Suppose the source sentence is:

```text
The contract expires on 15 September
```

Encoder states:

```text
h1 → The
h2 → contract
h3 → expires
h4 → on
h5 → 15
h6 → September
```

Now the decoder is trying to generate the translated date.

Its state:

```text
s_(t-1)
```

encodes what it has generated so far.

Bahdanau attention compares this decoder state against:

```text
h1, h2, ..., h6
```

using the learned scoring network.

It may produce:

```text
e = [0.2, 0.5, 1.1, 0.4, 3.7, 3.2]
```

After softmax:

```text
α ≈ [small, small, small, small, high, high]
```

So the dynamic context emphasizes:

```text
15
September
```

---

# 🧠 Part 21 — Why This Is More Flexible Than Plain Dot Product

A plain dot product assumes compatibility is well captured by:

```text
s^T h
```

Bahdanau instead learns:

```text
nonlinear(
project(s)
+
project(h)
)
```

This allows the model to learn a richer compatibility function.

It can compare representations even when their original dimensions differ.

---

# ⚠️ Part 22 — More Flexible Does Not Mean Always Better

Bahdanau scoring is more expressive than a raw dot product, but:

* it has additional parameters;
* it requires more computation;
* it is less matrix-multiplication-friendly than simpler multiplicative forms.

This trade-off becomes important later.

---

# 📐 Part 23 — Parameter Count Intuition

Ignoring exact architecture variants, attention parameters include:

```text
W_s
W_h
b_a
v_a
```

If dimensions are:

```text
H_d, H_e, H_a
```

rough parameter count is:

```text
H_a × H_d
+
H_a × H_e
+
H_a
+
H_a
```

or roughly:

```text
H_a(H_d + H_e + 2)
```

This is additional learned capacity.

---

# 🧠 Part 24 — Parameter Sharing

These attention parameters are:

> shared across decoder timesteps and source positions.

We do not learn a separate:

```text
W_h
```

for every source token.

The same scoring function is reused everywhere.

---

# ⭐ Part 25 — Why Parameter Sharing Matters

This makes attention:

* length-independent in parameter count;
* reusable across positions;
* able to generalize to different source lengths.

Like recurrent networks, sequence length affects:

> compute,

not:

> number of learned attention parameters.

---

# 📐 Part 26 — Matrix View

Instead of computing:

```text
W_h h_i
```

separately for every source position, implementations can precompute:

```text
W_h h_1
W_h h_2
...
W_h h_T
```

once for the encoded source.

Why?

Because during decoding:

```text
h_i
```

does not change.

This can reduce repeated work.

---

# 🧠 Part 27 — What Changes Per Decoder Step?

For one encoded source:

### Reusable

```text
W_h h_i
```

for every source state.

### Changes each decoder timestep

```text
W_s s_(t-1)
```

because the decoder state changes.

Then the model combines:

```text
projected decoder state
+
precomputed encoder projections
```

to calculate current scores.

---

# ⭐ Part 28 — Practical Optimization Insight

A naive implementation might repeatedly recompute:

```text
W_h h_i
```

for all source states at every target timestep.

That is unnecessary.

Because encoder states are fixed during decoding, those projections can be cached.

This is a useful implementation-level interview point.

---

# 📖 Part 29 — Why Previous Decoder State?

A classic Bahdanau formulation commonly uses:

```text
s_(t-1)
```

when computing alignment for target timestep `t`.

Conceptually:

> before generating the current output, use what the decoder knows so far to decide where to look.

Then:

```text
c_t
```

helps construct/update the decoder information used for current prediction.

---

# ⚠️ Part 30 — Do Not Overgeneralize the Ordering

Different attention implementations and notation conventions may use:

* previous decoder state;
* current intermediate decoder state;
* slightly different ordering.

So the important concept is:

> a decoder representation expressing the current target-side need is compared with source representations.

Do not turn one notation convention into a universal law.

---

# 🧠 Part 31 — Historical Insight

Bahdanau attention was important because it showed that encoder-decoder models did not need to rely on:

> one fixed source vector.

The model could learn:

> soft alignment while generating.

This was a major improvement for neural machine translation.

---

# ⭐ Part 32 — Alignment

For each target timestep:

```text
t
```

we get an attention distribution:

```text
α_(t,1), ..., α_(t,T)
```

This describes soft alignment between:

> target position `t`

and:

> source positions `1...T`.

Stack these distributions across target steps and you get:

> an attention / alignment matrix.

---

# 📐 Part 33 — Alignment Matrix Shape

If:

```text
source length = T_src
target length = T_tgt
```

then attention weights can be arranged as:

```text
T_tgt × T_src
```

Rows:

> target steps

Columns:

> source positions.

---

# 🧠 Part 34 — Bahdanau Attention Is Soft Alignment

The model does not need to pick exactly one source position.

Example:

```text
α_t =
[0.05, 0.15, 0.50, 0.30]
```

means multiple source states contribute.

This is differentiable and trainable end-to-end.

---

# 📐 Part 35 — Gradient Flow

Because:

```text
e
→ softmax
→ α
→ c_t
→ prediction
→ loss
```

is differentiable, gradients flow into:

* `v_a`
* `W_s`
* `W_h`
* `b_a`
* decoder state
* encoder states

So the model learns:

> how to align source and target from the final task loss.

---

# 🧠 Part 36 — No Separate Alignment Labels Required

In standard use, we do not need labels like:

```text
target token 3 ↔ source token 5
```

The attention scorer learns indirectly because:

> better alignments improve the output loss.

---

# ⭐ Part 37 — Why Additive Attention Helps Different Dimensions

Suppose:

```text
decoder hidden size = 256
encoder hidden size = 512
```

A raw dot product fails.

Bahdanau works because:

```text
W_s : 256 → H_a
W_h : 512 → H_a
```

Both are mapped to:

```text
R^(H_a)
```

before comparison.

---

# 🧠 Part 38 — Is the Attention Hidden Size Equal to Encoder Hidden Size?

Not necessarily.

The attention dimension:

```text
H_a
```

is a design choice.

It can differ from:

* encoder hidden size
* decoder hidden size

This gives architectural flexibility.

---

# ⚠️ Part 39 — Large Attention Dimension Trade-Off

Increasing `H_a` gives:

> more scoring capacity

but also increases:

* parameters;
* memory;
* computation.

So larger is not automatically better.

---

# 📖 Part 40 — Bahdanau With a Bidirectional Encoder

Suppose a BiLSTM encoder produces:

```text
h_i = [→h_i ; ←h_i]
```

Then each encoder state may have dimension:

```text
2H
```

Bahdanau attention works normally.

`W_h` simply projects this larger representation into:

```text
H_a
```

No conceptual change is required.

---

# ⭐ Part 41 — Bahdanau + LSTM Decoder

A classic architecture may look like:

```text
Source
↓
BiRNN / BiLSTM Encoder
↓
h1 ... hT
↓
Bahdanau Attention
↓
c_t
↓
RNN / LSTM Decoder
↓
Target
```

Attention does not replace recurrence here.

It improves:

> source-decoder communication.

---

# 🧠 Part 42 — What Exactly Is Learned?

The network learns:

```text
W_s
W_h
b_a
v_a
```

These parameters determine:

> what kinds of decoder-source interactions produce high compatibility scores.

The attention weights themselves are not permanent model parameters.

They are:

> dynamically computed values.

---

# 🚨 Part 43 — Parameters vs Attention Weights

Do not confuse:

### Learned parameters

```text
W_s, W_h, v_a, b_a
```

with:

### Dynamic outputs

```text
e_(t,i)
α_(t,i)
c_t
```

Parameters persist across examples.

Scores, weights, and contexts depend on the current input and decoder state.

---

# ⭐ Part 44 — Additive Attention Summary

Bahdanau scoring can be mentally decomposed as:

```text
Project Decoder
+
Project Source
↓
Add
↓
tanh
↓
Scalar Projection
↓
Score
```

Then generic attention continues:

```text
Scores
↓
Softmax
↓
Weights
↓
Weighted Sum
↓
Context
```

---

# 📐 Part 45 — Complete Equation Chain

### Alignment score

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

### Attention distribution

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

This is the complete Bahdanau attention mechanism.

---

# 🧠 Part 46 — Why `v_a^T` Is Needed

Without:

```text
v_a^T
```

the tanh output is still a vector.

But softmax needs:

> one score per source position.

So `v_a` performs the final vector-to-scalar projection.

---

# 🚨 Part 47 — Common Mistake: Softmax Over Attention Hidden Dimension

Suppose tanh output has:

```text
H_a = 128
```

You do **not** softmax those 128 internal features.

First:

```text
v_a^T
```

reduces each source comparison to a scalar.

Then softmax is applied across:

> source positions.

---

# 🧠 Part 48 — Why Not Softmax Before `v_a`?

Because the attention hidden vector is:

> an internal learned compatibility representation.

It is not yet a set of competing source-position scores.

Competition happens only after each source position has one scalar:

```text
e_(t,i)
```

---

# ⭐ Part 49 — Bahdanau vs Generic Attention

Generic:

```text
e_(t,i) = score(s, h_i)
```

Bahdanau:

```text
score(s,h_i)
=
v_a^T tanh(W_s s + W_h h_i + b_a)
```

Everything else:

```text
softmax
+
weighted sum
```

remains the same.

---

# 🧠 Part 50 — Why This Matters for Transformers Later

Bahdanau attention teaches a foundational idea:

> relevance can be learned by comparing one representation against many others.

That idea survives into Transformers.

But the implementation changes dramatically.

Do not jump to Query/Key/Value yet.

For now remember:

```text
decoder need
+
source representations
→ learned compatibility
```

---

# ⚠️ Part 51 — Limitations of Bahdanau Attention

Bahdanau attention improves source access, but several limitations remain.

### 1. Decoder is still recurrent

Target generation remains sequential.

### 2. Encoder may still be recurrent

Source processing may also remain sequential.

### 3. Every decoder step scans source states

This adds computation.

### 4. Scoring network has extra parameters

More flexible but more expensive.

### 5. Matrix multiplication efficiency is weaker than simpler dot-product methods

This motivates later multiplicative attention.

---

# 🧠 Part 52 — Why Luong Attention Becomes Natural

Once we understand Bahdanau, the next question is:

> Do we really need a small neural network to calculate every compatibility score?

Could we use something simpler?

For example:

```text
s_t^T h_i
```

or:

```text
s_t^T W_a h_i
```

This leads to:

# **Luong / Multiplicative Attention**

---

# 🔗 Part 53 — Evolution

```text
Generic Attention
↓
Need score(...)
↓
Bahdanau
↓
Learned nonlinear additive scorer
↓
Good flexibility
↓
But extra computation
↓
Can scoring be simpler?
↓
Luong / Multiplicative Attention
```

---

# 🎤 30-Second Interview Answer

> **Bahdanau attention, also called additive attention, implements the attention compatibility function using a small feed-forward neural network. The decoder state and each encoder state are projected into a shared attention space, added together, passed through a nonlinear activation such as tanh, and then projected to a scalar score using a learned vector. These scores are normalized with softmax to produce attention weights, which are used to compute the dynamic context vector.**

---

# 🎤 Why Is It Called Additive Attention?

> **Because the projected decoder state and encoder state are added together before the nonlinear transformation: `W_s s + W_h h_i + b`. This contrasts with multiplicative approaches that directly use dot-product or bilinear interactions.**

---

# 🎤 Why Use Projection Matrices?

> **The projections let encoder and decoder states have different original dimensions while mapping both into a common learned attention space where compatibility can be modeled.**

---

# 🚫 High-Yield Traps

### ❌ Bahdanau Attention is the generic attention mechanism itself

No.

It is one scoring-function implementation.

---

### ❌ Encoder and decoder states must have equal dimensions

No.

Projection matrices handle dimension differences.

---

### ❌ `v_a` is an attention weight

No.

It is a learned model parameter.

---

### ❌ `e_(t,i)` is a vector

No.

It is a scalar.

---

### ❌ Softmax is applied over `H_a`

No.

Softmax is across source-position scores.

---

### ❌ The model learns a separate scorer for every source position

No.

Parameters are shared.

---

### ❌ Attention weights are learned parameters

No.

They are dynamically computed.

---

### ❌ Bahdanau removes recurrence

No.

Classic Bahdanau attention is commonly used with recurrent encoder-decoder networks.

---

# 📐 Formula Card

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

### Typical Shapes

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

# 🧠 10 Things to Know Cold

1. Bahdanau Attention is **Additive Attention**.
2. It defines a specific attention scoring function.
3. Encoder and decoder states are projected into a common attention space.
4. Projected states are added.
5. A nonlinearity such as `tanh` is applied.
6. `v_a` converts the hidden compatibility representation into a scalar.
7. One scalar score is produced per source position.
8. Softmax is applied across source positions.
9. Parameters are shared across positions and timesteps.
10. Bahdanau is flexible but more computationally involved than simple multiplicative scoring.

---

# 🧠 Final Mental Model

Start with the generic question:

> **How relevant is this source state to what the decoder needs now?**

Bahdanau answers:

```text
Decoder State
↓
Learned Projection
          +
Encoder State
↓
Learned Projection
          ↓
         Add
          ↓
        tanh
          ↓
Learned Scalar Projection
          ↓
         Score
```

Then:

```text
Scores
↓
Softmax
↓
Attention Weights
↓
Weighted Source Context
```

Shortest mental model:

# **Project → Add → Nonlinearity → Score**

followed by:

# **Normalize → Aggregate**

---

# 🔗 Next Lecture

Bahdanau gives us a flexible learned neural scoring function.

The next engineering question is:

> **Can compatibility be calculated more simply and efficiently through multiplication?**

That leads directly to:

# **04_Luong_Multiplicative_Attention**

# Some Additional notes

## 🧠 How `W_s` and `W_h` Bring Decoder and Encoder States into a Common Attention Space

One of the most important ideas in Bahdanau Attention is this transformation:

```text
W_s s_(t-1)
+
W_h h_i
```

At first, this may look like ordinary matrix multiplication.

But these two matrices are doing something very important:

> **They transform the decoder state and encoder state into a shared representation space where they can be meaningfully compared.**

Let's understand this from first principles.

---

# 📖 Part 1 — The Original Problem

Suppose the decoder state is:

```text
s_(t-1) ∈ R^2
```

and an encoder state is:

```text
h_i ∈ R^3
```

For example:

```text
s_(t-1) = [1, 2]

h_i = [3, 4, 5]
```

The decoder vector contains:

```text
2 values
```

while the encoder vector contains:

```text
3 values
```

We cannot directly add them:

```text
[1, 2]
+
[3, 4, 5]

❌ Not possible
```

A direct dot product is also not possible because the dimensions are different.

So before asking:

> "How compatible are these two states?"

we first need to transform them into representations that can actually be compared.

---

# 🧠 Part 2 — Introduce an Attention Space

Bahdanau Attention introduces a new internal space called the:

> **attention space**

Let's choose its dimension as:

```text
H_a = 2
```

This means we want both:

```text
decoder representation
```

and:

```text
encoder representation
```

to eventually become vectors of size:

```text
2
```

Conceptually:

```text
Decoder State                  Encoder State

R^2                               R^3
 │                                 │
 │                                 │
W_s                               W_h
 │                                 │
 ▼                                 ▼
R^2                               R^2

          Common Attention Space
```

Now both representations have the same dimensionality.

---

# 📐 Part 3 — Project the Decoder State

Our decoder state is:

```text
s_(t-1) =
[1
 2]
```

Since:

```text
s_(t-1) ∈ R^2
```

and our attention dimension is:

```text
H_a = 2
```

we choose:

```text
W_s ∈ R^(2 × 2)
```

For a simple example, let:

```text
W_s =
[1  0
 0  1]
```

Now compute:

```text
W_s s_(t-1)
```

So:

```text
[1  0      [1
 0  1]  ×   2]
```

which gives:

```text
[1
 2]
```

Therefore:

```text
W_s s_(t-1) = [1, 2]
```

The decoder state is now represented inside the attention space.

---

# 📐 Part 4 — Project the Encoder State

Now consider:

```text
h_i =
[3
 4
 5]
```

Since:

```text
h_i ∈ R^3
```

but we want an attention-space vector of size:

```text
2
```

we need:

```text
W_h ∈ R^(2 × 3)
```

For example:

```text
W_h =
[1  0  0
 0  1  0]
```

Now compute:

```text
W_h h_i
```

So:

```text
[1  0  0      [3
 0  1  0]  ×   4
                5]
```

which gives:

```text
[3
 4]
```

Therefore:

```text
W_h h_i = [3, 4]
```

Now the encoder state is also represented inside:

```text
R^2
```

---

# ⭐ Part 5 — Now They Can Be Combined

We now have:

```text
Decoder projection:

W_s s_(t-1) = [1, 2]
```

and:

```text
Encoder projection:

W_h h_i = [3, 4]
```

Both are:

```text
R^2
```

Therefore we can add them:

```text
[1, 2]
+
[3, 4]
=
[4, 6]
```

This is what is meant by:

> **bringing the decoder state and encoder state into a common attention space.**

---

# 🧠 Part 6 — But Same Dimension Is Only Half the Story

It is tempting to think:

> "`W_s` and `W_h` are only used to make the vector dimensions equal."

That is incomplete.

The more important fact is:

> **`W_s` and `W_h` are learned matrices.**

Their values are not manually chosen like our simple example.

During training, gradient descent learns them.

So they learn:

> how the decoder representation should be transformed;

and:

> how the encoder representation should be transformed;

so that compatibility becomes easier to measure.

---

# 📖 Part 7 — Different Representation Languages

Imagine the decoder state conceptually contains information like:

```text
s_(t-1)

=
[
  grammatical_need,
  semantic_need
]
```

while an encoder state might contain something like:

```text
h_i

=
[
  word_identity,
  tense_information,
  entity_information
]
```

These representations are not directly comparable.

They are almost like two systems speaking different languages.

We can imagine:

```text
Decoder Representation
        │
        │ W_s
        ▼
[semantic relevance,
 structural relevance]
```

while:

```text
Encoder Representation
        │
        │ W_h
        ▼
[semantic relevance,
 structural relevance]
```

Now both representations describe information in terms useful for:

> **attention compatibility.**

---

# 🧠 Part 8 — The Translation Analogy

A useful mental model is:

```text
Decoder speaks Language A

Encoder speaks Language B

        ↓

W_s and W_h act like translators

        ↓

Both are translated into
"Attention Language"

        ↓

Now comparison becomes meaningful
```

This is why we call it a:

> **common attention space**

rather than merely:

> same-sized vectors.

---

# 📐 Part 9 — General Mathematical View

Suppose:

```text
decoder hidden size = H_d
encoder hidden size = H_e
attention size       = H_a
```

Then:

```text
s_(t-1) ∈ R^(H_d)
```

and:

```text
h_i ∈ R^(H_e)
```

We define:

```text
W_s ∈ R^(H_a × H_d)
```

and:

```text
W_h ∈ R^(H_a × H_e)
```

Therefore:

```text
W_s s_(t-1)
```

has shape:

```text
R^(H_a)
```

and:

```text
W_h h_i
```

also has shape:

```text
R^(H_a)
```

So:

```text
W_s s_(t-1)
+
W_h h_i
```

is now mathematically valid.

---

# 🧮 Part 10 — A More Realistic Shape Example

Suppose:

```text
decoder hidden size = 256
encoder hidden size = 512
attention size       = 128
```

Then:

```text
s_(t-1) ∈ R^256
```

and:

```text
h_i ∈ R^512
```

Choose:

```text
W_s ∈ R^(128 × 256)
```

and:

```text
W_h ∈ R^(128 × 512)
```

Now:

```text
W_s s_(t-1)
→ R^128
```

and:

```text
W_h h_i
→ R^128
```

Therefore:

```text
R^128
+
R^128
```

is possible.

The resulting vector lives in the:

```text
128-dimensional attention space
```

---

# 🔗 Part 11 — What Happens After the Projection?

Once both states are projected:

```text
W_s s_(t-1)
```

and:

```text
W_h h_i
```

Bahdanau combines them:

```text
z_(t,i)
=
W_s s_(t-1)
+
W_h h_i
+
b_a
```

Then applies:

```text
tanh
```

to obtain a nonlinear compatibility representation:

```text
u_(t,i)
=
tanh(z_(t,i))
```

So:

```text
u_(t,i) ∈ R^(H_a)
```

But attention ultimately needs:

```text
one scalar score
```

for this source position.

Therefore:

```text
v_a^T
```

projects the attention-space vector to a scalar:

```text
e_(t,i)
=
v_a^T u_(t,i)
```

Putting everything together:

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

# 🧠 Part 12 — Full Visual Flow

```text
Decoder State
s_(t-1) ∈ R^(H_d)
        │
        │ W_s
        ▼
     R^(H_a)
        │
        │
        ├──────────────┐
                       │
                       +
                       │
        ├──────────────┘
        │
        │
Encoder State
h_i ∈ R^(H_e)
        │
        │ W_h
        ▼
     R^(H_a)

        ↓

      + b_a

        ↓

       tanh

        ↓

Compatibility Representation
      R^(H_a)

        ↓
       v_a^T

        ↓

Scalar Score
e_(t,i) ∈ R
```

---

# ⭐ Part 13 — What Exactly Is Being Learned?

The model learns:

```text
W_s
W_h
b_a
v_a
```

During training, these parameters gradually learn:

```text
What information in the decoder state
is useful for deciding where to look?

+

What information in the encoder state
is useful for deciding whether this
source position is relevant?
```

For example, the model may learn that when the decoder needs:

```text
a person's name
```

certain dimensions of the decoder state should interact strongly with encoder representations containing:

```text
entity information
```

That behavior emerges through training.

---

# 🚨 Part 14 — Important Misconception

Do not think:

```text
W_s = resize decoder vector

W_h = resize encoder vector
```

A better mental model is:

```text
W_s
→ learned decoder representation for attention

W_h
→ learned encoder representation for attention
```

Yes, they also make the dimensions compatible.

But their deeper role is:

> **representation learning for compatibility.**

---

# 🧠 Part 15 — Why Not Compare the Original States Directly?

Because the original encoder and decoder states were learned for different jobs.

The encoder state primarily represents:

```text
source information
```

while the decoder state primarily represents:

```text
target-generation history
```

Attention asks a different question:

> **How relevant is this source representation to the decoder's current need?**

So Bahdanau learns new transformations specifically for that question.

---

# 🎤 Interview Answer

If asked:

> **Why do Bahdanau Attention use `W_s` and `W_h`?**

A strong answer is:

> `W_s` and `W_h` are learned projection matrices that transform the decoder state and encoder state into a shared attention space. This is useful because the two states may have different dimensions and may encode information differently. Once projected into the same attention dimension, they can be combined and passed through the nonlinear alignment network to compute a compatibility score.

---

# ⭐ Final Mental Model

Remember:

```text
Encoder Representation
and
Decoder Representation

may have

Different Dimensions
+
Different Semantic Roles

        ↓

W_h and W_s

        ↓

Learned Projections

        ↓

Shared Attention Space

        ↓

Meaningful Compatibility Calculation
```

The shortest way to remember it is:

> **`W_s` and `W_h` do not merely make the vectors the same size — they teach the model how to express both vectors in a common language for attention.**
