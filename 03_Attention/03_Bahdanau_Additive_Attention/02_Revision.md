# 📝 Bahdanau / Additive Attention — Revision Notes

> Goal: Reconstruct Bahdanau Attention quickly from first principles, including why the projection matrices exist, how the score is computed, and how the mechanism fits into generic attention.

---

# 🎯 One-Line Definition

> **Bahdanau Attention computes source-decoder compatibility using a small learned nonlinear neural network before converting those scores into attention weights.**

It is also called:

# **Additive Attention**

---

# 🧠 Core Problem

From generic attention, we already had:

```text
e_(t,i) = score(s_(t-1), h_i)
```

The unresolved question was:

> **How should `score(...)` actually be implemented?**

Bahdanau's answer:

```text
Project decoder state
+
Project encoder state
↓
Add
↓
tanh
↓
Scalar projection
↓
Compatibility score
```

---

# 📐 Core Equation

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

Then generic attention continues:

```text
α_(t,i) = softmax_i(e_(t,i))
```

and:

```text
c_t = Σ_i α_(t,i) h_i
```

---

# ⭐ Core Mental Model

Bahdanau scoring:

# **Project → Add → Nonlinearity → Score**

Then generic attention:

# **Normalize → Aggregate**

Full pipeline:

```text
Decoder State
+
Encoder State
↓
Learned Projections
↓
Common Attention Space
↓
Add
↓
tanh
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

# 🧠 Why Do We Need `W_s` and `W_h`?

Encoder and decoder states may have:

* different dimensions
* different semantic roles
* different learned representation spaces

Example:

```text
s_(t-1) ∈ R^256
h_i     ∈ R^512
```

They cannot be directly added.

So Bahdanau learns:

```text
W_s ∈ R^(H_a × 256)
W_h ∈ R^(H_a × 512)
```

Then:

```text
W_s s_(t-1) ∈ R^(H_a)
```

and:

```text
W_h h_i ∈ R^(H_a)
```

Now both live in the same:

```text
attention space R^(H_a)
```

---

# ⭐ Important: Projection Means More Than Resizing

Do not think:

```text
W_s = resize decoder
W_h = resize encoder
```

Better interpretation:

```text
W_s
→ learned decoder representation for attention

W_h
→ learned encoder representation for attention
```

They learn how to express both representations in a shared "attention language."

---

# 🧠 Common Attention Space

Conceptually:

```text
Decoder speaks one representation language

Encoder speaks another representation language

        ↓

W_s and W_h

        ↓

Both translated into
the same attention language

        ↓

Compatibility can be measured
```

This is the meaning of:

> **common attention space**

---

# 📐 Shape Intuition

Suppose:

```text
decoder hidden size = H_d
encoder hidden size = H_e
attention size       = H_a
```

Then:

```text
s_(t-1) ∈ R^(H_d)
h_i     ∈ R^(H_e)

W_s ∈ R^(H_a × H_d)
W_h ∈ R^(H_a × H_e)
```

After projection:

```text
W_s s_(t-1) ∈ R^(H_a)

W_h h_i ∈ R^(H_a)
```

So they can be added.

---

# 🧮 Small Shape Example

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

W_s ∈ R^(128 × 256)
W_h ∈ R^(128 × 512)
```

Therefore:

```text
W_s s_(t-1) → R^128
W_h h_i     → R^128
```

Now:

```text
R^128 + R^128
```

is valid.

---

# 📐 Step 1 — Project Decoder State

```text
W_s s_(t-1)
```

Interpretation:

> transform the decoder's current need into the attention space.

---

# 📐 Step 2 — Project Encoder State

```text
W_h h_i
```

Interpretation:

> transform the source representation into the same attention space.

---

# 📐 Step 3 — Add Them

```text
z_(t,i)
=
W_s s_(t-1)
+
W_h h_i
+
b_a
```

Now:

```text
z_(t,i) ∈ R^(H_a)
```

This is a joint compatibility representation.

---

# 🧠 Why Is It Called Additive Attention?

Because the projected representations are combined using:

```text
+
```

Specifically:

```text
W_s s_(t-1)
+
W_h h_i
+
b_a
```

Hence:

# **Additive Attention**

---

# 📐 Step 4 — Apply `tanh`

```text
u_(t,i)
=
tanh(z_(t,i))
```

This introduces nonlinearity.

Without the nonlinearity, stacked linear operations would still behave like a linear mapping.

---

# 🧠 Why `tanh`?

Main reason:

> give the scoring network nonlinear modeling capacity.

Historically, `tanh` was a natural choice in recurrent architectures.

Do not memorize:

> "Bahdanau must always use tanh"

as a universal architectural law.

---

# 📐 Step 5 — Convert to Scalar

After `tanh`:

```text
u_(t,i) ∈ R^(H_a)
```

But attention needs one scalar score per source position.

So use:

```text
v_a ∈ R^(H_a)
```

and:

```text
e_(t,i)
=
v_a^T u_(t,i)
```

Therefore:

```text
e_(t,i) ∈ R
```

---

# 🧠 What Does `v_a` Do?

`v_a` is a learned:

> vector-to-scalar readout.

It converts the nonlinear compatibility representation into one number:

```text
compatibility representation
↓
v_a^T
↓
compatibility score
```

---

# 🚨 `v_a` Is NOT an Attention Weight

Do not confuse:

```text
v_a
```

with:

```text
α_(t,i)
```

`v_a`:

> learned model parameter

`α_(t,i)`:

> dynamically computed attention weight

---

# 📐 Step 6 — Softmax

Once we have:

```text
e_(t,1), e_(t,2), ..., e_(t,T)
```

apply softmax across source positions:

```text
α_(t,i)
=
exp(e_(t,i))
/
Σ_j exp(e_(t,j))
```

For fixed `t`:

```text
Σ_i α_(t,i) = 1
```

---

# 📐 Step 7 — Build Context

```text
c_t
=
Σ_i α_(t,i) h_i
```

Important:

The context is still computed from the original encoder states:

```text
h_i
```

not from:

```text
W_h h_i
```

in the standard formulation.

The projected representation is used for:

> scoring compatibility.

The original encoder state contributes to:

> the context vector.

---

# ⭐ Bahdanau Is a Scoring Function

Very important distinction:

Bahdanau Attention is not a completely different generic attention pipeline.

Generic attention says:

```text
e_(t,i) = score(s, h_i)
```

Bahdanau says:

```text
score(s,h_i)
=
v_a^T tanh(
W_s s
+
W_h h_i
+
b_a
)
```

Everything after scoring remains:

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

# 🧠 Learned Parameters vs Dynamic Values

### Learned parameters

```text
W_s
W_h
b_a
v_a
```

These persist after training.

### Dynamic values

```text
e_(t,i)
α_(t,i)
c_t
```

These depend on the current:

* source sentence
* source position
* decoder timestep

---

# 🧠 Parameter Sharing

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

We do not learn a separate scoring network for each word position.

---

# 📐 Parameter Count Intuition

Approximate attention parameter count:

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

So increasing `H_a` increases:

* modeling capacity
* parameters
* computation

---

# 🧠 Why Different Encoder and Decoder Dimensions Are Fine

Example:

```text
decoder = 256 dimensions
encoder = 512 dimensions
```

Direct dot product:

```text
❌
```

Bahdanau:

```text
256
↓ W_s
128

512
↓ W_h
128
```

Now both are comparable.

This flexibility is one major property of additive attention.

---

# ⭐ Source Projection Can Be Reused

For one encoded source:

```text
h_1, ..., h_T
```

the values:

```text
W_h h_i
```

do not change during decoding.

So they can be precomputed.

At each decoder timestep, only the decoder-side projection changes:

```text
W_s s_(t-1)
```

because:

```text
s_(t-1)
```

changes.

---

# 🧠 Classic Timing Intuition

A common Bahdanau formulation uses:

```text
s_(t-1)
```

to calculate attention for target step `t`.

Conceptually:

```text
What have I generated so far?
↓
What source information do I need next?
↓
Compute c_t
```

But notation and exact ordering can vary across implementations.

Do not claim that every attention implementation must use exactly the previous decoder state.

---

# 🔗 Bahdanau with BiRNN / BiLSTM Encoder

If:

```text
h_i = [→h_i ; ←h_i]
```

then encoder state may have dimension:

```text
2H
```

No problem.

`W_h` simply projects:

```text
2H
↓
H_a
```

into the attention space.

---

# 🔗 Bahdanau with LSTM Decoder

Classic architecture:

```text
Source
↓
BiRNN / BiLSTM Encoder
↓
h_1 ... h_T
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

> encoder-decoder communication.

---

# 🧠 Alignment

At each decoder timestep:

```text
t
```

we get:

```text
α_(t,1), ..., α_(t,T)
```

This gives a soft alignment between:

```text
target position t
```

and:

```text
source positions 1...T
```

Across all target steps, the alignment matrix is approximately:

```text
T_target × T_source
```

---

# 🧠 Why No Alignment Labels Are Needed

Training pipeline:

```text
score
↓
softmax
↓
context
↓
decoder prediction
↓
loss
↓
backpropagation
```

Because the entire chain is differentiable, the task loss can train:

```text
W_s
W_h
v_a
b_a
```

without explicit source-target alignment labels.

---

# ⚠️ Limitations

Bahdanau Attention is flexible, but:

* recurrent decoding is still sequential;
* recurrent encoding may also be sequential;
* every target step must score source positions;
* the scoring network introduces extra parameters;
* nonlinear scoring is computationally heavier than simpler multiplicative scoring.

This motivates the next question:

> Can compatibility be computed more simply?

---

# 🔗 Bahdanau → Luong

Evolution:

```text
Generic Attention
↓
Need a scoring function
↓
Bahdanau
↓
Learned nonlinear additive scorer
↓
Flexible
↓
But more computation
↓
Can multiplication be enough?
↓
Luong Attention
```

---

# 🎤 30-Second Interview Answer

> **Bahdanau Attention, or additive attention, computes compatibility using a small learned neural network. The decoder state and encoder state are projected into a shared attention space using separate learned matrices, added together, passed through a nonlinear activation such as tanh, and then projected to a scalar using a learned vector. The resulting scores are softmax-normalized across source positions and used to create the dynamic context vector.**

---

# 🎤 Why `W_s` and `W_h`?

> **They map encoder and decoder representations into a common learned attention space. This handles different hidden-state dimensions and also lets the model learn representations specifically useful for measuring source-decoder compatibility.**

---

# 🎤 Why Is It Called Additive?

> **Because the projected encoder and decoder representations are added before the nonlinear scoring transformation.**

---

# 🚨 High-Yield Mistakes

### ❌ `W_s` and `W_h` only resize vectors

No.

They are learned attention-specific transformations.

---

### ❌ Encoder and decoder dimensions must match

No.

The projection matrices can map both to `H_a`.

---

### ❌ `v_a` is the attention distribution

No.

It is a learned parameter.

---

### ❌ Softmax operates over attention hidden features

No.

Softmax operates over source-position scores.

---

### ❌ `e_(t,i)` is a vector

No.

It is a scalar.

---

### ❌ `W_h h_i` is always the context value

No.

It is used for compatibility scoring in the standard formulation.

---

### ❌ Attention weights are permanent learned parameters

No.

They are computed dynamically.

---

### ❌ Bahdanau removes recurrence

No.

Classic Bahdanau attention is used with recurrent encoder-decoder architectures.

---

# 📐 Formula Card

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

```text
α_(t,i)
=
softmax_i(e_(t,i))
```

```text
c_t
=
Σ_i α_(t,i) h_i
```

Shapes:

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

# ⚡ 10 Things to Know Cold

1. Bahdanau = Additive Attention.
2. It defines the generic attention `score(...)` function.
3. `W_s` projects the decoder state.
4. `W_h` projects the encoder state.
5. Both projections enter the same attention space `R^(H_a)`.
6. Their projected representations are added.
7. `tanh` creates nonlinear compatibility features.
8. `v_a^T` turns those features into one scalar score.
9. Softmax converts source-position scores into attention weights.
10. Bahdanau is flexible but computationally heavier than simple multiplicative scoring.

---

# 🧠 Final Mental Model

```text
Decoder State
↓ W_s

            Common Attention Space
                    +

Encoder State
↓ W_h

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
             Weighted Context
```

Shortest version:

# **Translate both → Combine → Score**

Then:

# **Normalize → Retrieve**

---

# 🔗 Next Topic

The remaining question is:

> **Do we really need a nonlinear neural network for every compatibility calculation?**

That leads to:

# **Luong / Multiplicative Attention**
