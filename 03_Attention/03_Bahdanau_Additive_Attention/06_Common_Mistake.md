# 🚫 Bahdanau / Additive Attention — Common Mistakes

> Goal: Avoid the most common conceptual, mathematical, implementation, and interview mistakes around Bahdanau Attention.

---

# 1. ❌ Thinking Bahdanau Attention Is the Entire Attention Mechanism

Wrong:

> Bahdanau Attention is completely different from generic attention.

Correct:

Bahdanau mainly defines the:

```text
score(...)
```

function inside generic attention.

Generic attention:

```text
e_(t,i) = score(s_(t-1), h_i)
```

Bahdanau:

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

After that, the normal attention pipeline remains:

```text
score
↓
softmax
↓
attention weights
↓
weighted sum
↓
context
```

---

# 2. ❌ Thinking `W_s` and `W_h` Only Resize Vectors

Incomplete mental model:

```text
W_s
→ resize decoder vector

W_h
→ resize encoder vector
```

Better mental model:

```text
W_s
→ learned decoder representation for attention

W_h
→ learned encoder representation for attention
```

Yes, they also solve dimension mismatch.

But more importantly, they learn how to express both representations in a shared space useful for:

> compatibility scoring.

---

# 3. ❌ Assuming Encoder and Decoder Hidden Sizes Must Match

Not true.

Suppose:

```text
s_(t-1) ∈ R^256

h_i ∈ R^512
```

Bahdanau can project both to:

```text
R^128
```

using:

```text
W_s ∈ R^(128 × 256)

W_h ∈ R^(128 × 512)
```

So:

```text
W_s s_(t-1) ∈ R^128

W_h h_i ∈ R^128
```

Now they can be combined.

---

# 4. ❌ Confusing "Same Dimension" with "Same Representation"

Two vectors having the same size does not automatically mean they are meaningfully comparable.

For example:

```text
decoder state ∈ R^256
encoder state ∈ R^256
```

Even though their dimensions match, they may encode different kinds of information.

Bahdanau still uses learned projections because the objective is not only:

> same shape

but also:

> a useful common attention representation.

---

# 5. ❌ Thinking the Common Attention Space Is a Fixed Semantic Meaning

The attention space does not come with predefined dimensions like:

```text
dimension 1 = grammar
dimension 2 = entity
dimension 3 = tense
```

Those meanings are not manually assigned.

The model learns whatever internal representation helps minimize the training loss.

So the common attention space is:

> learned, distributed, and task-dependent.

---

# 6. ❌ Forgetting Why It Is Called Additive Attention

The defining interaction is:

```text
W_s s_(t-1)
+
W_h h_i
+
b_a
```

The projected encoder and decoder states are:

> added together

before the nonlinear transformation.

That is why it is called:

# **Additive Attention**

---

# 7. ❌ Thinking `tanh(...)` Already Produces the Attention Score

Wrong:

```text
tanh(...)
=
scalar score
```

Usually:

```text
tanh(...)
∈ R^(H_a)
```

It is still a vector.

Bahdanau then uses:

```text
v_a^T
```

to convert that vector into:

```text
e_(t,i) ∈ R
```

one scalar compatibility score.

---

# 8. ❌ Confusing `v_a` with Attention Weights

`v_a` is not:

```text
α_t
```

and it is not a distribution over source positions.

`v_a` is:

> a learned model parameter used to convert the attention hidden representation into a scalar score.

Attention weights are:

```text
α_(t,i)
```

and are dynamically computed after softmax.

---

# 9. ❌ Applying Softmax Over the Attention Hidden Dimension

Suppose:

```text
H_a = 128
```

The `tanh(...)` result has 128 features.

Do not softmax over those 128 values.

The correct process is:

```text
attention hidden vector
↓
v_a^T
↓
one scalar score per source position
↓
softmax across source positions
```

The competition is between:

> source positions,

not between attention hidden features.

---

# 10. ❌ Forgetting That `e_(t,i)` Is a Scalar

For each:

```text
target step t
```

and:

```text
source position i
```

Bahdanau produces:

```text
e_(t,i) ∈ R
```

one scalar.

Across all source positions:

```text
e_t =
[e_(t,1), ..., e_(t,T)]
```

Then softmax is applied.

---

# 11. ❌ Using the Projected Encoder State as the Context by Default

Bahdanau uses:

```text
W_h h_i
```

for compatibility scoring.

But the standard context is:

```text
c_t
=
Σ_i α_(t,i) h_i
```

using the original encoder states.

So remember:

```text
W_h h_i
→ scoring representation

h_i
→ value being aggregated
```

---

# 12. ❌ Thinking `W_h h_i` Must Be Recomputed at Every Decoder Step

For a fixed source sequence:

```text
h_1, ..., h_T
```

the encoder states do not change during decoding.

Therefore:

```text
W_h h_i
```

can be computed once and cached.

What changes each decoder step is:

```text
W_s s_(t-1)
```

because the decoder state changes.

---

# 13. ❌ Thinking Each Source Position Has Its Own Attention Parameters

Wrong:

```text
W_h^(1)
W_h^(2)
W_h^(3)
...
```

Standard Bahdanau Attention shares the same:

```text
W_s
W_h
b_a
v_a
```

across:

* source positions
* decoder timesteps
* examples

The dynamic states change.

The parameters do not.

---

# 14. ❌ Confusing Parameters with Dynamic Attention Values

### Learned parameters

```text
W_s
W_h
b_a
v_a
```

### Dynamic values

```text
e_(t,i)
α_(t,i)
c_t
```

Do not say:

> the model directly learns the attention weights.

It learns parameters that generate attention weights dynamically.

---

# 15. ❌ Saying Bahdanau Requires Explicit Alignment Labels

Not normally.

The entire path is differentiable:

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
```

Backpropagation trains the attention parameters from the final task objective.

So source-target alignment can emerge without explicit alignment annotations.

---

# 16. ❌ Saying Bahdanau Removes Recurrence

No.

Classic Bahdanau Attention was designed for recurrent encoder-decoder systems.

A typical architecture is:

```text
BiRNN / BiLSTM Encoder
↓
Bahdanau Attention
↓
RNN / LSTM Decoder
```

Attention improves:

> source access.

It does not automatically remove recurrent computation.

---

# 17. ❌ Saying Bahdanau Is Always Better Than Dot-Product Attention

Not necessarily.

Bahdanau can be more flexible because it uses:

* learned projections
* nonlinear scoring
* a learned readout vector

But it also costs more:

* extra parameters
* extra computation
* less direct matrix-multiplication efficiency

So:

> more expressive does not mean universally better.

---

# 18. ❌ Overstating Why `tanh` Is Used

Weak answer:

> "Bahdanau Attention works because tanh is the best activation."

No.

The important point is:

> a nonlinearity gives the scoring function richer modeling capacity.

`tanh` is historically common, but the deeper principle is nonlinear compatibility modeling.

---

# ⚡ Quick Debug Checklist

If a Bahdanau Attention implementation behaves incorrectly, check:

1. Are `W_s s` and `W_h h_i` projected to the same attention dimension?
2. Are the projected vectors being added correctly?
3. Is bias broadcasting correct?
4. Is `tanh` applied before `v_a^T`?
5. Does each source position produce exactly one scalar score?
6. Is softmax applied across source positions?
7. Are padding positions masked before softmax?
8. Are original encoder states used for the context?
9. Are encoder-side projections being unnecessarily recomputed?
10. Are `W_s`, `W_h`, `v_a`, and `b_a` shared rather than recreated per step?

---

# 📐 Formula Reminder

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

---

# 🧠 Role Separation

Keep these roles distinct:

```text
s_(t-1)
→ decoder's current target-side state

h_i
→ source representation

W_s
→ decoder-to-attention projection

W_h
→ encoder-to-attention projection

b_a
→ attention bias

tanh(...)
→ nonlinear compatibility features

v_a
→ learned scalar readout direction

e_(t,i)
→ raw compatibility score

α_(t,i)
→ normalized source relevance

c_t
→ weighted source context
```

---

# ⭐ Final Mental Model

Most mistakes disappear if you remember the pipeline in two stages.

## Stage 1 — Build the Score

```text
Decoder State
↓ W_s

Encoder State
↓ W_h

        ↓

Common Attention Space

        ↓

Add

        ↓

tanh

        ↓

v_a^T

        ↓

Scalar Score
```

## Stage 2 — Use the Score

```text
Scores
↓
Softmax
↓
Attention Weights
↓
Weighted Encoder States
↓
Context
```

Shortest memory version:

# **Project → Add → Nonlinearity → Score → Normalize → Aggregate**

---

# 🔗 Next Topic

Bahdanau uses a learned nonlinear scoring network.

The next question is:

> **Can we calculate source-decoder compatibility more simply using multiplication?**

That takes us to:

# **Luong / Multiplicative Attention**
