# 🚫 Luong / Multiplicative Attention — Common Mistakes

> Goal: Avoid the most common conceptual, mathematical, implementation, and interview mistakes around Luong Attention.

---

# 1. ❌ Thinking Luong Attention Means Only Dot Attention

Luong is commonly associated with multiple scoring functions.

The two most important multiplicative variants are:

```text
Dot:
e_(t,i) = s_t^T h_i
```

and:

```text
General:
e_(t,i) = s_t^T W_a h_i
```

So:

> **Luong Attention is broader than only plain dot-product scoring.**

---

# 2. ❌ Thinking Dot Attention Works for Any Dimensions

Wrong:

```text
s_t ∈ R^256
h_i ∈ R^512

s_t^T h_i   ✅
```

This is invalid.

A direct dot product requires compatible dimensions:

```text
dim(s_t) = dim(h_i)
```

If dimensions differ, use a transformation such as:

```text
e_(t,i)
=
s_t^T W_a h_i
```

---

# 3. ❌ Thinking Equal Dimensions Automatically Mean Dot Attention Is Ideal

Suppose:

```text
s_t ∈ R^256
h_i ∈ R^256
```

A direct dot product is mathematically valid.

But that does not mean:

> the two representation spaces are automatically well aligned.

Equal dimension only means:

> the multiplication is possible.

General Attention can still be useful because:

```text
W_a
```

learns how encoder features should be transformed before comparison.

---

# 4. ❌ Thinking `W_a` Is the Attention Weight Matrix

This is one of the most common notation mistakes.

`W_a` is:

> a learned model parameter used inside the scoring function.

For General Attention:

```text
e_(t,i)
=
s_t^T W_a h_i
```

The actual dynamically computed attention weights are:

```text
α_(t,i)
```

after softmax.

So:

```text
W_a ≠ α_t
```

---

# 5. ❌ Thinking `W_a` Only Fixes Dimensions

Incomplete explanation:

```text
W_a
→ resize h_i
```

Better:

```text
W_a
→ learned transformation of encoder representation
   for compatibility with the decoder state
```

It may solve dimension mismatch, but its deeper role is:

> representation transformation for multiplicative scoring.

---

# 6. ❌ Confusing Raw Score with Attention Weight

For Luong Dot:

```text
e_(t,i)
=
s_t^T h_i
```

For General:

```text
e_(t,i)
=
s_t^T W_a h_i
```

These are:

> raw compatibility scores.

They are not yet attention weights.

Correct pipeline:

```text
Raw Scores
↓
Softmax
↓
Attention Weights
```

So:

```text
e_(t,i) ≠ α_(t,i)
```

---

# 7. ❌ Forgetting Softmax Is Still Required

Luong simplifies:

> how scores are computed.

It does not remove normalization.

Wrong:

```text
s_t^T h_i
↓
use directly as context weight
```

Correct:

```text
s_t^T h_i
↓
scores
↓
softmax across source positions
↓
α_(t,i)
↓
weighted context
```

---

# 8. ❌ Applying Softmax Across the Wrong Dimension

Suppose scores have shape:

```text
[batch_size, source_length]
```

Attention should normally normalize across:

```text
source_length
```

not across:

```text
batch_size
```

So if:

```text
scores.shape = [32, 20]
```

then each of the 32 examples should have its own distribution over 20 source positions.

---

# 9. ❌ Thinking Dot Product Is the Same as Cosine Similarity

Dot product:

```text
s_t^T h_i
```

depends on:

* direction
* magnitude

Cosine similarity is:

```text
(s_t^T h_i)
/
(||s_t|| ||h_i||)
```

and explicitly normalizes vector magnitudes.

Therefore:

```text
dot product ≠ cosine similarity
```

unless vectors are appropriately normalized.

---

# 10. ❌ Ignoring Vector Magnitude

Suppose:

```text
a = [1, 0]

b = [1, 0]

c = [10, 0]
```

Then:

```text
a^T b = 1
```

while:

```text
a^T c = 10
```

even though `b` and `c` point in exactly the same direction.

So dot-product attention is affected by:

> representation scale.

This becomes important later when learning:

# **Scaled Dot-Product Attention**

---

# 11. ❌ Thinking Larger Dot Scores Always Mean Better Alignment

A large score can occur because:

* vectors are well aligned;
* vector magnitudes are large;
* both effects happen together.

So do not interpret:

```text
large dot product
```

as pure semantic similarity.

The representations are learned, and score magnitude depends on the learned vector space.

---

# 12. ❌ Confusing General Luong Attention with Bahdanau Attention

Luong General:

```text
e_(t,i)
=
s_t^T W_a h_i
```

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

They are structurally different.

Mental model:

```text
Bahdanau
→ nonlinear additive scoring

Luong General
→ bilinear multiplicative scoring
```

The presence of a learned matrix in both does not make them equivalent.

---

# 13. ❌ Thinking Luong Removes the Need for the Context Vector

No.

Luong still computes:

```text
c_t
=
Σ_i α_(t,i) h_i
```

The multiplicative mechanism only changes:

> how compatibility scores are produced.

The weighted source context remains part of classical attention.

---

# 14. ❌ Confusing `c_t` and `s_t`

These play different roles.

```text
s_t
→ decoder / target-history representation
```

```text
c_t
→ source information retrieved through attention
```

They are complementary, not interchangeable.

---

# 15. ❌ Forgetting the Attentional Hidden State

Luong commonly combines:

```text
c_t
```

and:

```text
s_t
```

using:

```text
[c_t ; s_t]
```

followed by:

```text
h̃_t
=
tanh(
W_c [c_t ; s_t]
)
```

So the prediction may use:

```text
h̃_t
```

rather than only the raw decoder state.

Keep these distinct:

```text
s_t
≠
c_t
≠
h̃_t
```

---

# 16. ❌ Thinking `h̃_t` Is Another Encoder Hidden State

No.

`h̃_t` is the:

> attentional hidden state

formed after combining:

```text
decoder state
+
attended source context
```

It belongs to the decoder-side prediction process.

---

# 17. ❌ Saying Luong Always Uses `s_t` and Bahdanau Always Uses `s_(t-1)`

This is too absolute.

A common historical description is:

```text
Bahdanau
→ often previous decoder state

Luong
→ often current decoder state
```

But notation and implementation details vary.

Better interview wording:

> Luong-style attention is commonly described using the current decoder state, while Bahdanau-style attention is commonly described using the previous state during alignment computation.

---

# 18. ❌ Thinking Matrix Multiplication Means Luong Is Fully Parallel

Luong scoring is matrix-friendly.

For example:

```text
e_t
=
H_enc s_t
```

can score all source positions efficiently.

But the classical decoder is still recurrent:

```text
s_1
↓
s_2
↓
s_3
↓
...
```

So:

> efficient attention scoring does not remove autoregressive sequential decoding.

---

# 19. ❌ Thinking Luong Attention Is Transformer Self-Attention

Luong:

```text
decoder state
attends to
encoder states
```

Transformer self-attention:

```text
sequence positions
attend to
other positions in the same sequence
```

These are not the same architecture.

Both may use dot products, but:

> shared mathematical operation does not imply shared architectural role.

---

# 20. ❌ Introducing Query/Key/Value Too Early

It is tempting to say:

```text
s_t = Query
h_i = Key
h_i = Value
```

This can be a useful future analogy.

But classical Luong Attention is best understood first in its own terms:

```text
decoder state
+
encoder states
↓
multiplicative compatibility
```

Formal Query/Key/Value terminology belongs naturally to the Transformer progression.

---

# 21. ❌ Thinking Global Attention Means One Global Context for the Whole Sequence

The word:

> **Global**

does not mean:

```text
one fixed context vector
```

It means:

> each decoder timestep considers all source positions.

So:

```text
Global Attention
=
full-source search at each target step
```

The context is still dynamic:

```text
c_1, c_2, ..., c_T
```

---

# 22. ❌ Thinking Local Attention Uses One Fixed Source Window

Local attention is not necessarily:

```text
always look at positions 1–10
```

The relevant source region can depend on:

```text
decoder timestep
```

Conceptually:

```text
decoder state
↓
predict relevant source region
↓
attend within local window
```

So the window can move during decoding.

---

# 23. ❌ Thinking Local Attention Is Always Better Than Global Attention

Local attention can reduce computation.

But it introduces a risk:

```text
important source information
outside selected window
↓
cannot be attended to
```

So the trade-off is:

```text
Global
→ higher accessibility + higher compute

Local
→ lower compute + possible information miss
```

---

# 24. ❌ Calling the Concat Variant Multiplicative

Luong discussed a concat-style scoring alternative such as:

```text
v_a^T tanh(
W_a [s_t ; h_i]
)
```

This contains a nonlinear neural scoring network.

So when explaining:

> **Luong / Multiplicative Attention**

focus primarily on:

```text
Dot
General
```

as the multiplicative variants.

---

# 25. ❌ Saying Luong Is Always Faster Than Bahdanau

A better statement is:

> Luong Dot/General scoring is structurally simpler and often maps more efficiently to matrix multiplication.

Actual system performance still depends on:

* implementation
* tensor dimensions
* hardware
* batching
* source length
* decoder length

Do not turn an architectural tendency into an absolute performance guarantee.

---

# 26. ❌ Saying Simpler Scoring Means Lower Quality

Not necessarily.

Luong scoring is structurally simpler than Bahdanau.

But task quality depends on:

* representation learning
* data
* model capacity
* optimization
* architecture
* decoding strategy

A simpler compatibility function can still perform very well.

---

# ⚡ Quick Debug Checklist

If a Luong Attention implementation behaves incorrectly, check:

1. Do Dot Attention dimensions match?
2. Does `W_a` have the correct orientation in General Attention?
3. Does each source position produce exactly one scalar score?
4. Is softmax applied across source positions?
5. Are PAD positions masked before softmax?
6. Is `W_a` being confused with dynamic attention weights?
7. Is the context computed using weighted encoder states?
8. Are `s_t`, `c_t`, and `h̃_t` being kept conceptually separate?
9. Is matrix multiplication orientation correct?
10. Is recurrent decoding still being accounted for in latency analysis?

---

# 📐 Formula Reminder

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

# 🧠 Role Separation

Keep these quantities distinct:

```text
s_t
→ current decoder state

h_i
→ source representation

W_a
→ learned compatibility transformation

e_(t,i)
→ raw multiplicative score

α_(t,i)
→ normalized source relevance

c_t
→ weighted source context

W_c
→ learned source-decoder combination matrix

h̃_t
→ attentional hidden state
```

---

# ⭐ Bahdanau vs Luong — Mistake-Proof Mental Model

Bahdanau:

```text
Decoder
↓ project

Encoder
↓ project

Add
↓
tanh
↓
scalar score
```

Luong Dot:

```text
Decoder
×
Encoder
↓
scalar score
```

Luong General:

```text
Encoder
↓ W_a

Decoder
×
transformed encoder
↓
scalar score
```

Then all continue:

```text
Scores
↓
Softmax
↓
Weights
↓
Weighted Context
```

---

# 🧠 Final Mental Model

The easiest way to avoid Luong mistakes is to remember:

```text
Do dimensions already match?
        │
   ┌────┴────┐
   │         │
  Yes        No
   │         │
   ▼         ▼
  Dot      General
s_t^T h_i  s_t^T W_a h_i
   │         │
   └────┬────┘
        ↓
      Scores
        ↓
     Softmax
        ↓
     Weights
        ↓
     Context
        ↓
Combine with decoder state
```

Shortest memory version:

# **Transform if needed → Multiply → Normalize → Aggregate → Combine**

---

# 🔗 Next Topic

We now understand how classical attention computes source-target compatibility using:

```text
Bahdanau
→ Additive Scoring

Luong
→ Multiplicative Scoring
```

The next question is:

> **Once the model has learned all these attention weights, what do they look like across a complete source-target sequence, and what can we safely learn from them?**

That takes us to:

# **05_Attention_Alignment_And_Visualization**
