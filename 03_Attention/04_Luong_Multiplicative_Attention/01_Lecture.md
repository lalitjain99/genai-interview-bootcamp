# 📘 Luong / Multiplicative Attention

## 🌟 Introduction

In the previous chapter, we studied **Bahdanau / Additive Attention**.

Its scoring function was:

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

Bahdanau gave us a flexible learned nonlinear scorer.

But it also introduced:

* multiple learned projections;
* a nonlinear activation;
* a learned readout vector;
* additional computation at every source-decoder comparison.

That naturally leads to a new engineering question:

> **Do we really need a small neural network to compute every attention score?**

What if compatibility could be computed more directly using multiplication?

That is the motivation behind:

# **Luong / Multiplicative Attention**

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to:

* explain why Luong Attention was introduced;
* understand why it is called **multiplicative attention**;
* distinguish:

  * dot attention
  * general attention
  * concat-style scoring
* understand the role of `W_a`;
* derive the Luong score mathematically;
* reason about tensor dimensions;
* compare Bahdanau and Luong attention;
* understand current-state vs previous-state timing differences;
* explain the computational advantages;
* understand global vs local Luong attention;
* connect multiplicative attention to later dot-product attention in Transformers.

---

# 📖 Part 1 — Where We Left Off

Generic attention says:

```text
e_(t,i) = score(decoder_state, encoder_state)
```

Bahdanau implemented this as:

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

So the scoring network does:

```text
Project
↓
Add
↓
Nonlinearity
↓
Project to Scalar
```

It works well.

But the question is:

> Can compatibility be computed with fewer operations?

---

# 🧠 Part 2 — Start with the Simplest Idea

Suppose:

```text
decoder state = s_t
encoder state = h_i
```

If both vectors have the same dimension:

```text
s_t ∈ R^H
h_i ∈ R^H
```

then the simplest possible score is:

```text
e_(t,i) = s_t^T h_i
```

This is just a:

> **dot product**

---

# 📐 Part 3 — What Does a Dot Product Mean?

For two vectors:

```text
a = [a1, a2, ..., aH]

b = [b1, b2, ..., bH]
```

their dot product is:

```text
a^T b
=
a1b1
+
a2b2
+
...
+
aHbH
```

The result is:

```text
scalar
```

Exactly what attention needs.

So:

```text
decoder vector
×
encoder vector
↓
scalar compatibility score
```

This is much simpler than a small neural network.

---

# 🧮 Part 4 — Small Dot-Product Example

Suppose:

```text
s_t = [1, 2]

h_i = [3, 4]
```

Then:

```text
e_(t,i)
=
s_t^T h_i
```

```text
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

That single number represents compatibility.

---

# ⭐ Part 5 — Why Is It Called Multiplicative Attention?

Because compatibility is computed through:

> multiplication between decoder and encoder representations

rather than primarily through additive nonlinear combination.

For example:

```text
s_t^T h_i
```

or:

```text
s_t^T W_a h_i
```

Both are multiplicative interactions.

Hence:

# **Multiplicative Attention**

---

# 📖 Part 6 — Luong Attention

Luong Attention is commonly associated with several scoring functions.

The three frequently discussed forms are:

```text
dot
general
concat
```

But the most important multiplicative variants are:

1. Dot
2. General

Let's study them carefully.

---

# 📐 Part 7 — Dot Scoring

The simplest form is:

```text
e_(t,i)
=
s_t^T h_i
```

Requirement:

```text
dim(s_t) = dim(h_i)
```

because dot product requires matching dimensions.

---

# 🧠 Part 8 — Dot Attention Intuition

Imagine:

```text
s_t
```

represents:

> what the decoder currently needs

and:

```text
h_i
```

represents:

> what source position `i` contains.

Their dot product asks:

> How strongly do these two vectors point in compatible directions?

A higher score means:

> stronger alignment under the learned representation space.

---

# ⚠️ Part 9 — Limitation of Plain Dot Attention

Suppose:

```text
s_t ∈ R^256
```

and:

```text
h_i ∈ R^512
```

Then:

```text
s_t^T h_i
```

is invalid.

So plain dot attention needs:

> equal dimensionality.

This motivates the:

# **General Score**

---

# 📐 Part 10 — General Luong Score

The general form is:

```text
e_(t,i)
=
s_t^T W_a h_i
```

where:

```text
W_a
```

is a learned matrix.

This allows the encoder representation to be transformed before interacting with the decoder state.

---

# 🧠 Part 11 — What Does `W_a` Do?

Suppose:

```text
h_i ∈ R^(H_e)
```

and:

```text
s_t ∈ R^(H_d)
```

Then choose:

```text
W_a ∈ R^(H_d × H_e)
```

so:

```text
W_a h_i ∈ R^(H_d)
```

Now:

```text
s_t^T (W_a h_i)
```

is valid.

So `W_a` acts as:

> a learned bridge between encoder and decoder representation spaces.

---

# 🧮 Part 12 — Shape Example

Suppose:

```text
decoder size = 256
encoder size = 512
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

Now:

```text
s_t^T W_a h_i
```

becomes:

```text
(1 × 256)
×
(256)
```

which results in:

```text
scalar
```

So:

```text
e_(t,i) ∈ R
```

---

# ⭐ Part 13 — General vs Bahdanau Projection

This is an important distinction.

Bahdanau:

```text
W_s s
+
W_h h_i
```

projects both sides separately into an attention space.

Luong General:

```text
s_t^T W_a h_i
```

typically transforms one side and then performs a multiplicative interaction.

So:

```text
Bahdanau
→ project + add + nonlinear scoring

Luong General
→ learned transformation + multiplication
```

---

# 📐 Part 14 — Complete Luong Attention Pipeline

Once scores are computed, the rest is generic attention.

### Step 1 — Score

For dot:

```text
e_(t,i) = s_t^T h_i
```

or general:

```text
e_(t,i) = s_t^T W_a h_i
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

# 🧠 Part 15 — One Score Per Source Position

For:

```text
h_1, h_2, ..., h_T
```

we compute:

```text
e_(t,1)
e_(t,2)
...
e_(t,T)
```

Then:

```text
e_t ∈ R^T
```

Softmax gives:

```text
α_t ∈ R^T
```

Then:

```text
c_t ∈ R^(H_e)
```

assuming the original encoder states are used as values.

---

# 📖 Part 16 — Example Story

Suppose the source is:

```text
The payment is due on Friday
```

The decoder is generating the corresponding target-side word for:

```text
Friday
```

The decoder state:

```text
s_t
```

encodes its current generation context.

Luong attention compares `s_t` against:

```text
h_1, h_2, ..., h_T
```

using either:

```text
s_t^T h_i
```

or:

```text
s_t^T W_a h_i
```

If the state corresponding to:

```text
Friday
```

is most aligned with the decoder state, it receives the highest score.

---

# 🧠 Part 17 — Why Multiplicative Scoring Is Efficient

Multiplicative scoring can be implemented efficiently using:

> matrix multiplication

Suppose encoder states are stacked into:

```text
H_enc
```

Then many source scores can be computed together.

For dot attention:

```text
scores
=
H_enc s_t
```

up to orientation convention.

This is much more hardware-friendly than repeatedly running a small nonlinear network for every source position.

---

# ⭐ Part 18 — Matrix View

Suppose:

```text
H_enc ∈ R^(T × H)
```

and:

```text
s_t ∈ R^H
```

Then:

```text
H_enc s_t
```

gives:

```text
R^T
```

That means:

> all source-position scores in one matrix-vector operation.

This is a major implementation advantage.

---

# 🧠 Part 19 — Compare with Bahdanau Computation

Bahdanau needs something like:

```text
tanh(
W_s s
+
W_h h_i
)
```

for every source position.

Luong dot can often do:

```text
H_enc s_t
```

directly.

So multiplicative attention is:

* simpler;
* easier to batch;
* efficient on matrix hardware.

---

# ⚠️ Part 20 — Simpler Does Not Mean Always Better

Dot-product scoring is cheaper.

But Bahdanau has:

* nonlinear compatibility modeling;
* separate learned projections;
* more representational flexibility.

So there is a trade-off:

```text
Bahdanau
→ richer scoring

Luong
→ simpler / more efficient scoring
```

Neither is automatically superior in every setting.

---

# 📖 Part 21 — Decoder State Timing Difference

A commonly taught distinction is:

Bahdanau often uses:

```text
s_(t-1)
```

when computing attention for current output.

Luong often uses:

```text
s_t
```

the current decoder state.

This is a useful historical distinction.

---

# 🧠 Part 22 — Luong Timing Intuition

Conceptually:

```text
previous target token
+
previous decoder state
↓
compute current decoder state s_t
↓
use s_t to attend to encoder states
↓
build c_t
↓
combine s_t and c_t
↓
predict output
```

So Luong attention is often described as:

> post-decoder-state attention

---

# ⚠️ Part 23 — Do Not Turn This into a Universal Rule

Notation varies across papers and implementations.

So the safer conceptual statement is:

> Luong-style attention commonly scores encoder states using the current decoder hidden state, while Bahdanau-style attention is commonly described using the previous decoder state during alignment computation.

Do not say:

> Luong always uses current state and Bahdanau always uses previous state

as an absolute architectural law.

---

# ⭐ Part 24 — Attentional Hidden State

Luong introduced an important combination step.

After computing:

```text
c_t
```

and having:

```text
s_t
```

the model can concatenate them:

```text
[c_t ; s_t]
```

Then apply a learned transformation:

```text
h̃_t
=
tanh(
W_c [c_t ; s_t]
)
```

This creates an:

> **attentional hidden state**

---

# 🧠 Part 25 — Why Combine `c_t` and `s_t`?

Because:

```text
s_t
```

contains target-side decoding information.

And:

```text
c_t
```

contains source-side information retrieved by attention.

So:

```text
[c_t ; s_t]
```

combines:

```text
what I know from target history
+
what I retrieved from source
```

Then the model uses that combined representation for prediction.

---

# 📐 Part 26 — Attentional Hidden State Equation

A common form is:

```text
h̃_t
=
tanh(
W_c [c_t ; s_t]
)
```

Then output probabilities can be produced from:

```text
h̃_t
```

rather than directly from `s_t`.

---

# 🧠 Part 27 — Important Role Separation

Keep these roles distinct:

```text
s_t
→ current decoder state

e_(t,i)
→ compatibility score

α_(t,i)
→ normalized attention weight

c_t
→ weighted source context

h̃_t
→ combined source + decoder representation
```

---

# 📖 Part 28 — Luong Global Attention

Luong also described:

# **Global Attention**

The idea is:

> compare the decoder state against all encoder states.

So:

```text
h_1, h_2, ..., h_T
```

all participate in scoring.

This is essentially the full-source attention mechanism we have been discussing.

---

# 🧠 Part 29 — Global Attention Pipeline

```text
Current Decoder State
↓
Compare with every encoder state
↓
T scores
↓
Softmax
↓
T attention weights
↓
Weighted sum
↓
Context
```

Conceptual cost across decoding:

```text
T_src × T_tgt
```

source-target comparisons.

---

# 📖 Part 30 — Why Consider Local Attention?

If source sequences become long, attending over every source position at every target timestep becomes expensive.

So another idea is:

> instead of looking everywhere, predict a smaller source region to inspect.

This leads to:

# **Local Attention**

---

# 🧠 Part 31 — Local Attention Intuition

Global:

```text
look at every source position
```

Local:

```text
predict where to look
↓
inspect only nearby source positions
```

This attempts to reduce computation while preserving useful alignment.

---

# 📐 Part 32 — Local Window Idea

Suppose the model predicts a source position:

```text
p_t
```

Then attention may focus around:

```text
[p_t - D, ..., p_t + D]
```

instead of:

```text
[1, ..., T]
```

This creates a smaller attention window.

---

# ⭐ Part 33 — Global vs Local Attention

### Global

Advantages:

* full source access;
* simple conceptual model.

Disadvantages:

* more computation for long source sequences.

### Local

Advantages:

* reduced attention search region;
* potentially faster.

Disadvantages:

* may miss relevant information outside the selected window;
* introduces position/window prediction complexity.

---

# 🧠 Part 34 — Luong Scoring Variants

A common summary:

### Dot

```text
score(s_t, h_i)
=
s_t^T h_i
```

### General

```text
score(s_t, h_i)
=
s_t^T W_a h_i
```

### Concat

Often written in a neural form similar to:

```text
score(s_t, h_i)
=
v_a^T tanh(
W_a [s_t ; h_i]
)
```

The important Luong multiplicative forms for our progression are:

> dot and general.

---

# ⚠️ Part 35 — Concat Terminology Can Be Confusing

The concat scoring form uses a nonlinear neural network.

So although it is discussed in the Luong paper's scoring alternatives, it is not the core reason we call Luong-style attention:

> multiplicative attention.

For our learning progression, focus on:

```text
dot
general
```

as the multiplicative forms.

---

# 🧮 Part 36 — Worked General-Score Example

Suppose:

```text
s_t =
[1
 2]
```

```text
h_i =
[3
 4
 5]
```

and:

```text
W_a =
[1 0 0
 0 1 0]
```

First:

```text
W_a h_i
=
[3
 4]
```

Then:

```text
s_t^T W_a h_i
```

```text
=
[1, 2]
·
[3, 4]
```

```text
=
1×3 + 2×4
```

```text
=
11
```

So:

```text
e_(t,i) = 11
```

---

# 🧠 Part 37 — What Does `W_a` Learn?

`W_a` learns:

> how encoder features should be transformed so that multiplication with the decoder state gives a useful compatibility score.

Unlike Bahdanau, we are not building an intermediate nonlinear attention hidden vector.

Instead:

```text
encoder representation
↓ learned transformation
decoder representation
×
↓
scalar
```

---

# 📐 Part 38 — Parameter Count Intuition

For general attention:

```text
W_a ∈ R^(H_d × H_e)
```

So parameter count is approximately:

```text
H_d × H_e
```

Dot attention:

```text
e = s_t^T h_i
```

may require:

> no additional scoring parameters

if dimensions already match.

This is extremely simple.

---

# ⭐ Part 39 — Dot Attention Has Almost No Scoring Parameters

This is an important difference.

Bahdanau needs:

```text
W_s
W_h
v_a
b_a
```

Luong dot scoring needs:

```text
none
```

for the compatibility function itself.

The representations are already learned by the surrounding network.

---

# 🧠 Part 40 — But Representation Quality Matters More

Dot attention assumes:

> encoder and decoder states already live in spaces where dot-product similarity is useful.

If they do not, dot product may be too restrictive.

General attention relaxes this by learning:

```text
W_a
```

---

# 📖 Part 41 — Matrix Computation for All Scores

Suppose:

```text
H_enc ∈ R^(T × H)
```

and:

```text
s_t ∈ R^H
```

Then dot scores:

```text
e_t
=
H_enc s_t
```

produce:

```text
e_t ∈ R^T
```

All source positions are scored at once.

This is one reason multiplicative attention maps naturally to efficient linear algebra.

---

# 🔗 Part 42 — Relation to Transformer Attention

Do not jump ahead too far, but notice the conceptual direction:

```text
Luong dot attention
↓
compatibility through dot products
↓
later
scaled dot-product attention
```

Transformers will eventually formalize this idea using:

```text
Query
Key
Value
```

But classical Luong attention is still:

> recurrent encoder-decoder attention.

---

# ⚠️ Part 43 — Luong Attention Is NOT Transformer Self-Attention

Very important.

Luong Attention:

```text
decoder state
attends to
encoder states
```

Transformer self-attention:

```text
positions within a sequence
attend to
other positions
```

Do not confuse them just because both may use dot products.

---

# 🧠 Part 44 — Source vs Target Roles

In Luong encoder-decoder attention:

```text
decoder state
→ asks the question

encoder states
→ candidate source information
```

Conceptually:

```text
decoder need
×
source representation
↓
compatibility
```

Later, this intuition will become useful when understanding:

```text
Query
Key
```

but we are not introducing QKV formally yet.

---

# 🚨 Part 45 — High-Yield Mistake: Dot Product Means Cosine Similarity

Not necessarily.

Dot product:

```text
s^T h
```

depends on:

* direction
* magnitude

Cosine similarity explicitly normalizes magnitudes.

So:

```text
dot product ≠ cosine similarity
```

unless vectors are normalized appropriately.

---

# 🧠 Part 46 — Why Large Magnitudes Matter

Suppose two pairs have similar directions but one has larger vector magnitudes.

The dot product can become much larger.

This will become important later when we study:

> scaled dot-product attention.

For now, remember:

```text
dot-product score magnitude
depends on vector scale
```

---

# ⭐ Part 47 — Bahdanau vs Luong

| Aspect                          | Bahdanau               | Luong                   |
| ------------------------------- | ---------------------- | ----------------------- |
| Core interaction                | Additive               | Multiplicative          |
| Typical state                   | Often `s_(t-1)`        | Often `s_t`             |
| Nonlinearity in score           | Yes                    | Not for dot/general     |
| Separate attention hidden space | Yes                    | Not necessarily         |
| Extra score parameters          | More                   | Few / none              |
| Matrix efficiency               | Lower                  | Higher                  |
| Dimension mismatch              | Handled by projections | General form handles it |
| Recurrent architecture          | Yes                    | Yes                     |

---

# 🧠 Part 48 — Do Not Oversimplify the Comparison

Avoid saying:

> Bahdanau is accurate, Luong is fast.

That is too simplistic.

A better statement:

> Bahdanau uses a learned nonlinear additive compatibility network, while Luong's dot/general variants use simpler multiplicative interactions that are typically more efficient to compute.

Actual quality depends on:

* architecture;
* dimensions;
* training;
* data;
* implementation.

---

# 📐 Part 49 — Complete Global Luong Flow

```text
Source
↓
Encoder
↓
h_1 ... h_T

Target History
↓
Decoder
↓
s_t

s_t × h_i
or
s_t^T W_a h_i
↓
Scores
↓
Softmax
↓
α_t
↓
Weighted Sum
↓
c_t

[c_t ; s_t]
↓
W_c + tanh
↓
h̃_t
↓
Output Prediction
```

---

# 🧠 Part 50 — What Does Luong Change?

Generic attention already had:

```text
score
↓
softmax
↓
weighted context
```

Luong changes mainly:

> the scoring function and the integration timing around the decoder state.

It does not change the fundamental idea of dynamic source retrieval.

---

# ⚠️ Part 51 — Limitations Remain

Luong attention still has several classical encoder-decoder limitations:

### 1. Recurrent decoding

Target generation is sequential.

### 2. Often recurrent encoding

Source computation may also be sequential.

### 3. Global attention scans the source

Each target step interacts with many source positions.

### 4. Long-range computation remains costly

Source-target interactions grow with sequence lengths.

### 5. Dot-product behavior depends on representation scale

This later becomes relevant for scaled attention.

---

# 🔗 Part 52 — What Have We Learned So Far?

Evolution:

```text
Fixed Context
↓
Generic Attention
↓
Need a score function
↓
Bahdanau
↓
Nonlinear additive scoring
↓
Can scoring be simpler?
↓
Luong
↓
Multiplicative scoring
```

Now we have two major classical attention families.

---

# 🧠 Part 53 — The Bigger Question

Attention solved:

> fixed source access.

But the architecture still looks like:

```text
RNN Encoder
+
Attention
+
RNN Decoder
```

So a much larger question remains:

> **If attention is so powerful, do we still need recurrence at all?**

Before answering that, we should first understand:

* how alignments behave;
* what attention visualizations mean;
* where classical attention still fails.

That prepares us naturally for Transformers.

---

# 🎤 30-Second Interview Answer

> **Luong Attention is a classical encoder-decoder attention mechanism that computes compatibility using multiplicative interactions. In the dot variant, the score is `s_t^T h_i`, while the general variant uses `s_t^T W_a h_i`. These scores are softmax-normalized across source positions to create attention weights and a weighted context vector. Compared with Bahdanau's nonlinear additive scorer, Luong's multiplicative forms are simpler and more matrix-multiplication friendly.**

---

# 🎤 Dot vs General

> **Luong dot attention directly computes `s_t^T h_i` and therefore requires compatible dimensions. General attention inserts a learned matrix, `s_t^T W_a h_i`, allowing the model to learn how encoder representations should be transformed before comparison.**

---

# 🎤 Bahdanau vs Luong

> **Bahdanau computes compatibility with a learned nonlinear additive network, while Luong dot/general attention uses multiplicative interactions. Bahdanau introduces a separate attention hidden space and more parameters, whereas Luong is generally simpler and more efficient for matrix computation.**

---

# 🚫 High-Yield Traps

### ❌ Luong Attention is Self-Attention

No.

It is classical encoder-decoder attention.

---

### ❌ Dot product can always be used regardless of dimensions

No.

Dimensions must match.

---

### ❌ `W_a` is an attention weight matrix over source tokens

No.

It is a learned parameter in the scoring function.

---

### ❌ Dot product is the same as cosine similarity

No.

Dot product is also affected by magnitude.

---

### ❌ Luong Attention removes recurrence

No.

The classical architecture still uses recurrent networks.

---

### ❌ General attention and Bahdanau are the same because both use matrices

No.

Their interaction structures are fundamentally different.

---

### ❌ Local attention always gives the same result as global attention

No.

Local attention restricts the search region and may miss information.

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

1. Luong is commonly called **Multiplicative Attention**.
2. Dot scoring is `s_t^T h_i`.
3. General scoring is `s_t^T W_a h_i`.
4. Dot attention typically requires matching dimensions.
5. General attention can bridge representation dimensions.
6. Scores are softmax-normalized across source positions.
7. The context is a weighted sum of encoder states.
8. Luong commonly uses the current decoder state.
9. Multiplicative scoring is matrix-multiplication friendly.
10. Luong is still classical recurrent encoder-decoder attention, not Transformer self-attention.

---

# 🧠 Final Mental Model

Bahdanau asks:

```text
Can a small neural network learn compatibility?
```

Luong asks:

```text
Can compatibility be measured more directly
through multiplication?
```

Core Luong pipeline:

```text
Decoder State
        ×
Encoder State
        ↓
Compatibility Score
        ↓
Softmax
        ↓
Attention Weight
        ↓
Weighted Source Context
```

For general scoring:

```text
Encoder State
↓ W_a
Transformed Source Representation
        ×
Decoder State
        ↓
Scalar Score
```

Shortest memory version:

# **Transform if needed → Multiply → Normalize → Aggregate**

---

# 🔗 Next Lecture

Now that we understand both major classical scoring families:

```text
Bahdanau
+
Luong
```

the next useful question is:

> **What do the learned attention distributions actually look like, and how should we interpret them?**

That leads to:

# **05_Attention_Alignment_And_Visualization**
