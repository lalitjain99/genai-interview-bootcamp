# 📘 Lecture — Attention Mechanism

## 🌟 Introduction

In the previous chapter, **Attention Motivation**, we answered:

> **Why was attention needed?**

We discovered that classic Seq2Seq had a structural limitation:

```text
Source Sequence
↓
Encoder
↓
One Fixed Context Vector
↓
Decoder
```

The decoder had to generate every target token from the same compressed summary.

Attention changed the idea completely:

```text
Keep all encoder states
↓
At every decoder step,
retrieve the source information
that matters right now
```

We introduced the key equation:

# `c_t = Σ_i α_(t,i) h_i`

where:

* `h_i` = encoder state at source position `i`
* `α_(t,i)` = importance of source position `i` for decoder step `t`
* `c_t` = dynamic context vector

But we intentionally left one major question unanswered:

# **Where do the attention weights `α_(t,i)` come from?**

That is the purpose of this lecture.

We will build the attention mechanism from first principles:

```text
Decoder Need
+
Encoder State
↓
Compatibility Score
↓
Normalize Scores
↓
Attention Weights
↓
Weighted Sum
↓
Dynamic Context
```

This is the core mechanism behind classical encoder-decoder attention.

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to:

## 🧠 Conceptual Understanding

* Explain the complete attention pipeline.
* Explain what an attention score represents.
* Understand why every encoder state is compared with the decoder state.
* Explain the difference between:

  * attention scores
  * attention weights
  * context vector
* Explain why softmax is used.
* Understand how the decoder uses dynamic context.

## 📐 Mathematical Understanding

Understand:

# `e_(t,i) = score(s_(t-1), h_i)`

# `α_(t,i) = exp(e_(t,i)) / Σ_j exp(e_(t,j))`

# `c_t = Σ_i α_(t,i)h_i`

and how these equations form one pipeline.

## 🧮 Practical Understanding

* Calculate softmax attention weights.
* Compute a weighted context vector.
* Trace one decoder timestep.
* Understand tensor dimensions conceptually.
* Interpret attention distributions.

## 🔗 Architectural Understanding

Prepare for:

* Bahdanau / Additive Attention
* Luong / Multiplicative Attention
* later Self-Attention

without jumping ahead prematurely.

---

# 📖 Part 1 — Start With the Problem

Suppose the encoder processes:

```text
I love machine learning
```

and produces:

```text
h1 → representation of "I"
h2 → representation of "love"
h3 → representation of "machine"
h4 → representation of "learning"
```

Now the decoder is about to generate its next token.

The question is:

> Which of `h1, h2, h3, h4` should matter most right now?

Attention needs a way to answer that.

---

# 🧠 Part 2 — What Information Does the Decoder Have?

The decoder has a state representing:

* what it has generated so far
* what source information it has already used
* what it may need next

Let us denote the decoder state used for attention as:

# `s_(t-1)`

Why previous decoder state?

Because before predicting target token `t`, the decoder has already processed:

```text
y1, y2, ..., y_(t-1)
```

So `s_(t-1)` summarizes:

> the decoder's current need.

---

# ⭐ Part 3 — First Question of Attention

For every source position `i`, ask:

# **How compatible is encoder state `h_i` with the decoder's current need `s_(t-1)`?**

Conceptually:

```text
s_(t-1) + h1 → relevance score
s_(t-1) + h2 → relevance score
s_(t-1) + h3 → relevance score
...
```

This gives us:

# **attention scores**

---

# 📐 Part 4 — Attention Score Equation

We define:

# `e_(t,i) = score(s_(t-1), h_i)`

where:

* `t` = decoder timestep
* `i` = source position
* `s_(t-1)` = decoder state
* `h_i` = encoder state
* `e_(t,i)` = raw compatibility score

The exact form of:

# `score(...)`

is not fixed.

Different attention mechanisms define it differently.

That will lead later to:

* Bahdanau attention
* Luong attention

For now, focus on the role of the score.

---

# 🧠 Part 5 — What Does `e_(t,i)` Mean?

If:

```text
e_(t,1) = 0.5
e_(t,2) = 1.2
e_(t,3) = 4.7
e_(t,4) = 0.9
```

then source position 3 appears:

> much more compatible with the decoder's current need.

But these numbers are only:

# **raw scores**

They are not yet attention weights.

---

# 🚨 Part 6 — Attention Score ≠ Attention Weight

This distinction is extremely important.

### Attention Score

# `e_(t,i)`

Raw compatibility value.

May be:

* negative
* positive
* unbounded
* not normalized

### Attention Weight

# `α_(t,i)`

Normalized relevance value used in the weighted sum.

So:

```text
Raw Score e
↓
Softmax
↓
Attention Weight α
```

---

# ⭐ Part 7 — Why Can't We Directly Use Raw Scores?

Suppose:

```text
e = [2, 5, -1]
```

Can we use:

```text
2h1 + 5h2 - 1h3
```

as our context?

Mathematically, we could define such a mechanism.

But it would be harder to interpret and control.

We usually want relevance weights that:

* are non-negative
* have a meaningful relative scale
* sum to 1

That gives us:

> a normalized distribution over source positions.

Softmax provides exactly that.

---

# 📐 Part 8 — Attention Softmax

For decoder step `t`:

# `α_(t,i) = exp(e_(t,i)) / Σ_j exp(e_(t,j))`

The denominator sums across:

> all source positions `j`

for the current decoder step.

So:

```text
e_(t,1)
e_(t,2)
...
e_(t,T)
↓
Softmax across source positions
↓
α_(t,1)
α_(t,2)
...
α_(t,T)
```

---

# 🧠 Part 9 — Softmax Direction Matters

Suppose source length is 4.

For decoder step `t`, we have:

```text
[e_(t,1), e_(t,2), e_(t,3), e_(t,4)]
```

Softmax is applied across:

# **source positions**

not across decoder timesteps.

Therefore:

# `Σ_i α_(t,i) = 1`

for a fixed decoder step `t`.

---

# 🧮 Part 10 — Worked Softmax Example

Suppose scores are:

```text
e = [1, 2, 3]
```

Compute exponentials approximately:

```text
exp(1) ≈ 2.72
exp(2) ≈ 7.39
exp(3) ≈ 20.09
```

Total:

```text
2.72 + 7.39 + 20.09
= 30.20
```

Weights:

```text
α1 ≈ 2.72 / 30.20 = 0.09

α2 ≈ 7.39 / 30.20 = 0.24

α3 ≈ 20.09 / 30.20 = 0.67
```

So:

# `α ≈ [0.09, 0.24, 0.67]`

The third source position receives:

> the strongest attention.

---

# ⭐ Part 11 — Softmax Preserves Relative Preference

Notice:

```text
score 3 > score 2 > score 1
```

After softmax:

```text
0.67 > 0.24 > 0.09
```

So softmax preserves ranking while creating:

> normalized relative importance.

---

# 🧠 Part 12 — Now We Have Attention Weights

We now know:

```text
Decoder State
+
Each Encoder State
↓
Raw Scores e_(t,i)
↓
Softmax
↓
α_(t,i)
```

But we still need to answer:

> What do we do with these weights?

We use them to construct:

# **the context vector**

---

# 📐 Part 13 — Dynamic Context Vector

The context at decoder step `t` is:

# `c_t = Σ_i α_(t,i) h_i`

Expanded:

```text
c_t =
α_(t,1)h1
+
α_(t,2)h2
+
...
+
α_(t,T)hT
```

This creates a:

> weighted source summary customized for the current decoder step.

---

# 🧮 Part 14 — Worked Context Example

Suppose:

```text
h1 = [1, 0]
h2 = [0, 1]
h3 = [1, 1]
```

and:

```text
α = [0.2, 0.3, 0.5]
```

Then:

```text
c_t
=
0.2[1,0]
+
0.3[0,1]
+
0.5[1,1]
```

Calculate:

```text
= [0.2, 0]
+ [0, 0.3]
+ [0.5, 0.5]
```

So:

# `c_t = [0.7, 0.8]`

This is the source context used at that decoder step.

---

# 🧠 Part 15 — Why Weighted Sum?

Why not simply concatenate all encoder states?

Suppose source length changes:

```text
5 tokens
10 tokens
100 tokens
```

Concatenation would create:

> a representation whose dimensionality changes with source length.

But the decoder expects:

> a fixed-dimensional input.

A weighted sum gives:

# `c_t ∈ R^H`

if:

# `h_i ∈ R^H`

regardless of source length.

---

# ⭐ Part 16 — Variable Number of States → Fixed Context

This is elegant.

Input:

```text
h1, h2, ..., hT
```

can contain any number `T` of states.

But weighted sum produces:

```text
c_t
```

with the same dimensionality as an encoder hidden state.

So:

```text
Variable-Length Source Representation Bank
↓
Attention
↓
Fixed-Dimensional Context for Current Step
```

---

# 🧠 Part 17 — Attention Does Not Store New Information

Attention does not magically invent source information.

The context vector contains information derived from:

# `h_1, ..., h_T`

Therefore attention quality depends strongly on:

> encoder representation quality.

If the encoder never captured an important fact:

> attention cannot retrieve it.

This is an important system-level distinction.

---

# 🔄 Part 18 — Complete Attention Pipeline

At target timestep `t`:

### Step 1

Take decoder state:

# `s_(t-1)`

### Step 2

Compare it with every encoder state:

# `e_(t,i) = score(s_(t-1), h_i)`

### Step 3

Normalize:

# `α_(t,i) = softmax(e_(t,i))`

### Step 4

Create context:

# `c_t = Σ_i α_(t,i)h_i`

### Step 5

Use:

* decoder information
* `c_t`

to produce target prediction.

The entire mechanism is:

```text
Decoder State
+
Encoder States
↓
Scores
↓
Softmax
↓
Weights
↓
Weighted Sum
↓
Context
↓
Prediction
```

---

# ⭐ Part 19 — A Full Example From Words to Context

Source:

```text
I love machine learning
```

Encoder:

```text
h1 → I
h2 → love
h3 → machine
h4 → learning
```

Suppose decoder currently needs to generate something corresponding to:

`learning`.

Compatibility scores:

```text
h1 → 0.2
h2 → 0.4
h3 → 1.5
h4 → 3.0
```

After softmax, perhaps:

```text
α =
[0.04, 0.05, 0.18, 0.73]
```

Context:

```text
c_t =
0.04h1
+
0.05h2
+
0.18h3
+
0.73h4
```

So the decoder receives:

> a context strongly dominated by the source representation for "learning".

---

# 🧠 Part 20 — Attention Is Query-Dependent

Suppose the next decoder step changes.

Now the decoder needs information corresponding more closely to:

`love`.

The decoder state changes.

Therefore:

```text
score(s_t, h_i)
```

changes.

The new attention distribution may become:

```text
[0.05, 0.75, 0.10, 0.10]
```

So:

# **same encoder states + different decoder state = different attention**

That is why attention is dynamic.

---

# ⭐ Part 21 — The Encoder States Stay the Same

During decoding of one source sequence:

```text
h1, h2, ..., hT
```

are usually computed once by the encoder.

Then at each decoder step:

> attention weights change.

So:

```text
Encoder States
→ mostly fixed during decoding

Attention Distribution
→ changes per decoder step
```

This distinction matters.

---

# 📐 Part 22 — Shape Intuition

Suppose:

* source length = `T`
* encoder hidden size = `H`

Then:

# `h_i ∈ R^H`

Stack all encoder states:

# `H_enc ∈ R^(T × H)`

Attention scores for one decoder step:

# `e_t ∈ R^T`

Attention weights:

# `α_t ∈ R^T`

Context:

# `c_t ∈ R^H`

So:

```text
T source states
↓
T scores
↓
T weights
↓
1 weighted H-dimensional context
```

---

# 🧠 Part 23 — Why Scores Are Scalars

For each source state we want a simple answer:

> how relevant is this state?

So the compatibility function typically produces:

# one scalar per source position

not another large hidden representation.

Thus:

```text
decoder state + h_i
↓
e_(t,i) ∈ R
```

---

# ⭐ Part 24 — Score Function Is the Design Choice

So far we have left:

# `score(s_(t-1), h_i)`

undefined.

This is intentional.

There are multiple ways to compute compatibility.

Examples include:

```text
small neural network
```

or:

```text
dot product
```

or:

```text
learned bilinear transformation
```

These different scoring choices create different forms of attention.

---

# 🧠 Part 25 — Generic vs Specific Attention

The generic mechanism is always:

```text
score
↓
normalize
↓
weighted sum
```

What changes is mainly:

> how the score is computed.

This is why we should understand the generic mechanism before memorizing Bahdanau or Luong.

---

# ⭐ Part 26 — The Three Fundamental Operations

Attention can be reduced to three conceptual operations:

## 1. Compare

```text
Decoder Need
vs
Source Representation
```

## 2. Normalize

```text
Which source positions matter relatively?
```

## 3. Aggregate

```text
Build weighted source context
```

So:

# **Compare → Normalize → Aggregate**

This is one of the most useful mental models.

---

# 🧠 Part 27 — Compare

For each source state:

# `e_(t,i) = score(s_(t-1), h_i)`

Question:

> How relevant is source position `i` right now?

---

# 🧠 Part 28 — Normalize

Across all source positions:

# `α_t = softmax(e_t)`

Question:

> How should attention be distributed across the source?

---

# 🧠 Part 29 — Aggregate

Then:

# `c_t = Σ_i α_(t,i)h_i`

Question:

> What source information should the decoder receive?

---

# ⭐ Part 30 — Score vs Weight vs Context

You must know this distinction cold.

| Quantity  | Meaning                       |
| --------- | ----------------------------- |
| `e_(t,i)` | raw compatibility score       |
| `α_(t,i)` | normalized attention weight   |
| `h_i`     | encoder/source representation |
| `c_t`     | weighted source context       |

Pipeline:

```text
e
↓ softmax
α
↓ weighted sum with h
c
```

---

# 🚨 Part 31 — Common Confusion: `α` Does Not Contain Source Meaning

An attention weight like:

```text
0.8
```

does not contain the representation of:

> "John"

It only says:

> how strongly the corresponding `h_i` should contribute.

The semantic information resides in:

# `h_i`

The relevance is encoded by:

# `α_(t,i)`

---

# 🧠 Part 32 — Another Analogy: Search Engine

Imagine:

```text
Decoder State
→ search query

Encoder States
→ documents

Attention Score
→ relevance score

Softmax
→ normalized ranking weights

Context
→ weighted retrieved information
```

This analogy is not technically exact, but it captures:

> dynamic relevance-based access.

---

# ⭐ Part 33 — Attention Is Content-Based Access

The decoder does not simply say:

> always use source position 4.

Instead it bases access on:

> current decoder state + encoder content.

So classical attention is often understood as:

# **content-based addressing**

This is a precursor to later Transformer intuitions.

---

# 🧠 Part 34 — Why Attention Can Learn Alignment

The model is trained on the final prediction objective.

Suppose focusing on:

`h3`

helps predict the correct target token.

Then gradient descent adjusts:

* score-function parameters
* encoder representations
* decoder representations

so that future attention can assign:

> greater relevance to useful source states.

Thus alignment can emerge:

> without explicit word-to-word alignment labels.

---

# 📐 Part 35 — Differentiability

The pipeline:

```text
score
↓
softmax
↓
weighted sum
```

is differentiable.

Therefore loss gradients can flow into:

* attention scoring parameters
* decoder state
* encoder states

This lets the entire system learn:

> end to end.

---

# 🔄 Part 36 — Gradient Flow Through Attention

Suppose target prediction is wrong.

Loss gradient can flow:

```text
Loss
↓
Decoder Output
↓
Context c_t
↓
Attention Weights α
↓
Attention Scores e
↓
Encoder States + Decoder State
```

So attention is not an external heuristic.

It is:

# **part of the trainable neural network**

---

# ⭐ Part 37 — Direct Gradient Paths to Encoder States

Because:

# `c_t = Σ_i α_(t,i)h_i`

the target loss can directly influence:

> each encoder state that contributed to the context.

This gives a shorter route than forcing every signal through:

> one final encoder state.

Again, attention was not primarily invented as a gradient fix, but this path is beneficial.

---

# 🧠 Part 38 — Do Attention Weights Always Become Sharp?

No.

Sometimes attention may be:

```text
[0.01, 0.02, 0.95, 0.02]
```

very concentrated.

Other times:

```text
[0.20, 0.30, 0.25, 0.25]
```

more distributed.

The correct pattern depends on:

> the task and current prediction.

---

# ⭐ Part 39 — Sharp vs Diffuse Attention

### Sharp Attention

Most weight on a small number of source positions.

Useful when:

> a specific source detail dominates.

### Diffuse Attention

Weight spread across many positions.

Useful when:

> multiple source regions jointly matter.

Neither is automatically:

> better.

---

# 🧠 Part 40 — Attention Score Can Be Negative

Example:

```text
e = [-3, 1, 5]
```

This is valid.

Scores are:

> arbitrary real-valued compatibility values.

After softmax, weights become positive:

```text
α_i > 0
```

So:

# **score may be negative; standard softmax attention weight is non-negative**

---

# 🚨 Part 41 — Softmax Does Not Mean Only One Position Survives

Softmax does not perform:

> argmax.

Example:

```text
scores = [1, 2, 3]
```

may produce:

```text
weights = [0.09, 0.24, 0.67]
```

All positions still contribute.

This is the reason classical attention is:

# **soft**

---

# 🧠 Part 42 — What if Two Source Positions Are Equally Relevant?

Suppose:

```text
e = [1, 4, 4, 1]
```

Then softmax will give roughly equal high weights to:

> positions 2 and 3.

So attention can naturally represent:

> multiple relevant positions.

---

# ⭐ Part 43 — Attention Does Not Necessarily Mean "Copy"

If the model attends strongly to:

`John`

that does not mean the decoder must output:

`John`.

The attended representation can be used to produce:

* translated word
* grammatical transformation
* summary
* related target phrase

Attention means:

> use this source information,

not:

> copy this token exactly.

---

# 🧠 Part 44 — How the Decoder Uses `c_t`

Different architectures combine context and decoder state differently.

A simple conceptual formulation:

```text
Decoder State
+
Context c_t
↓
Output Layer
↓
Token Distribution
```

Another architecture may feed context into the recurrent update itself.

Exact ordering varies.

The invariant is:

> `c_t` influences the target prediction at step `t`.

---

# 📐 Part 45 — Conceptual Output Equation

One simple formulation:

# `o_t = tanh(W_c [s_t ; c_t] + b_c)`

Then:

# `z_t = W_o o_t + b_o`

Then:

# `P(y_t | ...) = softmax(z_t)`

Do not memorize this as the only valid implementation.

It simply shows how:

> decoder state + source context

can jointly influence output.

---

# ⭐ Part 46 — Concatenation Here Is Different From Concatenating All Source States

We may concatenate:

```text
[s_t ; c_t]
```

because both are:

> fixed-dimensional vectors.

This is completely different from concatenating:

```text
[h1 ; h2 ; ... ; hT]
```

which would grow with source length.

---

# 🧠 Part 47 — One Decoder Step End-to-End

At decoder timestep `t`:

```text
Previous Target Token
↓
Decoder State / Current Need
↓
Compare With Every Encoder State
↓
Scores e_(t,i)
↓
Softmax
↓
Weights α_(t,i)
↓
Weighted Sum
↓
Context c_t
↓
Combine Context + Decoder State
↓
Vocabulary Logits
↓
Softmax
↓
Target Prediction
```

Notice:

> there are two different softmax operations possible here.

---

# 🚨 Part 48 — Attention Softmax vs Output Softmax

This is a very important distinction.

## Attention Softmax

Normalizes:

> source relevance scores.

Produces:

# `α_(t,i)`

Distribution over:

> source positions.

## Output Softmax

Normalizes:

> vocabulary logits.

Produces:

# `P(y_t)`

Distribution over:

> target vocabulary.

So:

```text
Attention Softmax
→ WHERE to look

Output Softmax
→ WHAT token to predict
```

Excellent interview distinction.

---

# ⭐ Part 49 — Attention Distribution Size vs Vocabulary Distribution Size

If:

```text
source length = 12
vocabulary size = 50,000
```

then:

### Attention Distribution

has:

```text
12 values
```

one per source position.

### Output Distribution

has:

```text
50,000 values
```

one per vocabulary token.

They are fundamentally different distributions.

---

# 🧠 Part 50 — Masking Padded Source Positions

Suppose batched source is:

```text
A B C <PAD> <PAD>
```

The decoder should not attend to:

`PAD`.

So attention should mask padded positions before softmax.

Conceptually:

```text
PAD score
→ -∞
↓
softmax
↓
weight ≈ 0
```

This is:

# **attention masking**

---

# ⭐ Part 51 — Why Mask Before Softmax?

If PAD positions were allowed normal scores:

> some probability mass could be assigned to meaningless padding.

Masking before softmax ensures:

```text
real source positions
→ share the attention mass

PAD positions
→ effectively 0 weight
```

---

# 🧠 Part 52 — Source Length and Attention Computation

At each decoder step:

> compare with all source states.

If source length is:

`T`

then each decoder step computes approximately:

# `T attention scores`

For target length:

`T'`

total conceptual score computations:

# `T × T'`

This is the computational price of:

> dynamic source access.

---

# 🔗 Part 53 — Connection Back to Beam Search

Suppose beam width is:

`k`.

Each beam has its own decoder history and state.

Therefore each beam may compute:

> its own attention distribution.

So:

```text
Beam 1 decoder state
→ attention distribution 1

Beam 2 decoder state
→ attention distribution 2
```

Different target hypotheses can:

> attend to different source regions.

---

# 🧠 Part 54 — Attention Does Not Choose the Target Token

Attention determines:

> source context.

It does not itself choose:

> the output vocabulary token.

The full flow is:

```text
Attention
→ source context

Decoder + Output Layer
→ target logits

Decoding Strategy
→ selected target token
```

Do not collapse these concepts.

---

# ⭐ Part 55 — Score Function Requirements

A useful score function should let the model learn:

> how well a source representation matches the decoder's current state.

There is no single universal formula.

This creates our next question:

> What score function should we use?

One answer is:

# **a small neural network**

That gives us:

# **Bahdanau / Additive Attention**

---

# 🔗 Part 56 — Why Bahdanau Comes Next

Our generic attention has:

# `e_(t,i) = score(s_(t-1), h_i)`

Bahdanau says:

> let's learn this compatibility using a feedforward neural network.

Conceptually:

```text
decoder state
+
encoder state
↓
learned nonlinear transformation
↓
scalar compatibility score
```

This makes Bahdanau a concrete implementation of:

> the generic score function.

---

# 🔗 Part 57 — Why Luong Comes After Bahdanau

Later Luong-style attention asks:

> Can the score be calculated more simply using multiplication / dot-product-style operations?

That creates another family of scoring functions.

So the progression is:

```text
Generic Attention
↓
How should score(...) work?
↓
Bahdanau: additive neural scoring
↓
Luong: multiplicative scoring
```

---

# 🎤 Part 58 — 30-Second Interview Answer

> **The attention mechanism dynamically constructs a source context for every decoder step. The decoder state is compared with each encoder hidden state to produce raw compatibility scores `e_(t,i)`. These scores are normalized across source positions using softmax to produce attention weights `α_(t,i)`. The context vector is then computed as the weighted sum `c_t = Σ_i α_(t,i)h_i`, and this context is combined with decoder information to predict the next target token. The score function itself can be implemented in different ways, which leads to mechanisms such as Bahdanau additive attention and Luong multiplicative attention.**

---

# 🎤 Part 59 — Score vs Weight Interview Answer

> **An attention score is an unnormalized compatibility value between the decoder state and a source representation. An attention weight is the normalized value obtained after applying softmax across all source scores. The weights are then used to compute the weighted source context.**

---

# 🎤 Part 60 — Why Softmax?

> **Softmax converts arbitrary compatibility scores into a normalized, differentiable distribution over source positions. This makes the values non-negative, sums them to one, preserves relative preference, and allows the context to be constructed as a weighted average of encoder representations.**

---

# 🚫 Part 61 — High-Yield Traps

### ❌ `e_(t,i)` and `α_(t,i)` are the same thing

No.

---

### ❌ Attention scores must be probabilities

No.

---

### ❌ Attention scores cannot be negative

They can.

---

### ❌ Softmax is applied across target vocabulary

Not for attention softmax.

---

### ❌ Attention weight contains the source meaning

No.

`h_i` contains representation; `α` controls contribution.

---

### ❌ Context is one selected encoder state

Usually no.

Weighted sum.

---

### ❌ Softmax performs hard selection

No.

---

### ❌ Attention chooses the final output token

No.

---

### ❌ Encoder states change for every decoder step

Usually the same encoded source states are reused.

---

### ❌ All decoder steps have the same attention weights

No.

---

### ❌ PAD positions should receive normal attention

No.

Mask them.

---

### ❌ Attention and output softmax are the same operation conceptually

No.

Different distributions.

---

# 📐 Formula Flash Card

### Compatibility Score

# `e_(t,i) = score(s_(t-1), h_i)`

### Attention Weight

# `α_(t,i) = exp(e_(t,i)) / Σ_j exp(e_(t,j))`

### Weight Sum

# `Σ_i α_(t,i) = 1`

### Dynamic Context

# `c_t = Σ_i α_(t,i)h_i`

### Conceptual Pipeline

```text
Compare
↓
Normalize
↓
Aggregate
```

or:

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

# 🧠 Part 62 — 12 Things You Must Know Cold

1. **Attention compares decoder need with every encoder state.**
2. **`e_(t,i)` is a raw compatibility score.**
3. **Scores are not probabilities.**
4. **Softmax converts scores into `α_(t,i)`.**
5. **Softmax is applied across source positions.**
6. **Attention weights usually sum to 1.**
7. **`α` is a scalar weight; `h_i` is a vector.**
8. **`c_t` is a weighted sum of encoder states.**
9. **Different decoder steps can have different attention distributions.**
10. **Attention softmax differs from vocabulary softmax.**
11. **PAD source positions should be masked.**
12. **Bahdanau and Luong mainly differ in how compatibility scores are computed.**

---

# 🧠 Final Mental Model

The complete attention mechanism is:

```text
                DECODER STATE
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
         h1         h2         h3 ... hT
          │          │          │
          ▼          ▼          ▼
       score       score      score
          │          │          │
          └──────────┼──────────┘
                     ▼
               Raw Scores e_t
                     │
                     ▼
                  Softmax
                     │
                     ▼
              Attention α_t
                     │
          ┌──────────┼──────────┐
          │          │          │
        α1h1       α2h2       α3h3 ...
          │          │          │
          └──────────┼──────────┘
                     ▼
               Context c_t
                     │
                     ▼
            Decoder Prediction
```

In three words:

# **Compare → Normalize → Aggregate**

Or in equations:

# `e_(t,i) = score(s_(t-1), h_i)`

↓

# `α_(t,i) = softmax(e_(t,i))`

↓

# `c_t = Σ_i α_(t,i)h_i`

---

# 🚀 Where We Go Next

We now understand the **generic attention mechanism**.

But one part is still abstract:

# `score(s_(t-1), h_i)`

How exactly should we compute that score?

The first major historical answer was:

> pass the encoder state and decoder state through a small learned neural network.

That gives us:

# `03_Bahdanau_Additive_Attention`

The progression is now:

```text
Attention Motivation
↓
Why dynamic context?
↓
Attention Mechanism
↓
Compare → Normalize → Aggregate
↓
Need a concrete scoring function
↓
Bahdanau Additive Attention ← NEXT
↓
Luong Multiplicative Attention
```

---

# ⭐ Golden Rule

> **The attention mechanism is a three-stage process: compare the decoder's current need with every source representation, normalize those compatibility scores into attention weights, and aggregate the source states into a dynamic context vector. Everything else is largely a design choice about how that comparison score is computed.**
