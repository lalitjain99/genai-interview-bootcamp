# 🎤 Attention Mechanism — Interview Questions

> Goal: Be able to explain the mechanism from intuition, equations, implementation details, and architectural trade-offs.

---

# 🟢 Beginner

## 1. What is the Attention Mechanism?

**Answer:**

Attention allows the decoder to build a **different source context for every target step** instead of relying on one fixed encoder summary.

At decoder step `t`:

```text
Decoder state
+
All encoder states
↓
Compatibility scores
↓
Softmax
↓
Attention weights
↓
Weighted sum
↓
Dynamic context c_t
```

Core equation:

```text
c_t = Σ_i α_(t,i) h_i
```

where:

* `h_i` = encoder state at source position `i`
* `α_(t,i)` = importance of that source position at decoder step `t`

---

## 2. Why was Attention needed in Seq2Seq models?

**Answer:**

Classic encoder-decoder models compressed the whole source sequence into a fixed context vector.

For long or information-rich sequences, this created a bottleneck:

```text
Variable-length source
↓
One fixed summary
↓
Decoder
```

Attention removes this fixed-access pattern by keeping all encoder states available and dynamically retrieving the relevant ones.

So the key improvement is:

> **dynamic source access instead of one static source summary.**

---

## 3. What is the difference between an attention score and an attention weight?

**Answer:**

The attention score:

```text
e_(t,i)
```

is a raw compatibility value between the decoder state and encoder state `h_i`.

The attention weight:

```text
α_(t,i)
```

is the normalized value obtained after softmax.

Pipeline:

```text
e_(t,i)
↓
softmax
↓
α_(t,i)
```

Scores can be arbitrary real numbers.

Weights are typically non-negative and sum to 1 across source positions.

---

## 4. What does the attention context vector represent?

**Answer:**

The context vector is a weighted combination of encoder states:

```text
c_t = Σ_i α_(t,i) h_i
```

It represents:

> the source information most relevant to the decoder at target step `t`.

Unlike the classic fixed context vector, this context changes during decoding.

---

## 5. Why is softmax used in Attention?

**Answer:**

Softmax converts arbitrary compatibility scores into a normalized relevance distribution.

For a fixed decoder timestep:

```text
α_(t,i) =
exp(e_(t,i))
/
Σ_j exp(e_(t,j))
```

This gives:

```text
α_(t,i) ≥ 0
```

and:

```text
Σ_i α_(t,i) = 1
```

So the weights can be interpreted as relative importance across source positions.

---

# 🟡 Intermediate

## 6. Walk me through one complete Attention timestep.

**Answer:**

At decoder step `t`:

### Step 1 — Obtain decoder state

Use the decoder state representing the current generation history.

For example:

```text
s_(t-1)
```

### Step 2 — Compare against every encoder state

```text
e_(t,i) = score(s_(t-1), h_i)
```

### Step 3 — Normalize scores

```text
α_(t,i) = softmax(e_(t,i))
```

Softmax is applied across source positions.

### Step 4 — Build context

```text
c_t = Σ_i α_(t,i) h_i
```

### Step 5 — Predict

Combine decoder information with `c_t` to produce the next-token distribution.

So the core sequence is:

```text
Compare
→ Normalize
→ Aggregate
→ Predict
```

---

## 7. Why does the attention distribution change at every decoder step?

**Answer:**

Because the decoder state changes after every generated token.

Even though the encoder states:

```text
h_1, ..., h_T
```

remain the same for the source sequence, the decoder's current need changes.

Therefore:

```text
same source representations
+
different decoder state
=
different compatibility scores
=
different attention distribution
```

That is what makes attention dynamic.

---

## 8. What dimension does the attention context vector have?

Assume every encoder state is:

```text
h_i ∈ R^H
```

**Answer:**

The context is:

```text
c_t = Σ_i α_(t,i) h_i
```

Since each `α_(t,i)` is a scalar and each `h_i` is `H`-dimensional:

```text
c_t ∈ R^H
```

The source length can vary, but the output context dimension remains fixed.

---

## 9. Why not concatenate all encoder states instead of taking a weighted sum?

**Answer:**

If source length is `T` and each encoder state has size `H`, concatenating all states would create roughly:

```text
T × H
```

features.

That means the dimensionality would depend on source length.

A decoder usually needs a fixed-dimensional interface.

Attention solves this by reducing a variable number of source states to one fixed-dimensional context:

```text
h_1, ..., h_T
↓
weighted sum
↓
c_t ∈ R^H
```

---

## 10. What is the difference between attention softmax and output softmax?

**Answer:**

They normalize completely different things.

### Attention softmax

Produces:

```text
α_(t,i)
```

Distribution over:

> source positions.

It answers:

> **Where should I look?**

### Output softmax

Produces:

```text
P(y_t)
```

Distribution over:

> target vocabulary.

It answers:

> **What token should I predict?**

This is a common interview trap.

---

# 🔴 Advanced

## 11. Does Attention eliminate the context bottleneck completely?

**Answer:**

No.

Attention removes the requirement that the entire source be compressed into one fixed context for the whole target sequence.

However, each decoder step still receives a fixed-dimensional vector:

```text
c_t
```

So compression still happens, but:

> it is recomputed dynamically based on current relevance.

This is much better than one global fixed summary.

---

## 12. How does Attention learn alignment without explicit alignment labels?

**Answer:**

The scoring function, encoder, and decoder are trained end-to-end using the task loss.

Suppose assigning more attention to `h_i` helps reduce next-token loss.

Gradient descent can update:

* scoring parameters
* encoder representations
* decoder representations

so that the useful source state receives a higher compatibility score in similar future situations.

Therefore alignment can emerge indirectly from:

> prediction loss.

Explicit word-level alignment supervision is not required in the standard setup.

---

## 13. Why should padded source tokens be masked before softmax?

**Answer:**

Consider:

```text
A B C <PAD> <PAD>
```

Without masking, padded positions could receive non-zero attention weights.

That wastes probability mass on meaningless representations.

So padded positions are given a very negative score:

```text
e_PAD → -∞
```

before softmax.

Then:

```text
softmax(e_PAD) ≈ 0
```

The key detail is:

> **mask before softmax, not after.**

---

## 14. Does a high attention weight prove that a token caused the model's prediction?

**Answer:**

No.

Attention weights tell us how strongly source representations contribute to the attention context.

They can be useful for visualization and alignment analysis, but:

> attention weight is not guaranteed to be a causal explanation of the final prediction.

The output also depends on:

* decoder state
* hidden transformations
* output projection
* interactions from previous timesteps

So attention is useful for interpretation, but should not be treated as perfect explainability.

---

## 15. What is the computational cost of recurrent encoder-decoder Attention?

**Answer:**

If:

* source length = `T_src`
* target length = `T_tgt`

then each target step compares with approximately all source states.

So conceptual scoring work grows roughly like:

```text
O(T_src × T_tgt)
```

ignoring the internal cost of the score function.

Attention improves accessibility, but adds compute compared with using one fixed source context.

---

# ⭐ Staff Engineer Challenge

## Scenario

You have a translation system:

```text
BiLSTM Encoder
↓
Final Encoder State
↓
LSTM Decoder
↓
Beam Search
```

It works well on short sentences.

For long sentences, you observe:

* names and dates are sometimes omitted
* early clauses are forgotten
* Beam Search improves fluency but not source fidelity
* increasing hidden size helps slightly
* training is stable
* gradients are not obviously exploding

### Question

What is the likely architectural problem, and what would you change?

---

## Strong Answer

The main issue is likely the:

> **fixed source-to-decoder communication bottleneck.**

The BiLSTM may encode richer representations, and a larger hidden size increases capacity, but the decoder still receives a fixed number of summary states independent of source length.

That means the real limitation is not only:

> storage capacity,

but also:

> accessibility.

The decoder cannot directly retrieve specific source information when it needs it.

Beam Search does not fix this because Beam Search operates on:

> target-side search.

The architectural change would be:

```text
Keep all encoder hidden states
↓
At every decoder timestep
↓
Compute compatibility with each source state
↓
Normalize scores
↓
Build dynamic context c_t
↓
Condition decoder prediction on c_t
```

In other words:

> add an attention mechanism.

I would evaluate the change using both quality and system metrics:

* sequence-level accuracy / BLEU-style task metrics
* entity/date preservation
* long-sequence performance
* attention alignment diagnostics
* latency
* memory
* source and target length sensitivity

The expected trade-off is:

> better source access in exchange for additional per-decoder-step computation.

---

# ⚡ Rapid-Fire Interview Traps

## Does `e_(t,i)` have to be positive?

No.

---

## Does `α_(t,i)` usually have to be positive?

With standard softmax attention, yes.

---

## For a fixed decoder step, what does `Σ_i α_(t,i)` equal?

```text
1
```

---

## Is `α_(t,i)` a scalar or vector?

Scalar.

---

## Is `h_i` a scalar or vector?

Vector.

---

## Does Attention choose the output token?

No.

It creates source context.

---

## Does softmax perform hard selection?

No.

It gives soft weights.

---

## Can multiple source positions receive high attention?

Yes.

---

## Does Attention replace LSTM memory?

Not necessarily.

Classic attention was commonly used on top of recurrent encoders and decoders.

---

## Does Beam Search solve the fixed-context bottleneck?

No.

---

## Does a larger hidden state eliminate the bottleneck?

No.

It may improve capacity, but accessibility remains limited.

---

## Is Attention primarily a vanishing-gradient solution?

No.

Its central role here is source access and encoder-decoder communication.

---

## Can Attention recover information the encoder never represented?

No.

---

# 📐 Formula Card

## Score

```text
e_(t,i) = score(s_(t-1), h_i)
```

## Attention Weight

```text
α_(t,i) =
exp(e_(t,i))
/
Σ_j exp(e_(t,j))
```

## Normalization

```text
Σ_i α_(t,i) = 1
```

## Context

```text
c_t = Σ_i α_(t,i) h_i
```

---

# 🧠 One Diagram to Remember

```text
                 Decoder State
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
      h1              h2             ... hT
       │               │               │
       ▼               ▼               ▼
     score           score           score
       │               │               │
       └───────────────┼───────────────┘
                       ▼
                  Raw Scores
                       │
                       ▼
                    Softmax
                       │
                       ▼
                Attention Weights
                       │
                       ▼
            Weighted Encoder States
                       │
                       ▼
                    Context c_t
                       │
                       ▼
                Decoder Prediction
```

---

# 🎯 30-Second Interview Response

> **Attention lets an encoder-decoder model dynamically retrieve source information at every decoder timestep. The decoder state is compared with each encoder hidden state to produce compatibility scores. Softmax converts those scores into normalized attention weights, and the weighted sum of encoder states forms the context vector. This avoids forcing the entire source sequence through one fixed context and gives the decoder step-specific access to relevant source information.**

---

# ⭐ Final Interview Mental Model

When asked about Attention, reconstruct it in this order:

```text
Why?
Fixed context bottleneck

↓

What changes?
Keep all encoder states

↓

How?
Compare decoder need with each state

↓

What do comparisons produce?
Scores

↓

How are scores made usable?
Softmax

↓

What do weights do?
Weighted sum

↓

What is produced?
Dynamic context c_t
```

The shortest technical answer is:

# **Compare → Normalize → Aggregate**

And the next interview question should naturally be:

> **How do you implement `score(s_(t-1), h_i)`?**

That takes us to:

# **Bahdanau / Additive Attention**
