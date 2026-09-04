# 🚫 Attention Mechanism — Common Mistakes

> Goal: Avoid the most common conceptual, mathematical, and interview mistakes around classical encoder-decoder attention.

---

# 1. ❌ Confusing Attention Score with Attention Weight

Wrong:

```text
e_(t,i) = α_(t,i)
```

Correct:

```text
e_(t,i)
↓
softmax
↓
α_(t,i)
```

* `e_(t,i)` = raw compatibility score
* `α_(t,i)` = normalized attention weight

A score can be negative.

A standard softmax attention weight is non-negative.

---

# 2. ❌ Saying Attention Uses One Fixed Context Vector

Classic Seq2Seq:

```text
c = h_T
```

Attention:

```text
c_t = Σ_i α_(t,i) h_i
```

The context is recomputed for each decoder step.

That dynamic behavior is the whole point.

---

# 3. ❌ Saying Attention Simply Makes the Context Vector Bigger

Attention does not mainly solve the problem by increasing dimensionality.

The important change is:

```text
Static Access
↓
Dynamic Access
```

Even if `c_t` has the same dimension as the old context vector, the decoder can retrieve different source information at different timesteps.

---

# 4. ❌ Applying Softmax Across the Wrong Dimension

For one decoder step, softmax should normalize across:

> source positions.

If scores have shape:

```text
[batch, source_length]
```

softmax should usually operate over the `source_length` dimension.

Wrong normalization can mix unrelated batch examples or timesteps.

---

# 5. ❌ Forgetting That Attention Softmax and Output Softmax Are Different

Attention softmax:

```text
source scores
↓
α
```

Distribution over:

> source positions

Output softmax:

```text
vocabulary logits
↓
P(y_t)
```

Distribution over:

> target vocabulary

Memory trick:

```text
Attention Softmax
→ WHERE to look

Output Softmax
→ WHAT to predict
```

---

# 6. ❌ Treating Attention as Hard Selection

Wrong mental model:

```text
pick exactly one encoder state
```

Standard soft attention usually does:

```text
c_t = Σ_i α_(t,i) h_i
```

Multiple source states can contribute simultaneously.

Example:

```text
α = [0.1, 0.2, 0.7]
```

not necessarily:

```text
[0, 0, 1]
```

---

# 7. ❌ Thinking the Weight Contains Semantic Information

Suppose:

```text
α_(t,3) = 0.8
```

`0.8` does not contain the meaning of source token 3.

The semantic representation is in:

```text
h_3
```

The weight only controls:

> how strongly `h_3` contributes to `c_t`.

---

# 8. ❌ Forgetting to Mask Padding Before Softmax

Wrong:

```text
scores
↓
softmax
↓
set PAD weights to 0
```

Better:

```text
PAD scores → -∞
↓
softmax
↓
PAD weights ≈ 0
```

Mask before normalization so valid source positions receive the full probability mass.

---

# 9. ❌ Saying Attention Completely Eliminates Compression

Attention still produces:

```text
c_t
```

which is typically fixed-dimensional.

The difference is that `c_t` is:

> dynamically recomputed from all encoder states.

So attention reduces the fixed-summary bottleneck, but does not remove every form of compression.

---

# 10. ❌ Saying Attention Was Invented Mainly to Fix Vanishing Gradients

Attention can create shorter information and gradient paths, which helps learning.

But its central motivation in classic Seq2Seq was:

> improving source access and encoder-decoder communication.

Vanishing gradients and the context bottleneck are related to sequence difficulty, but they are not the same problem.

---

# 11. ❌ Assuming LSTM and Attention Solve the Same Problem

They do not.

```text
LSTM / GRU
→ recurrent memory and gradient-flow improvement

Attention
→ dynamic access to encoder representations
```

Historically they were commonly used together:

```text
LSTM Encoder
+
Attention
+
LSTM Decoder
```

---

# 12. ❌ Assuming Beam Search Can Replace Attention

Beam Search improves:

> target-side search.

Attention improves:

> source-side access.

A wider beam cannot fix the fact that the decoder cannot access relevant source details.

---

# 13. ❌ Assuming Teacher Forcing Solves the Same Problem as Attention

Teacher Forcing addresses:

> decoder training history.

Attention addresses:

> source information retrieval.

They operate on different parts of the system.

---

# 14. ❌ Assuming All Decoder Steps Use the Same Attention Distribution

Wrong:

```text
α_1 = α_2 = α_3
```

In general:

```text
α_1 ≠ α_2 ≠ α_3
```

because decoder state changes with the generated target history.

That is why the context is dynamic.

---

# 15. ❌ Assuming Higher Attention Weight Means Guaranteed Causal Importance

A high attention weight means:

> the corresponding encoder representation contributes strongly to the attention context.

It does not prove that the source token was the sole cause of the final output.

The final prediction also depends on:

* decoder state
* output transformations
* previous target history
* other hidden representations

So:

> attention visualization is useful, but not perfect causal explainability.

---

# 16. ❌ Assuming Attention Can Recover Information the Encoder Lost

Attention can only retrieve from:

```text
h_1, ..., h_T
```

If the encoder never captured the information:

> attention cannot reconstruct it.

Attention improves accessibility, not information creation.

---

# 17. ❌ Memorizing Bahdanau or Luong Before Understanding Generic Attention

Do not start with formulas like:

```text
v^T tanh(...)
```

or:

```text
s_t^T W h_i
```

before understanding:

```text
Compare
↓
Normalize
↓
Aggregate
```

Bahdanau and Luong are mainly different ways to implement:

```text
score(decoder_state, encoder_state)
```

The generic mechanism comes first.

---

# ⚡ Quick Debug Checklist

When an attention implementation behaves strangely, check:

1. Are encoder states being retained?
2. Is one score produced per source position?
3. Is softmax applied over source positions?
4. Are PAD positions masked before softmax?
5. Do attention weights sum to approximately 1?
6. Is the context computed as `Σ α_i h_i`?
7. Are tensor shapes aligned correctly?
8. Is attention softmax being confused with output softmax?
9. Are different decoder steps producing different distributions?
10. Is the issue really source access, or actually model/search/training quality?

---

# 📐 Formula Reminder

### Score

```text
e_(t,i) = score(s_(t-1), h_i)
```

### Weight

```text
α_(t,i) =
exp(e_(t,i))
/
Σ_j exp(e_(t,j))
```

### Context

```text
c_t = Σ_i α_(t,i) h_i
```

---

# 🧠 Final Mental Model

The easiest way to avoid most mistakes is to keep these roles separate:

```text
Decoder State
→ expresses current need

Encoder State h_i
→ contains source representation

Score e_(t,i)
→ raw compatibility

Weight α_(t,i)
→ normalized relevance

Context c_t
→ weighted source information
```

And remember:

# **Compare → Normalize → Aggregate**

---

# 🔗 Next Topic

The remaining question is:

> **How should the compatibility score be computed?**

That leads to:

# **Bahdanau / Additive Attention**
