# ⚡ Attention Mechanism — Cheat Sheet

> Goal: Reconstruct the full mechanism in a few minutes before an interview.

---

# 1. 🎯 One-Line Definition

> **Attention computes a different source context for every decoder step by scoring encoder states, normalizing the scores, and taking a weighted sum.**

---

# 2. 🧠 Why Attention?

Classic Seq2Seq:

```text
Source
↓
Encoder
↓
One Fixed Context
↓
Decoder
```

Problem:

> All source information must pass through one fixed summary.

Attention:

```text
Keep all encoder states
↓
Retrieve what matters at each decoder step
```

---

# 3. 📐 Core Equations

### Score

```text
e_(t,i) = score(s_(t-1), h_i)
```

### Attention Weight

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

# 4. ⭐ Core Mental Model

# **Compare → Normalize → Aggregate**

```text
Decoder Need
+
Encoder States
↓
Compare
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
```

---

# 5. 🔍 Score vs Weight vs Context

| Quantity  | Meaning                       |
| --------- | ----------------------------- |
| `e_(t,i)` | raw compatibility score       |
| `α_(t,i)` | normalized relevance weight   |
| `h_i`     | encoder/source representation |
| `c_t`     | weighted source context       |

Remember:

```text
Score
≠
Weight
≠
Representation
≠
Context
```

---

# 6. 🧠 What Does the Decoder State Represent?

The decoder state captures:

* generated target history
* current decoding context
* what information may be needed next

So attention asks:

> **Which source positions are most relevant to the decoder's current need?**

---

# 7. 📐 Softmax Rule

For a fixed decoder timestep `t`:

```text
Σ_i α_(t,i) = 1
```

Softmax is applied across:

> **source positions**

not across decoder timesteps.

---

# 8. 🧮 Tiny Example

Scores:

```text
e = [1, 2, 3]
```

Softmax:

```text
α ≈ [0.09, 0.24, 0.67]
```

Interpretation:

> Source position 3 matters most.

---

# 9. 🧮 Context Example

```text
h1 = [1,0]
h2 = [0,1]
h3 = [1,1]

α = [0.2,0.3,0.5]
```

Then:

```text
c_t
=
0.2h1
+
0.3h2
+
0.5h3
```

Result:

```text
c_t = [0.7,0.8]
```

---

# 10. 📐 Shape Intuition

If:

```text
source length = T
hidden size = H
```

then:

```text
h_i        → [H]
H_enc      → [T, H]
scores e_t → [T]
weights α_t→ [T]
context c_t→ [H]
```

---

# 11. 🔄 Why Attention Is Dynamic

Encoder states usually stay fixed:

```text
h1, h2, ..., hT
```

Decoder state changes each timestep.

Therefore:

```text
different decoder state
→ different scores
→ different weights
→ different context
```

So generally:

```text
c_1 ≠ c_2 ≠ c_3
```

---

# 12. 🚨 Attention Softmax vs Output Softmax

### Attention Softmax

Distribution over:

> source positions

Answers:

# **WHERE to look**

### Output Softmax

Distribution over:

> vocabulary tokens

Answers:

# **WHAT to predict**

---

# 13. 🛡️ Padding Mask

For padded source:

```text
A B C PAD PAD
```

Mask PAD scores before softmax:

```text
PAD score → -∞
```

Then:

```text
attention weight ≈ 0
```

Rule:

> **Mask before softmax.**

---

# 14. 🧠 What Attention Solves

Attention mainly improves:

# **source accessibility**

It does not simply increase storage capacity.

Classic:

```text
compress everything once
```

Attention:

```text
retain source states
+
retrieve dynamically
```

---

# 15. 🔗 Important Distinctions

### Attention vs LSTM / GRU

```text
LSTM / GRU
→ recurrent memory and information flow

Attention
→ dynamic source access
```

### Attention vs Beam Search

```text
Beam Search
→ target-side search

Attention
→ source-side access
```

### Attention vs Teacher Forcing

```text
Teacher Forcing
→ decoder training strategy

Attention
→ source retrieval mechanism
```

---

# 16. 🚫 What Attention Does NOT Do

Attention does not automatically:

* fix all vanishing-gradient problems
* remove recurrence
* choose the output token directly
* guarantee interpretability
* recover information the encoder never captured
* eliminate all compression
* solve exposure bias
* solve search errors

---

# 17. 🧠 Soft Attention

Standard attention usually produces:

```text
[0.1, 0.2, 0.7]
```

not:

```text
[0, 0, 1]
```

So it performs:

> weighted retrieval, not hard selection.

---

# 18. 🔗 Where Bahdanau and Luong Fit

Generic attention leaves this abstract:

```text
score(s_(t-1), h_i)
```

Then:

```text
Bahdanau
→ additive / neural scoring

Luong
→ multiplicative / dot-product-style scoring
```

So they are mainly different answers to:

> **How should compatibility be computed?**

---

# ⚡ 10 Things to Know Cold

1. Attention solves the fixed source-access problem.
2. `e_(t,i)` is a raw compatibility score.
3. `α_(t,i)` is a normalized attention weight.
4. Softmax runs across source positions.
5. `Σ_i α_(t,i) = 1`.
6. `c_t` is a weighted sum of encoder states.
7. Attention changes at each decoder step.
8. Attention softmax ≠ vocabulary softmax.
9. PAD positions should be masked before softmax.
10. Bahdanau and Luong mainly differ in the scoring function.

---

# 🎤 30-Second Interview Answer

> **Attention lets a decoder dynamically retrieve relevant source information at every target step. The decoder state is compared with each encoder state to produce compatibility scores. Softmax converts those scores into attention weights, and the weighted sum of encoder states forms the context vector. This avoids relying on one fixed source summary for the entire output sequence.**

---

# 🧠 Final Mental Model

```text
What does the decoder need?
↓
Which source states are relevant?
↓
How relevant is each?
↓
Normalize relevance
↓
Blend source information
↓
Use dynamic context for prediction
```

Shortest version:

# **Compare → Normalize → Aggregate**

---

# 🔗 Next

The next question is:

> **How should `score(s_(t-1), h_i)` be implemented?**

That leads to:

# **Bahdanau / Additive Attention**
