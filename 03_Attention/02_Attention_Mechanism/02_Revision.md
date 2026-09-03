# 📝 Attention Mechanism — Revision Notes

## 🎯 One-Line Definition

> **Attention dynamically builds a context vector for each decoder step by scoring all encoder states, normalizing those scores into attention weights, and taking a weighted sum of the encoder states.**

---

# 🧠 Core Problem It Solves

Classic Seq2Seq used one fixed context:

```text
Source
↓
Encoder
↓
Single Context Vector c
↓
Decoder
```

Problem:

> Every decoder step had to use the same compressed source summary.

Attention changes this to:

```text
Keep all encoder states
↓
At each decoder step,
retrieve what matters now
```

---

# ⭐ Core Attention Equation

```text
c_t = Σ_i α_(t,i) h_i
```

where:

* `h_i` = encoder state at source position `i`
* `α_(t,i)` = attention weight for source position `i` at decoder step `t`
* `c_t` = dynamic context vector

---

# 🧠 The Main Question

We already know how to compute `c_t`.

The real question is:

> **Where do the attention weights `α_(t,i)` come from?**

Answer:

```text
Decoder State
+
Encoder State
↓
Compatibility Score
↓
Softmax
↓
Attention Weight
```

---

# 📐 Step 1 — Compute Compatibility Scores

For each source position:

```text
e_(t,i) = score(s_(t-1), h_i)
```

where:

* `s_(t-1)` = decoder state
* `h_i` = encoder hidden state
* `e_(t,i)` = raw compatibility score

Interpretation:

> How relevant is source position `i` to what the decoder currently needs?

---

# 🚨 Score ≠ Weight

## Attention Score

```text
e_(t,i)
```

* raw compatibility value
* can be negative
* can be positive
* not normalized

## Attention Weight

```text
α_(t,i)
```

* normalized relevance
* typically non-negative
* typically sums to 1 across source positions

Pipeline:

```text
Score e
↓
Softmax
↓
Weight α
```

---

# 📐 Step 2 — Normalize With Softmax

```text
α_(t,i) =
exp(e_(t,i))
/
Σ_j exp(e_(t,j))
```

For a fixed decoder timestep `t`:

```text
Σ_i α_(t,i) = 1
```

Softmax is applied across:

> **source positions**

not across decoder timesteps.

---

# 🧮 Small Example

Suppose:

```text
e = [1, 2, 3]
```

Softmax approximately gives:

```text
α = [0.09, 0.24, 0.67]
```

So the third source position receives the highest relevance.

---

# 📐 Step 3 — Build the Context Vector

```text
c_t = Σ_i α_(t,i) h_i
```

Expanded:

```text
c_t =
α_(t,1)h_1
+
α_(t,2)h_2
+
...
+
α_(t,T)h_T
```

This is:

> a weighted source summary tailored for decoder step `t`.

---

# 🧮 Context Example

Suppose:

```text
h1 = [1, 0]
h2 = [0, 1]
h3 = [1, 1]

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

Result:

```text
c_t = [0.7, 0.8]
```

---

# ⭐ Complete Attention Pipeline

```text
Decoder State
+
All Encoder States
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
↓
Decoder Prediction
```

In short:

# **Compare → Normalize → Aggregate**

---

# 🧠 Why Attention Is Dynamic

The encoder states usually remain the same during decoding:

```text
h1, h2, ..., hT
```

But decoder state changes at every target step.

Therefore:

```text
same encoder states
+
different decoder state
=
different attention distribution
```

So:

```text
c_1 ≠ c_2 ≠ c_3
```

in general.

---

# ⭐ Query-Dependent Access

At decoder step `t`, attention asks:

> Which source information is relevant now?

That is why attention behaves like:

```text
Current Decoder Need
↓
Search Source Representations
↓
Retrieve Weighted Information
```

---

# 📐 Shape Intuition

Suppose:

* source length = `T`
* encoder hidden size = `H`

Then:

```text
h_i ∈ R^H
```

Stacked encoder states:

```text
H_enc ∈ R^(T × H)
```

Scores:

```text
e_t ∈ R^T
```

Weights:

```text
α_t ∈ R^T
```

Context:

```text
c_t ∈ R^H
```

Mental model:

```text
T source states
↓
T scores
↓
T weights
↓
1 H-dimensional context
```

---

# 🧠 Why Scores Are Scalars

For each source position, attention needs one answer:

> How relevant is this position?

So each compatibility score is typically:

```text
e_(t,i) ∈ R
```

a scalar.

---

# 🚨 α Is Not the Source Representation

If:

```text
α_(t,3) = 0.8
```

that does not mean `0.8` contains semantic information.

The meaning lives in:

```text
h_3
```

The weight only controls:

> how much `h_3` contributes.

---

# ⭐ Attention Softmax vs Output Softmax

This is a very important distinction.

## Attention Softmax

Input:

```text
source relevance scores
```

Output:

```text
α_(t,i)
```

Distribution over:

> source positions

---

## Output Softmax

Input:

```text
vocabulary logits
```

Output:

```text
P(y_t)
```

Distribution over:

> target vocabulary

Easy memory trick:

```text
Attention Softmax
→ WHERE to look

Output Softmax
→ WHAT to predict
```

---

# 🧠 Masking

Suppose source sequence is padded:

```text
A B C <PAD> <PAD>
```

PAD positions should not receive attention.

So before softmax:

```text
PAD score → -∞
```

Then:

```text
softmax → weight ≈ 0
```

This prevents attention mass from being wasted on padding.

---

# ⭐ Why Mask Before Softmax

Because softmax distributes probability mass across all provided positions.

If PAD is not masked:

> meaningless positions may still receive non-zero attention.

---

# 🧠 Soft Attention

Softmax usually gives non-zero weights to multiple positions.

Example:

```text
[0.1, 0.2, 0.7]
```

This means attention is normally:

> soft, not hard selection.

It can focus strongly on one position while still using others.

---

# 🧠 Sharp vs Diffuse Attention

### Sharp

```text
[0.01, 0.02, 0.95, 0.02]
```

Most weight concentrated on one or few positions.

### Diffuse

```text
[0.20, 0.30, 0.25, 0.25]
```

Information spread across several source positions.

Neither is always better.

---

# 📐 Generic Score Function

So far:

```text
e_(t,i) = score(s_(t-1), h_i)
```

The `score(...)` function is the key design choice.

Different choices lead to different attention mechanisms.

Examples:

* additive neural score
* dot product
* multiplicative / bilinear score

This is what leads to:

```text
Bahdanau Attention
↓
Luong Attention
```

---

# 🔗 Bahdanau Connection

Bahdanau attention says:

> Compute compatibility using a small learned neural network.

Conceptually:

```text
decoder state
+
encoder state
↓
nonlinear learned transformation
↓
scalar score
```

---

# 🔗 Luong Connection

Luong-style attention uses:

> multiplication / dot-product-style scoring.

So:

```text
Generic Attention
↓
Need score(...)
↓
Bahdanau: additive
↓
Luong: multiplicative
```

---

# 🧠 Differentiability

The pipeline:

```text
score
↓
softmax
↓
weighted sum
```

is differentiable.

Therefore the model can learn attention end-to-end using task loss.

Gradients can flow into:

* attention parameters
* encoder states
* decoder states

---

# ⭐ Attention Learns Alignment

The model does not necessarily need explicit word-alignment labels.

If attending to a source position helps reduce prediction loss:

> training will reinforce that behavior.

This allows alignment to emerge naturally.

---

# 🧠 Alignment Is Usually Soft

One target token may depend on:

* one source token
* several source tokens
* a phrase
* a reordered source region

So attention is not necessarily one-to-one.

---

# 🚨 Attention Weight ≠ Perfect Explanation

High attention weight means:

> this representation contributed strongly to the computed context.

It does not automatically prove:

> causal interpretability.

Use attention visualization carefully.

---

# 🧠 Attention Does Not Invent Information

Attention can only retrieve from:

```text
h_1, ..., h_T
```

If the encoder failed to represent something:

> attention cannot recover it.

---

# ⭐ Attention Improves Access, Not Necessarily Storage

This is a key conceptual distinction.

Classic Seq2Seq:

```text
Store everything in one summary
```

Attention:

```text
Keep all source states
+
retrieve dynamically
```

So attention mainly solves:

> **accessibility**

not simply raw memory capacity.

---

# 🔗 Relation to Context Bottleneck

Classic:

```text
c = h_T
```

one source-to-decoder summary.

Attention:

```text
c_t = Σ_i α_(t,i) h_i
```

different source context per decoder step.

This is why attention helps long and detail-rich sequences.

---

# 🔗 Relation to LSTM / GRU

LSTM / GRU solve:

> recurrent memory and gradient-flow problems

Attention solves:

> source-access and encoder-decoder communication problems

They are complementary.

Early attention models often used:

```text
LSTM Encoder
+
Attention
+
LSTM Decoder
```

---

# 🔗 Relation to Beam Search

Beam Search:

> improves target-side search.

Attention:

> improves source-side access.

Beam Search does not solve the context bottleneck.

---

# 🔗 Relation to Teacher Forcing

Teacher Forcing:

> training strategy for decoder history.

Attention:

> source retrieval mechanism.

Different problems.

---

# 🔗 Computational Cost

At each target step, attention compares with all source states.

If:

* source length = `T`
* target length = `T'`

then conceptual scoring work is roughly:

```text
T × T'
```

This gives better source access at additional compute cost.

---

# 🎤 30-Second Interview Answer

> **Attention dynamically computes a source context for each decoder step. The decoder state is compared with every encoder hidden state to generate raw compatibility scores. These scores are normalized with softmax into attention weights, and the context vector is computed as the weighted sum of encoder states. This allows the decoder to retrieve different source information at different target steps instead of relying on one fixed context vector.**

---

# 🎤 Score vs Weight Interview Answer

> **An attention score is a raw compatibility value between the decoder state and an encoder state. An attention weight is the normalized value obtained after softmax across source positions. The weights are then used to compute the weighted context vector.**

---

# 🎤 Why Softmax?

> **Softmax converts arbitrary compatibility scores into a normalized differentiable distribution over source positions, making the values non-negative and summing them to one so they can be used as relative relevance weights.**

---

# 🚫 High-Yield Interview Traps

### ❌ Attention score and attention weight are the same

No.

```text
score → softmax → weight
```

---

### ❌ Attention directly chooses the output token

No.

Attention creates source context.

The decoder/output layer creates vocabulary logits.

---

### ❌ Softmax selects one encoder state

No.

It usually produces soft weights across many states.

---

### ❌ Attention softmax is vocabulary softmax

No.

Different distributions.

---

### ❌ Attention solves vanishing gradients only

No.

Its primary role here is dynamic source access.

---

### ❌ Larger hidden size completely solves context bottleneck

No.

Capacity may increase, but access pattern remains fixed.

---

### ❌ Attention eliminates compression completely

No.

Each `c_t` is still fixed-dimensional.

But it is dynamically recomputed.

---

### ❌ Attention weights are always interpretable

No.

They are useful signals, not guaranteed causal explanations.

---

# 📐 Formula Card

### Score

```text
e_(t,i) = score(s_(t-1), h_i)
```

### Normalize

```text
α_(t,i) =
exp(e_(t,i))
/
Σ_j exp(e_(t,j))
```

### Sum Constraint

```text
Σ_i α_(t,i) = 1
```

### Context

```text
c_t = Σ_i α_(t,i) h_i
```

---

# ⚡ 10 Things to Know Cold

1. Attention creates a **dynamic context vector**.
2. Decoder state is compared with every encoder state.
3. `e_(t,i)` is a raw compatibility score.
4. `α_(t,i)` is a normalized attention weight.
5. Softmax runs across source positions.
6. `c_t` is a weighted sum of encoder states.
7. Different decoder steps produce different attention distributions.
8. Attention softmax and vocabulary softmax are different.
9. Padding positions should be masked.
10. Bahdanau and Luong mainly differ in how `score(...)` is computed.

---

# 🧠 Final Mental Model

```text
Decoder Need
+
Source Representations
↓
Compare
↓
Scores
↓
Normalize
↓
Attention Weights
↓
Weighted Retrieval
↓
Dynamic Context
↓
Prediction
```

The shortest memory version:

# **Compare → Normalize → Aggregate**

And mathematically:

```text
e_(t,i) = score(s_(t-1), h_i)

α_(t,i) = softmax(e_(t,i))

c_t = Σ_i α_(t,i) h_i
```

---

# 🔗 Next Topic

The only major unanswered question is:

> **How exactly should `score(s_(t-1), h_i)` be computed?**

The next chapter answers that with:

# **Bahdanau / Additive Attention**
