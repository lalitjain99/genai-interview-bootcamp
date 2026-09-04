# 💡 Attention Mechanism — Assignment

> Goal: Move beyond definitions and practice the actual reasoning behind attention: scoring, normalization, weighted context, masking, shapes, and architecture decisions.

---

# 🧮 1. Compute Attention Weights

Suppose a decoder timestep produces these raw attention scores:

```text
e_t = [1.0, 2.0, 0.0]
```

Calculate the softmax attention weights:

```text
α_t
```

Use:

```text
α_i = exp(e_i) / Σ_j exp(e_j)
```

### Answer

Approximate exponentials:

```text
exp(1) ≈ 2.72
exp(2) ≈ 7.39
exp(0) = 1
```

Total:

```text
2.72 + 7.39 + 1 = 11.11
```

Therefore:

```text
α ≈ [
  2.72 / 11.11,
  7.39 / 11.11,
  1 / 11.11
]
```

So:

```text
α ≈ [0.245, 0.665, 0.090]
```

The second source position receives the strongest attention.

---

# 🧮 2. Compute the Context Vector

Given:

```text
h1 = [1, 0]
h2 = [0, 2]
h3 = [1, 1]
```

and:

```text
α = [0.2, 0.5, 0.3]
```

compute:

```text
c_t = Σ_i α_i h_i
```

### Answer

```text
c_t
=
0.2[1,0]
+
0.5[0,2]
+
0.3[1,1]
```

```text
=
[0.2,0]
+
[0,1.0]
+
[0.3,0.3]
```

Therefore:

```text
c_t = [0.5, 1.3]
```

---

# 🧠 3. Score or Weight?

For each value below, identify whether it is more likely to be:

* raw attention score `e_(t,i)`
* normalized attention weight `α_(t,i)`

### A

```text
-3.7
```

### B

```text
0.62
```

### C

```text
5.4
```

### D

```text
0.08
```

### Answer

A → likely raw score

B → could be attention weight

C → likely raw score

D → could be attention weight

Why?

Raw scores can be any real number.

Standard softmax weights are non-negative and typically lie between 0 and 1.

---

# 🧠 4. Check the Attention Distribution

Suppose:

```text
α = [0.2, 0.4, 0.5]
```

Is this a valid softmax attention distribution?

### Answer

No.

Because:

```text
0.2 + 0.4 + 0.5 = 1.1
```

For a standard softmax distribution over source positions:

```text
Σ_i α_i = 1
```

---

# 🔍 5. Find the Softmax-Dimension Bug

A batch has:

```text
batch_size = 32
source_length = 10
```

The attention-score tensor has shape:

```text
[32, 10]
```

The developer applies:

```text
softmax(scores, dim=0)
```

What is wrong?

### Answer

`dim=0` normalizes across the batch.

Attention should normally normalize across source positions.

For shape:

```text
[batch, source]
```

softmax should typically be applied along:

```text
dim=1
```

so each example gets its own distribution across its source tokens.

---

# 🧠 6. Reason About Shapes

Suppose:

```text
source length T = 7
encoder hidden size H = 128
```

What are the shapes of:

* one encoder state `h_i`
* all encoder states
* attention scores for one decoder step
* attention weights for one decoder step
* context vector

### Answer

```text
h_i                 → [128]
all encoder states  → [7, 128]
scores e_t           → [7]
weights α_t          → [7]
context c_t          → [128]
```

---

# 🔍 7. Diagnose This Incorrect Context Calculation

A developer writes:

```text
c_t = Σ_i α_i
```

What is missing?

### Answer

The encoder representations.

Correct:

```text
c_t = Σ_i α_i h_i
```

The weights alone contain relevance, not source meaning.

The semantic information is carried by:

```text
h_i
```

---

# 🧠 8. Fixed Context vs Dynamic Context

Consider these two systems.

### System A

```text
c = h_T
```

and the same `c` is used for every decoder step.

### System B

```text
c_t = Σ_i α_(t,i) h_i
```

Which system is more likely to preserve a date mentioned early in a long source sequence?

### Answer

System B.

Because the decoder can assign high attention to the early encoder state when that detail becomes relevant.

System A forces all information through one fixed source summary.

---

# 🧮 9. Compare Two Decoder Steps

Suppose the same source produces:

```text
h1, h2, h3, h4
```

At target step 1:

```text
α_1 = [0.7, 0.2, 0.05, 0.05]
```

At target step 2:

```text
α_2 = [0.1, 0.1, 0.2, 0.6]
```

What does this tell you?

### Answer

The decoder is retrieving different source information at different target steps.

Step 1 mainly uses `h1`.

Step 2 mainly uses `h4`.

This demonstrates the central property of attention:

> dynamic, target-step-specific source access.

---

# 🚨 10. Padding-Mask Debugging

Source batch:

```text
[token1, token2, token3, PAD, PAD]
```

Raw attention scores:

```text
[2.0, 1.5, 0.5, 3.0, 2.5]
```

The developer applies softmax immediately.

What is the problem?

### Answer

PAD positions have large scores and will receive substantial attention mass.

Correct approach:

```text
[2.0, 1.5, 0.5, -∞, -∞]
↓
softmax
```

Then PAD weights become approximately zero.

---

# 🧠 11. Why Mask Before Softmax?

A developer suggests:

1. softmax all scores
2. set PAD weights to zero afterward

Why is this less correct than masking before softmax?

### Answer

Because PAD positions already consumed probability mass during softmax.

If their weights are zeroed afterward, the remaining valid positions no longer sum to 1 unless renormalized.

Masking before softmax ensures normalization happens only over valid source positions.

---

# 🧠 12. Does Attention Solve the Encoder Problem?

Suppose an encoder completely fails to capture a person's name in any hidden state.

Can attention recover it?

### Answer

No.

Attention can only retrieve from available encoder representations.

If the information is absent from:

```text
h_1, ..., h_T
```

attention cannot reconstruct it.

Attention improves access, not magical information recovery.

---

# 🔍 13. Debug the Wrong Mental Model

Someone says:

> “Attention solves the problem by increasing the size of the context vector.”

What is wrong with this explanation?

### Answer

Attention's main innovation is not merely a larger context vector.

The important change is:

```text
one fixed context
↓
dynamic context per decoder step
```

Even if `c_t` has the same dimension as the original context, access is improved because it is recomputed from all encoder states according to current relevance.

---

# 🧠 14. Attention vs Beam Search

A translation model forgets source names.

The team increases beam width from:

```text
k = 4
```

to:

```text
k = 16
```

but the names are still frequently omitted.

Why might this happen?

### Answer

Beam Search improves:

> target-side search.

It does not improve:

> source-side information access.

If the decoder cannot access the relevant source representation effectively, wider beam search will not fix the underlying bottleneck.

---

# 🧠 15. Attention vs LSTM

A colleague says:

> “Once we use attention, LSTM is unnecessary.”

Is this historically and architecturally correct?

### Answer

No.

Classic attention was commonly added on top of recurrent encoder-decoder models.

For example:

```text
LSTM Encoder
+
Attention
+
LSTM Decoder
```

LSTM and attention solve different problems:

```text
LSTM
→ recurrent memory / information flow

Attention
→ dynamic source access
```

---

# 📐 16. Interpret This Attention Matrix

Suppose the attention matrix is:

```text
          src1 src2 src3 src4
tgt1      0.8  0.1  0.1  0.0
tgt2      0.1  0.7  0.1  0.1
tgt3      0.0  0.1  0.2  0.7
```

What can you infer?

### Answer

Each row is a decoder timestep.

Each column is a source position.

The decoder appears to align:

```text
tgt1 → mainly src1
tgt2 → mainly src2
tgt3 → mainly src4
```

This suggests learned alignment, but it should not automatically be treated as a perfect causal explanation.

---

# 🧠 17. Design the Attention Pipeline

Fill in the missing steps:

```text
Decoder State
+
Encoder States
↓
___________
↓
___________
↓
Attention Weights
↓
___________
↓
Context Vector
```

### Answer

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
Weighted Sum of Encoder States
↓
Context Vector
```

---

# ✅ True / False

## 1.

Attention scores must sum to 1.

**Answer:** False.

Attention weights usually do.

---

## 2.

Softmax is typically applied over source positions for each decoder step.

**Answer:** True.

---

## 3.

Attention weights contain the semantic content of the source.

**Answer:** False.

The encoder states contain the representations.

---

## 4.

The context vector may change at every decoder timestep.

**Answer:** True.

---

## 5.

Attention and vocabulary softmax serve the same purpose.

**Answer:** False.

---

## 6.

A high attention weight always proves causal importance.

**Answer:** False.

---

## 7.

Attention can use multiple source positions simultaneously.

**Answer:** True.

---

## 8.

Standard softmax attention weights are non-negative.

**Answer:** True.

---

## 9.

Beam Search fixes the context-vector bottleneck.

**Answer:** False.

---

## 10.

Padding should normally be masked before attention softmax.

**Answer:** True.

---

# ⭐ Staff Engineer Challenge

You are evaluating two translation architectures.

## Architecture A

```text
BiLSTM Encoder
↓
Final Encoder Summary
↓
LSTM Decoder
↓
Beam Search
```

## Architecture B

```text
BiLSTM Encoder
↓
All Encoder States
↓
Attention
↓
LSTM Decoder
↓
Beam Search
```

The production workload contains:

* long legal sentences
* many names and dates
* high fidelity requirements
* strict latency SLOs

### Questions

1. Why might B improve output quality?
2. What additional cost does B introduce?
3. What metrics would you monitor?
4. Would you immediately increase beam size as well?
5. If attention weights become very diffuse, is that automatically a bug?

### Answer

### 1. Why quality may improve

Architecture B gives the decoder dynamic access to specific source positions.

This helps preserve details that may otherwise be weakened inside one fixed summary.

---

### 2. Additional cost

At each target step, the decoder scores many or all source states.

Conceptually:

```text
O(T_src × T_tgt)
```

attention-score work is introduced.

This increases:

* compute
* latency
* memory traffic

depending on implementation.

---

### 3. Metrics

Monitor both model quality and system performance.

Quality:

* entity preservation
* date/number preservation
* sequence accuracy
* long-input degradation
* alignment diagnostics

System:

* p50 / p95 / p99 latency
* memory
* throughput
* source-length sensitivity
* target-length sensitivity

---

### 4. Increase beam size immediately?

Not necessarily.

Attention improves source access.

Beam width affects search.

First measure the quality gain from attention independently.

Then tune beam width based on quality-latency trade-offs.

---

### 5. Is diffuse attention automatically wrong?

No.

Some predictions require information from multiple source positions.

Diffuse attention may be valid.

The right question is:

> Does the resulting model behavior improve task performance and preserve required information?

---

# 🧠 Final Exercise — Explain Without Equations

Explain Attention Mechanism to an engineer using only this conceptual chain:

```text
What does the decoder need now?
↓
Which source representations are relevant?
↓
How relevant is each one?
↓
Normalize the relevance
↓
Blend the useful source information
↓
Use that context for prediction
```

### Model Answer

> The decoder's needs change as it generates the target sequence. Attention compares the decoder's current state with all source representations, gives each source position a relevance score, normalizes those scores into weights, and combines the source states according to those weights. The resulting context therefore changes from one decoder step to another, allowing the model to retrieve the source information that matters at that moment.

---

# 🎯 Assignment Mental Model

If you can reconstruct this without memorizing:

```text
e_(t,i) = score(s_(t-1), h_i)
↓
α_(t,i) = softmax(e_(t,i))
↓
c_t = Σ_i α_(t,i)h_i
```

and explain:

```text
Score
≠
Weight
≠
Source Representation
≠
Context
```

then you understand the core Attention Mechanism.

---

# 🔗 Next Topic

The generic mechanism still leaves one engineering decision unresolved:

> **How should the compatibility score be calculated?**

That leads directly to:

# **Bahdanau / Additive Attention**
