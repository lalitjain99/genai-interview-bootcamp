# ⚡ Cheat Sheet — Teacher Forcing

> **Goal:** Fast revision of what teacher forcing is, why it helps Seq2Seq training, why it creates exposure bias, and how teacher forcing ratio and scheduled sampling fit into the training–inference trade-off.

---

# 🌟 1. Teacher Forcing in One Line

> **During training, feed the decoder the true previous target token instead of its own previous prediction.**

```text id="tf-cheat-1"
Model predicts X

but next decoder input = true A
```

This keeps training on:

> the correct target history.

---

# 🔄 2. Target Shifting

Target:

```text id="tf-cheat-2"
A B C <EOS>
```

Decoder input:

```text id="tf-cheat-3"
<SOS> A B C
```

Expected output:

```text id="tf-cheat-4"
A B C <EOS>
```

Alignment:

```text id="tf-cheat-5"
Input:   <SOS>   A   B   C
Target:    A     B   C  <EOS>
```

---

# 🧠 3. What Does Teacher Forcing Actually Feed?

At timestep `t`:

# `input = y_(t-1)^*`

and the model predicts:

# `y_t^*`

So teacher forcing gives:

> the previous correct token,

not:

> the current answer.

---

# ⭐ 4. Why Use Teacher Forcing?

Without teacher forcing:

```text id="tf-cheat-6"
Wrong Prediction
↓
Wrong Next Input
↓
Different Hidden State
↓
More Possible Errors
```

This is especially problematic early in training.

Teacher forcing gives:

```text id="tf-cheat-7"
Correct Prefix
↓
Predict Next Token
```

which makes learning:

* more stable
* faster
* easier to optimize

---

# 📐 5. Mathematical View

Seq2Seq:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

With full teacher forcing:

# `P(y_t^* | y_<t^*, X)`

Token loss:

`L_t = -log P(y_t^* | y_<t^*, X)`

Sequence loss:

# `L = -Σ_t log P(y_t^* | y_<t^*, X)`

---

# 🚨 6. Main Drawback — Training vs Inference

Training:

```text id="tf-cheat-8"
true y1
↓
true y2
↓
true y3
```

Inference:

```text id="tf-cheat-9"
predicted y1
↓
predicted y2
↓
predicted y3
```

So:

# **Training History ≠ Inference History**

This creates:

# **Exposure Bias**

---

# 🧠 7. Exposure Bias

Exposure bias means:

> the model trains mostly on ground-truth histories but must infer on self-generated histories.

Training may see:

```text id="tf-cheat-10"
<SOS> A B
```

Inference may produce:

```text id="tf-cheat-11"
<SOS> A X
```

The model may be less prepared for:

> its own mistakes.

---

# 🔄 8. Exposure Bias vs Error Propagation

### Exposure Bias

Cause:

> train/inference history mismatch.

### Error Propagation

Effect:

```text id="tf-cheat-12"
Wrong Token
↓
Different Next Input
↓
Different Hidden State
↓
Potential Later Errors
```

So:

# `Exposure Bias ≠ Error Propagation`

but they are related.

---

# 📐 9. Teacher Forcing Ratio

Let:

# `p_TF`

be the probability of using the ground-truth previous token.

Example:

`p_TF = 0.8`

roughly means:

```text id="tf-cheat-13"
80% → ground truth
20% → model prediction
```

Exact sampling may be per:

* timestep
* sequence
* batch

depending on implementation.

---

# ⭐ 10. Ratio Trade-Off

## `p_TF = 1.0`

100% teacher forcing.

Pros:

* stable training
* clean target history
* faster convergence

Cons:

* larger train/inference mismatch

## `p_TF = 0`

No teacher forcing.

Pros:

* closer to inference

Cons:

* harder early training
* more error compounding
* less stable optimization

---

# 🔁 11. Scheduled Sampling

Idea:

> start with more teacher forcing and gradually reduce it.

Example:

```text id="tf-cheat-14"
Early Epochs
→ 100%

Later
→ 80%

Later
→ 50%

Later
→ 20%
```

Goal:

```text id="tf-cheat-15"
Early
→ easier learning

Later
→ more inference-like history
```

---

# ⚠️ 12. Scheduled Sampling Is Not Perfect

Scheduled sampling may reduce the mismatch, but:

* it changes the training distribution
* discrete token sampling complicates optimization
* it does not guarantee elimination of exposure bias

So:

> mitigation, not perfect solution.

---

# 🔗 13. Teacher Forcing vs Related Concepts

```text id="tf-cheat-16"
Teacher Forcing
→ training input strategy

Target Shifting
→ input/label alignment

Exposure Bias
→ training/inference history mismatch

Autoregressive Decoding
→ self-fed inference

Greedy Decoding
→ local token selection strategy

Beam Search
→ sequence search strategy

Context Bottleneck
→ source access problem

Attention
→ dynamic source access
```

Keep these separate.

---

# 🧠 14. Teacher Forcing Does Not Change the Loss

Usually:

```text id="tf-cheat-17"
Teacher Forcing
→ changes decoder input history

Cross-Entropy
→ still the objective
```

Do not confuse:

> input strategy

with:

> loss function.

---

# 🔄 15. BPTT Still Applies

Teacher forcing does not remove recurrence.

For an RNN decoder:

```text id="tf-cheat-18"
d1 → d2 → d3 → d4
```

hidden states still depend sequentially on previous hidden states.

So:

> BPTT is still required.

---

# ⚠️ 16. No Future Leakage

Teacher forcing allows:

```text id="tf-cheat-19"
Predict y3
using
y1, y2
```

It does not allow:

```text id="tf-cheat-20"
Predict y3
using
y4
```

Only:

> previous ground-truth tokens

are valid.

---

# 🎯 17. Evaluation Warning

Teacher-forced token accuracy may be high while free-running generation is poor.

Why?

Because teacher-forced evaluation gives:

> correct history.

Production generation gives:

> self-generated history.

Therefore always evaluate with:

> real autoregressive decoding

when judging sequence-generation quality.

---

# 🎤 18. 30-Second Interview Answer

> **Teacher forcing is a training strategy for autoregressive sequence models where the decoder receives the ground-truth previous target token instead of its own previous prediction. This makes training more stable because every next-token prediction is conditioned on a correct history. The downside is that inference uses self-generated histories, creating a training–inference mismatch called exposure bias. A lower teacher forcing ratio or scheduled sampling can expose the model to more inference-like histories, but this makes training harder and does not completely eliminate the problem.**

---

# 🚫 19. High-Yield Traps

### ❌ Teacher forcing gives the current target token

No.

It gives the previous one.

### ❌ Teacher forcing is inference

No.

Training strategy.

### ❌ Teacher forcing removes autoregression

No.

### ❌ Teacher forcing allows future targets

No.

### ❌ Teacher forcing = target shifting

Related but different.

### ❌ Teacher forcing changes cross-entropy

No.

### ❌ 100% teacher forcing is always best

No.

### ❌ 0% teacher forcing is always best

No.

### ❌ Scheduled sampling completely solves exposure bias

No.

### ❌ Exposure bias = overfitting

No.

### ❌ Attention fixes exposure bias

No.

Different problem.

---

# 📐 20. Formula Flash Card

### Seq2Seq

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Full Teacher Forcing

# `P(y_t^* | y_<t^*, X)`

### Token Loss

# `L_t = -log P(y_t^* | y_<t^*, X)`

### Sequence Loss

# `L = -Σ_t log P(y_t^* | y_<t^*, X)`

### Teacher Forcing Ratio

```text id="tf-cheat-21"
p_TF
```

```text id="tf-cheat-22"
with probability p_TF
→ use true previous token

with probability 1-p_TF
→ use model prediction
```

---

# 🧠 10 Things You Must Know Cold

1. **Teacher forcing uses the true previous target token.**
2. **It still predicts the next token.**
3. **It stabilizes decoder training.**
4. **It prevents early predictions from immediately corrupting training history.**
5. **Inference does not have ground-truth target history.**
6. **That mismatch is exposure bias.**
7. **Exposure bias and error propagation are different but related.**
8. **Teacher forcing ratio controls ground-truth usage frequency.**
9. **Scheduled sampling gradually mixes in model predictions.**
10. **Teacher forcing is a training strategy, not an inference or source-access strategy.**

---

# 🧠 Final Mental Model

## With Teacher Forcing

```text id="tf-cheat-23"
<SOS>
↓
predict y1
↓
feed TRUE y1
↓
predict y2
↓
feed TRUE y2
```

Benefit:

```text id="tf-cheat-24"
Correct History
↓
Cleaner Learning
```

But:

```text id="tf-cheat-25"
TRAINING
Ground-Truth History

≠

INFERENCE
Self-Generated History

↓
Exposure Bias
```

And once inference makes a mistake:

```text id="tf-cheat-26"
Wrong Token
↓
Different Decoder History
↓
Different Hidden State
↓
Potential Error Cascade
```

Trade-off:

```text id="tf-cheat-27"
More Teacher Forcing
→ easier training
→ bigger mismatch

Less Teacher Forcing
→ harder training
→ more inference-like experience
```

---

# ⭐ Golden Rule

> **Teacher forcing improves training by giving the decoder the correct history, but every bit of that convenience increases the gap between how the model learns and how it must generate when the teacher is gone.**
