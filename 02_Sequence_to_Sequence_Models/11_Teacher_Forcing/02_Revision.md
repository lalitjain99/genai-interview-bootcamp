# 📝 Revision — Teacher Forcing

> **Goal:** Quickly revise what teacher forcing is, why it makes Seq2Seq training easier, how it differs from inference, and why it creates exposure bias.

---

# 🌟 1. What Is Teacher Forcing?

Teacher forcing means:

> **During training, the decoder receives the true previous target token instead of its own previous prediction.**

Example target:

```text
A B C <EOS>
```

Decoder input:

```text
<SOS> A B C
```

Expected output:

```text
A B C <EOS>
```

So:

```text
Given <SOS>
→ predict A

Given true A
→ predict B

Given true B
→ predict C

Given true C
→ predict <EOS>
```

---

# 🧠 2. Why Is It Needed?

Without teacher forcing, suppose the correct target is:

```text
A B C D
```

but the model predicts:

```text
X
```

If `X` becomes the next decoder input:

```text
Wrong Prediction
↓
Wrong Next Input
↓
Different Hidden State
↓
More Errors
```

Early in training, this can make learning unstable.

Teacher forcing keeps the decoder on:

> the correct target trajectory.

---

# ⭐ 3. Core Intuition

Think of a student solving a multi-step problem.

If step 1 is wrong, the teacher can say:

> “Use the correct answer for the next step.”

That is exactly what teacher forcing does.

```text
Model predicts wrong token
↓
Training still supplies correct previous token
↓
Next prediction is learned from correct history
```

---

# 📐 4. Mathematical View

Seq2Seq models:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

With full teacher forcing, training uses:

# `P(y_t^* | y_<t^*, X)`

where `*` means:

> ground truth.

Token loss:

`L_t = -log P(y_t^* | y_<t^*, X)`

Sequence loss:

# `L = -Σ_t log P(y_t^* | y_<t^*, X)`

---

# 🔄 5. Target Shifting

Suppose target is:

```text
A B C <EOS>
```

Decoder input:

```text
<SOS> A B C
```

Labels:

```text
A B C <EOS>
```

Alignment:

```text
Input:   <SOS>   A   B   C
Target:    A     B   C  <EOS>
```

The decoder still learns:

> previous tokens → next token.

---

# ⚠️ 6. Teacher Forcing Does Not Give the Current Answer

At step `t`:

### Input

`y_(t-1)`

### Target

`y_t`

So:

```text
Previous Correct Token
↓
Predict Current Token
```

Not:

```text
Current Correct Token
↓
Predict Itself
```

---

# 🧠 7. Why It Makes Training Easier

With teacher forcing:

```text
Correct Prefix
↓
Predict Next Token
```

The model gets cleaner training contexts.

Benefits:

* faster convergence
* more stable decoder training
* less early error drift
* better next-token learning signal

---

# 🚨 8. The Main Drawback

Training:

```text
Ground-Truth Previous Tokens
```

Inference:

```text
Model-Generated Previous Tokens
```

So:

# **Training conditions ≠ inference conditions**

This creates:

# **Exposure Bias**

---

# ⭐ 9. What Is Exposure Bias?

Exposure bias means:

> the model is trained mostly on correct target histories but must operate on its own imperfect generated histories during inference.

Training prefixes:

```text
<SOS>
<SOS> A
<SOS> A B
<SOS> A B C
```

Inference may produce:

```text
<SOS> A X
```

The model may have seen very few such prefixes during training.

---

# 🔄 10. Exposure Bias vs Error Propagation

These are related but not identical.

### Exposure Bias

Cause:

> training uses ground-truth histories.

### Error Propagation

Effect:

```text
Wrong Generated Token
↓
Wrong Next History
↓
Different State
↓
Potential Future Errors
```

Mental model:

```text
Exposure Bias
↓
Poor preparation for own mistakes
↓
Error propagation may become more damaging
```

---

# 🧠 11. Training vs Inference

## Training

```text
<SOS>
↓
predict y1
↓
feed TRUE y1
↓
predict y2
```

## Inference

```text
<SOS>
↓
predict ŷ1
↓
feed ŷ1
↓
predict ŷ2
```

This distinction is one of the most important things to remember.

---

# 📐 12. Teacher Forcing Ratio

Let:

# `p_TF`

be the probability of feeding the ground-truth previous token.

For example:

`p_TF = 0.8`

roughly means:

```text
80%
→ ground truth

20%
→ model prediction
```

depending on implementation.

---

# ⭐ 13. Full Teacher Forcing

If:

# `p_TF = 1.0`

then:

> every applicable decoder step uses the true previous token.

Pros:

* easiest training
* stable histories

Cons:

* strongest train/inference mismatch

---

# ⭐ 14. No Teacher Forcing

If:

# `p_TF = 0`

then:

> decoder always receives its own previous prediction.

Pros:

* inference-like histories

Cons:

* difficult early training
* more error compounding
* less stable optimization

---

# 🧠 15. Scheduled Sampling

A common idea:

> start with high teacher forcing and gradually reduce it.

Example:

```text
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

```text
Early
→ easier learning

Later
→ more exposure to own predictions
```

This is commonly called:

# **Scheduled Sampling**

---

# ⚠️ 16. Scheduled Sampling Is Not Perfect

It may help reduce exposure bias, but:

* training behavior changes
* mixed histories can complicate learning
* discrete predictions are awkward for gradient flow
* it does not guarantee perfect inference robustness

So:

> mitigation ≠ complete solution.

---

# 🧠 17. Teacher Forcing vs Target Shifting

They are related, but not identical.

### Target Shifting

Defines:

```text
Input:
<SOS> y1 y2

Target:
y1    y2 <EOS>
```

### Teacher Forcing

Decides:

> whether those decoder inputs come from the ground truth or the model's own predictions.

---

# 🔗 18. Teacher Forcing vs Autoregressive Decoding

Teacher forcing:

> training input strategy.

Autoregressive decoding:

> generation behavior where previous generated tokens condition future predictions.

The model still learns:

# `P(y_t | y_<t, X)`

Teacher forcing does not remove autoregression.

---

# 🔗 19. Teacher Forcing vs Greedy Decoding

Teacher forcing:

```text
Training
→ use true previous token
```

Greedy decoding:

```text
Inference
→ choose argmax token
→ feed it back
```

Different stages.

---

# 🔗 20. Teacher Forcing vs Beam Search

Teacher forcing:

> training strategy.

Beam Search:

> inference sequence-search strategy.

Do not mix them.

---

# 🔗 21. Teacher Forcing vs Context Bottleneck

### Context Bottleneck

Problem:

```text
Source Information
→ Decoder
```

### Teacher Forcing

Problem:

```text
Previous Target Token
→ Next Decoder Step
```

So:

```text
Context Bottleneck
→ source-access problem

Teacher Forcing
→ decoder training-history strategy
```

---

# 📐 22. Does Teacher Forcing Change the Loss?

Usually no.

Loss is still typically:

> next-token cross-entropy.

Teacher forcing changes:

> decoder input history.

So:

```text
Teacher Forcing
→ input strategy

Cross Entropy
→ training objective
```

---

# 🔄 23. Does BPTT Still Happen?

Yes.

Teacher forcing does not remove recurrent dependencies.

Gradients still flow through:

```text
Loss
↓
Output Layer
↓
Decoder
↓
Encoder Connection
↓
Encoder
```

BPTT still applies.

---

# ⚠️ 24. Argmax Gradient Nuance

If the model selects:

```text
argmax
```

to choose a token and feeds it back, that discrete token selection is generally:

> non-differentiable.

Standard training does not backpropagate through argmax as if it were a smooth function.

---

# 🧠 25. Future Target Leakage?

Teacher forcing does **not** mean the decoder can use future targets.

Correct:

```text
Predict y3
using
y1, y2
```

Wrong:

```text
Predict y3
using
y4
```

Teacher forcing only uses:

> previous ground-truth target tokens.

---

# ⭐ 26. Recurrent Decoder Parallelism Nuance

Teacher forcing may allow us to prepare the whole shifted target sequence in advance.

But in an RNN decoder:

```text
d1 → d2 → d3 → d4
```

hidden states are still sequential.

So:

> teacher forcing does not remove recurrent sequential computation.

---

# 🧠 27. Modern LLM Connection

Modern autoregressive language models are also trained using:

> correct previous tokens to predict the next token.

Example:

```text
I
→ love

I love
→ deep

I love deep
→ learning
```

During generation:

> the model must use its own generated history.

So the underlying train/generation distinction still exists.

---

# 🚨 28. Teacher-Forced Accuracy Can Be Misleading

A model may have:

```text
high next-token accuracy
under ground-truth prefixes
```

but poor:

```text
free-running generation quality
```

because deployment uses:

> self-generated prefixes.

Therefore realistic evaluation should include:

> actual autoregressive generation.

---

# 🧠 29. Problem Taxonomy

Keep these separate:

```text
Vanishing Gradient
→ gradient propagation

Context Bottleneck
→ source accessibility

Teacher Forcing
→ training history strategy

Exposure Bias
→ training/inference history mismatch

Autoregressive Decoding
→ inference feedback loop

Beam Search
→ sequence search
```

---

# 🚫 30. High-Yield Traps

### ❌ Teacher forcing gives the current answer

No.

It gives the previous correct token.

### ❌ Teacher forcing is used mainly at inference

No.

Training.

### ❌ Teacher forcing removes autoregression

No.

### ❌ Teacher forcing allows future target tokens

No.

### ❌ Target shifting = teacher forcing

Related, not identical.

### ❌ Teacher forcing changes cross-entropy itself

No.

### ❌ More teacher forcing is always better

No.

### ❌ Zero teacher forcing is always better

No.

### ❌ Scheduled sampling completely fixes exposure bias

No.

### ❌ Exposure bias = overfitting

No.

---

# 📐 31. Formula Flash Card

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Teacher-Forced Conditional

# `P(y_t^* | y_<t^*, X)`

### Token Loss

`L_t = -log P(y_t^* | y_<t^*, X)`

### Total Loss

# `L = -Σ_t log P(y_t^* | y_<t^*, X)`

### Teacher Forcing Ratio

```text
p_TF
```

```text
with probability p_TF
→ use ground truth

with probability 1 - p_TF
→ use prediction
```

---

# 🎤 32. 30-Second Interview Answer

> **Teacher forcing is a training strategy for autoregressive sequence models where the decoder receives the ground-truth previous target token instead of its own previous prediction. This stabilizes training because each next-token prediction is conditioned on a correct target prefix. However, during inference the ground truth is unavailable, so the decoder must use its own generated history. This difference creates exposure bias, where the model may be less robust to mistakes it makes during generation.**

---

# 🧠 10 Things You Must Know Cold

1. **Teacher forcing uses the true previous target token.**
2. **It does not give the current answer.**
3. **It helps stabilize decoder training.**
4. **It keeps training on correct target trajectories.**
5. **Inference uses model-generated history instead.**
6. **That mismatch creates exposure bias.**
7. **Exposure bias and error propagation are related but distinct.**
8. **Teacher forcing ratio can vary from 0 to 1.**
9. **Scheduled sampling mixes ground-truth and predicted histories.**
10. **Teacher forcing is a training strategy, not a decoding or source-access strategy.**

---

# 🧠 Final Mental Model

Without teacher forcing:

```text
<SOS>
↓
wrong prediction
↓
feed wrong token
↓
different state
↓
another possible error
```

With teacher forcing:

```text
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

```text
Correct Prefix
↓
Cleaner Learning
```

Cost:

```text
Training
→ Correct History

Inference
→ Self-Generated History

↓
Exposure Bias
```

And remember the trade-off:

```text
More Teacher Forcing
→ easier training
→ larger mismatch

Less Teacher Forcing
→ harder training
→ more inference-like experience
```

---

# ⭐ Golden Rule

> **Teacher forcing teaches next-token prediction under the correct target history, making training easier—but inference has no teacher, so the model must survive its own generated history.**
