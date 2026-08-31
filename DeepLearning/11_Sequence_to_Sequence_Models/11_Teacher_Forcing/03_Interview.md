# 🎤 Interview Questions — Teacher Forcing

> **Goal:** Test whether you understand what teacher forcing is, why it makes Seq2Seq training easier, how it differs from inference, why it creates exposure bias, and how scheduled sampling tries to reduce the mismatch.

---

# 🟢 Beginner

## Q1. What is teacher forcing?

### ✅ Answer

Teacher forcing is a training strategy where the decoder receives:

> the ground-truth previous target token

instead of:

> its own previous prediction.

Example:

```text id="tf-int-1"
Correct target:
A B C

Model predicts:
X ...
```

With teacher forcing:

```text id="tf-int-2"
Model predicted X

but next decoder input = true A
```

So the decoder continues training on:

> the correct target history.

---

## Q2. Why is teacher forcing used?

### ✅ Answer

Without teacher forcing, one early prediction error can become the next decoder input and push the model further away from the correct sequence.

Example:

```text id="tf-int-3"
Wrong Prediction
↓
Wrong Next Input
↓
Different Hidden State
↓
Another Possible Error
```

Teacher forcing avoids this during training by supplying:

> the correct previous target token.

This usually makes training:

* more stable
* faster
* easier to optimize

---

## Q3. Does teacher forcing give the model the current correct answer?

### ✅ Answer

No.

At decoder step `t`:

```text id="tf-int-4"
Input:
y_(t-1)

Target:
y_t
```

So teacher forcing gives:

> the previous correct token

not:

> the current token being predicted.

The model still performs next-token prediction.

---

## Q4. How is teacher forcing related to target shifting?

### ✅ Answer

Suppose target sequence is:

```text id="tf-int-5"
A B C <EOS>
```

Decoder input:

```text id="tf-int-6"
<SOS> A B C
```

Expected output:

```text id="tf-int-7"
A B C <EOS>
```

Target shifting defines:

> which previous token is aligned with which next-token target.

Teacher forcing determines:

> whether the decoder actually receives the ground-truth previous token during training.

They are related, but not exactly the same concept.

---

# 🟡 Intermediate

## Q5. What is the main drawback of teacher forcing?

### ✅ Answer

The main drawback is:

# **training–inference mismatch**

During training:

```text id="tf-int-8"
Ground-truth previous tokens
→ decoder
```

During inference:

```text id="tf-int-9"
Model-generated previous tokens
→ decoder
```

So the model trains under cleaner histories than it experiences at deployment.

This mismatch is commonly called:

# **Exposure Bias**

---

## Q6. What is exposure bias?

### ✅ Answer

Exposure bias means:

> the model is trained mostly on ground-truth prefixes but must operate on self-generated prefixes during inference.

During training:

```text id="tf-int-10"
<SOS>
<SOS> A
<SOS> A B
<SOS> A B C
```

During inference, the model may encounter:

```text id="tf-int-11"
<SOS> A X
```

where `X` is its own mistake.

The model may not have learned how to recover well from such histories.

---

## Q7. What is the difference between exposure bias and error propagation?

### ✅ Answer

They are related but different.

### Exposure Bias

Cause:

> training and inference use different history distributions.

### Error Propagation

Effect:

```text id="tf-int-12"
Wrong Generated Token
↓
Wrong Next Input
↓
Different Hidden State
↓
Potential More Errors
```

So:

```text id="tf-int-13"
Exposure Bias
→ model has less practice with its own mistakes

Error Propagation
→ those mistakes can compound during inference
```

---

## Q8. What is a teacher forcing ratio?

### ✅ Answer

Teacher forcing ratio controls how often the ground-truth previous token is used during training.

Let:

`p_TF`

be the teacher forcing probability.

For example:

`p_TF = 0.8`

roughly means:

```text id="tf-int-14"
80%
→ use ground-truth previous token

20%
→ use model prediction
```

The exact sampling granularity can vary by implementation.

---

## Q9. What happens with 100% teacher forcing?

### ✅ Answer

If:

# `p_TF = 1.0`

then every applicable decoder step uses:

> the correct previous target token.

Advantages:

* stable training
* faster convergence
* clean histories

Disadvantage:

> maximum training–inference mismatch.

---

## Q10. What happens with zero teacher forcing?

### ✅ Answer

If:

# `p_TF = 0`

the decoder always receives:

> its own previous prediction.

This is closer to inference conditions.

But early in training it may cause:

* unstable trajectories
* compounding errors
* harder optimization
* slower convergence

So zero teacher forcing is not automatically better.

---

# 🔴 Advanced

## Q11. How does teacher forcing appear mathematically in the training objective?

### ✅ Answer

Seq2Seq factorizes:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

With full teacher forcing, training conditions on:

> the ground-truth prefix.

So:

# `P(y_t^* | y_<t^*, X)`

Token loss:

`L_t = -log P(y_t^* | y_<t^*, X)`

Total loss:

# `L = -Σ_t log P(y_t^* | y_<t^*, X)`

with padding positions masked as needed.

---

## Q12. Does teacher forcing change the loss function?

### ✅ Answer

Usually no.

Teacher forcing changes:

> the decoder input history.

The objective is still commonly:

> next-token cross-entropy / negative log-likelihood.

So:

```text id="tf-int-15"
Teacher Forcing
→ input strategy

Cross Entropy
→ objective
```

Do not confuse the two.

---

## Q13. What is scheduled sampling?

### ✅ Answer

Scheduled sampling is a strategy that mixes:

* ground-truth previous tokens
* model-generated previous tokens

during training.

A common intuition is:

```text id="tf-int-16"
Early training
→ high teacher forcing

Later training
→ lower teacher forcing
```

This tries to balance:

> stable learning

with:

> exposure to inference-like histories.

It can reduce the mismatch, but it is not a perfect solution.

---

## Q14. Why is self-feeding predicted tokens harder to train through?

### ✅ Answer

If the model selects the next token using:

```text id="tf-int-17"
argmax
```

or discrete sampling, that selection is generally:

> non-differentiable.

Standard backpropagation does not treat the discrete token selection as a smooth differentiable operation.

So feeding model predictions introduces additional optimization complications.

---

## Q15. Why can teacher-forced validation accuracy look good while free-running generation is poor?

### ✅ Answer

Teacher-forced validation gives the model:

> correct previous tokens.

Free-running inference gives it:

> its own previous predictions.

So a model may be excellent at:

```text id="tf-int-18"
correct prefix
→ next token
```

but weak at:

```text id="tf-int-19"
imperfect self-generated prefix
→ recovery
```

Therefore realistic evaluation should include:

> actual autoregressive decoding.

---

# ⭐ Staff Engineer Challenge

## Scenario

You are training a recurrent Seq2Seq model for translation.

Training setup:

```text id="tf-int-20"
Teacher Forcing Ratio = 1.0
```

Offline metrics:

```text id="tf-int-21"
Teacher-forced token accuracy = very high
Training loss = low
```

But production generation shows:

* repeated phrases
* early `<EOS>`
* severe degradation after one wrong token
* locally fluent but globally incorrect sequences

The team debates several fixes.

---

## Question 1. What would you investigate first?

### ✅ Answer

I would investigate:

> the training–inference mismatch.

With 100% teacher forcing, the decoder is trained only on:

> ground-truth histories.

But production uses:

> self-generated histories.

So the model may not be robust when its own predictions deviate from the correct sequence.

This is a classic exposure-bias pattern.

---

## Question 2. Why can a single wrong token cause such a large degradation?

### ✅ Answer

During inference:

```text id="tf-int-22"
Wrong y_t
↓
becomes next decoder input
↓
changes decoder state
↓
changes probability distribution
↓
future errors may compound
```

The decoder operates autoregressively, so generated history directly influences future states.

---

## Question 3. Would increasing hidden size solve this?

### ✅ Answer

Not directly.

Increasing hidden size may improve model capacity.

But the core issue is:

> mismatch between training histories and inference histories.

A larger decoder still may have poor experience recovering from self-generated errors.

---

## Question 4. Would attention solve exposure bias?

### ✅ Answer

Not directly.

Attention improves:

> source information access.

Exposure bias concerns:

> target-history distribution.

So a model may have both:

* excellent attention
* exposure bias

at the same time.

---

## Question 5. Would Beam Search solve exposure bias?

### ✅ Answer

Not fundamentally.

Beam Search changes:

> how candidate sequences are explored during inference.

It may improve final output quality, but it does not change the fact that training relied heavily on:

> ground-truth prefixes.

So it addresses:

> search

not:

> training-history mismatch.

---

## Question 6. What training change could you experiment with?

### ✅ Answer

Possible experiments include:

* reducing teacher forcing ratio
* gradually reducing teacher forcing
* scheduled sampling
* evaluating with self-fed decoding
* explicitly measuring robustness after forced decoding errors

The goal is:

> expose the decoder to more inference-like histories without destabilizing training too much.

---

## Question 7. What trade-off would you monitor if reducing teacher forcing?

### ✅ Answer

I would monitor:

### Training Stability

Does optimization become noisier or slower?

### Free-Running Quality

Does autoregressive generation improve?

### Recovery Ability

Can the decoder recover after a wrong token?

### Sequence Metrics

Do BLEU / task-specific metrics improve?

### Termination Behavior

Does `<EOS>` timing become more stable?

The key trade-off is:

```text id="tf-int-23"
More teacher forcing
→ easier training
→ larger mismatch

Less teacher forcing
→ harder training
→ more inference-like experience
```

---

# ⚡ Rapid-Fire Traps

### ❌ “Teacher forcing gives the decoder the current target token.”

False.

It gives the previous target token.

---

### ❌ “Teacher forcing is an inference strategy.”

False.

Training strategy.

---

### ❌ “Teacher forcing removes autoregression.”

False.

The decoder still learns:

`P(y_t | y_<t, X)`.

---

### ❌ “Teacher forcing allows future target tokens.”

False.

Only valid previous targets are used.

---

### ❌ “Teacher forcing and target shifting are exactly the same.”

False.

Related but distinct.

---

### ❌ “Teacher forcing changes cross-entropy into another loss.”

False.

It changes input history.

---

### ❌ “100% teacher forcing is always best.”

False.

It can maximize train/inference mismatch.

---

### ❌ “0% teacher forcing is always best.”

False.

Training may become unstable.

---

### ❌ “Scheduled sampling completely solves exposure bias.”

False.

---

### ❌ “Exposure bias is the same as overfitting.”

False.

---

### ❌ “Attention fixes exposure bias.”

False.

Different problem.

---

### ❌ “Beam Search fixes exposure bias.”

False.

Different problem.

---

# 📐 Formula Flash Card

### Seq2Seq Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Teacher-Forced Training

# `P(y_t^* | y_<t^*, X)`

### Token Loss

`L_t = -log P(y_t^* | y_<t^*, X)`

### Sequence Loss

# `L = -Σ_t log P(y_t^* | y_<t^*, X)`

### Teacher Forcing Ratio

# `p_TF`

```text id="tf-int-24"
with probability p_TF
→ ground truth

with probability 1-p_TF
→ model prediction
```

---

# 🎤 30-Second Interview Answer

> **Teacher forcing is a training strategy for autoregressive sequence models where the decoder receives the ground-truth previous target token rather than its own previous prediction. This makes optimization easier because each next-token prediction is conditioned on a clean target history. However, inference has no ground truth, so the decoder must use its own generated history. That mismatch creates exposure bias, and errors can then propagate autoregressively. A common mitigation is to reduce teacher forcing or use scheduled sampling, although this does not perfectly eliminate the problem.**

---

# 🧠 Final Interview Mental Model

Think of the same decoder in two environments.

## Training

```text id="tf-int-25"
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

Environment:

> clean target history.

## Inference

```text id="tf-int-26"
<SOS>
↓
predict ŷ1
↓
feed ŷ1
↓
predict ŷ2
```

Environment:

> self-generated history.

The central problem:

```text id="tf-int-27"
Training History
≠
Inference History
↓
Exposure Bias
```

And once an error occurs:

```text id="tf-int-28"
Wrong Token
↓
Different History
↓
Different Hidden State
↓
Potential Error Cascade
```

---

# ⭐ Golden Rule

> **Teacher forcing solves an optimization problem by giving the decoder clean target histories during training, but that convenience creates a deployment problem: at inference time the model must survive the consequences of its own predictions.**
