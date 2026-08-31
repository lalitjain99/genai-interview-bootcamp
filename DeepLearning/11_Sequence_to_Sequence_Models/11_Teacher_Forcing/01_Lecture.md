# 📘 Lecture — Teacher Forcing

## 🌟 Introduction

In the previous lecture, we studied the **Context Vector Bottleneck**.

That problem was about:

# **How source information reaches the decoder**

Now we move to a different question.

Suppose the encoder has already produced the source representation.

The decoder now has to generate:

```text
y1, y2, y3, ..., yT
```

one token at a time.

At decoder step 1, we can start with:

`<SOS>`

But after that, what should we feed into the decoder?

Suppose the correct target sequence is:

```text
A B C <EOS>
```

At training time, we already know the correct answer.

So after predicting `A`, should the decoder receive:

```text
the correct A
```

or:

```text
whatever the model predicted
```

?

That choice leads to one of the most important training ideas in Seq2Seq:

# **Teacher Forcing**

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to:

* Explain what teacher forcing is.
* Understand why it was introduced.
* Explain how decoder training works with shifted targets.
* Distinguish teacher forcing from ordinary autoregressive inference.
* Understand why teacher forcing makes training easier.
* Understand the training–inference mismatch it creates.
* Explain exposure bias.
* Understand teacher forcing ratio.
* Understand scheduled sampling conceptually.
* Distinguish teacher forcing from context bottleneck, greedy decoding, and beam search.
* Understand when teacher forcing can help and where it can hurt.

---

# 📖 Part 1 — Revisit Decoder Training

Suppose source is:

```text
"I love AI"
```

Target is:

```text
"J'aime l'IA"
```

For training, target becomes:

```text
J'aime l'IA <EOS>
```

Decoder input is shifted:

```text
<SOS> J'aime l'IA
```

Expected output:

```text
J'aime l'IA <EOS>
```

So:

```text
Decoder Input:
<SOS>    J'aime    l'IA

Target:
J'aime   l'IA      <EOS>
```

This means:

```text
Given <SOS>
→ predict J'aime

Given J'aime
→ predict l'IA

Given l'IA
→ predict <EOS>
```

Now comes the key question.

At the second step, what token do we actually feed?

---

# 🧠 Part 2 — Two Possible Training Strategies

Suppose the correct first target token is:

`A`

but the model predicts:

`X`

We now want to predict the next token:

`B`.

We have two options.

## Option 1 — Feed Model Prediction

```text
Model predicted X
↓
feed X into next decoder step
```

Then:

```text
X
↓
predict B
```

## Option 2 — Feed Ground Truth

Even though the model predicted `X`, we ignore that for the next input and instead feed:

```text
A
```

the correct previous target token.

That is:

# **Teacher Forcing**

---

# ⭐ Part 3 — Definition of Teacher Forcing

Teacher forcing means:

> During decoder training, the ground-truth previous target token is used as the next decoder input instead of the model's own previous prediction.

Example:

Correct target:

```text
A B C
```

Model prediction:

```text
X ...
```

With teacher forcing:

```text
Model predicts X
but
next decoder input = true A
```

So the model gets:

> the correct history during training.

---

# 🧠 Part 4 — Why Is It Called “Teacher Forcing”?

Think of a student solving a multi-step problem.

Suppose:

```text
Step 1 answer
→ wrong
```

If the teacher allows the student to continue using that wrong result:

```text
all later steps may become wrong
```

Instead, the teacher says:

> “Your answer was wrong. For the next step, use the correct answer.”

That is exactly the intuition.

```text
Model predicts wrong token
↓
Teacher supplies correct previous token
↓
Training continues from correct history
```

Hence:

# **Teacher Forcing**

---

# 📖 Part 5 — Why Was Teacher Forcing Needed?

Imagine training without teacher forcing.

Correct target:

```text
A B C D
```

Model initially predicts:

```text
X
```

Now decoder receives:

`X`

instead of:

`A`

Then it predicts:

```text
Y
```

Then:

```text
Z
```

Very quickly, the decoder may operate on completely incorrect history:

```text
<SOS> → X → Y → Z
```

while the correct sequence is:

```text
<SOS> → A → B → C
```

This makes early training much harder.

---

# ⭐ Part 6 — The Problem of Compounding Errors During Training

Early in training, the model is weak.

Its predictions are often wrong.

Without teacher forcing:

```text
Wrong Prediction
↓
Wrong Next Input
↓
Wrong Hidden State
↓
Another Wrong Prediction
↓
More Wrong Inputs
```

The model can drift far away from the correct target trajectory.

So the learning signal becomes:

> harder to interpret and noisier.

Teacher forcing prevents that drift during training.

---

# 🧠 Part 7 — Teacher Forcing Stabilizes the Training Trajectory

With teacher forcing:

```text
<SOS>
↓
predict y1
↓
feed true y1
↓
predict y2
↓
feed true y2
↓
predict y3
```

Even if predictions are wrong:

```text
predicted token
≠
correct token
```

the next decoder state is still built from:

> the correct target history.

So every training step learns from a more meaningful context.

---

# 📐 Part 8 — Mathematical View

Let target sequence be:

`Y = (y_1, y_2, ..., y_T)`

The decoder models:

# `P(y_t | y_<t, X)`

With full teacher forcing during training, the history:

`y_<t`

contains:

> ground-truth previous target tokens.

So training optimizes:

# `P(y_t^* | y_1^*, ..., y_(t-1)^*, X)`

where `*` denotes ground truth.

This is important.

The model is trained under:

> correct target prefixes.

---

# 🧠 Part 9 — Training Example Step by Step

Suppose target:

```text
<SOS> A B C <EOS>
```

## Step 1

Input:

`<SOS>`

Expected:

`A`

Model predicts distribution:

```text
A → 0.6
X → 0.3
...
```

Loss computed against:

`A`

---

## Step 2

Teacher forcing feeds:

# `A`

even if model predicted `X`.

Input:

`A`

Expected:

`B`

---

## Step 3

Feed true:

`B`

Expected:

`C`

---

## Step 4

Feed true:

`C`

Expected:

`<EOS>`

So:

```text
Ground-Truth Decoder Inputs
↓
Next-Token Prediction
↓
Loss
```

---

# ⭐ Part 10 — Teacher Forcing Does Not Mean Giving the Current Answer

This is a common confusion.

At step `t`, we do **not** feed:

`y_t`

and ask the model to predict:

`y_t`.

Instead:

```text
input = y_(t-1)
target = y_t
```

So the model still performs:

> next-token prediction.

Teacher forcing only supplies:

> the correct previous token.

---

# 📖 Part 11 — Target Shift Revisited

Suppose:

```text
Target:
A B C <EOS>
```

Training decoder input:

```text
<SOS> A B C
```

Training label:

```text
A B C <EOS>
```

The input sequence is essentially:

> the target shifted right by one position.

This is how teacher forcing is usually implemented in batched training.

---

# 🧠 Part 12 — Does Teacher Forcing Mean One Token at a Time in Code?

Conceptually:

```text
feed y_(t-1)
predict y_t
```

But in many implementations, the entire shifted target sequence can be prepared as:

```text
<SOS> y1 y2 ... y_(T-1)
```

and passed through the decoder during training.

However, in a recurrent decoder:

> hidden-state computation remains sequential across time.

Teacher forcing does not magically remove recurrence.

---

# 📐 Part 13 — Loss with Teacher Forcing

For each position:

`L_t = -log P(y_t^* | y_<t^*, X)`

Total loss:

# `L = Σ_t L_t`

or a mean over valid target positions.

Padding positions are masked.

So the model learns:

> next-token prediction under correct prefixes.

---

# ⭐ Part 14 — Why Teacher Forcing Makes Optimization Easier

Without teacher forcing, when the model predicts a wrong token:

> future decoder states are based on an incorrect history.

Then the model is trying to learn from states it created because of its own mistakes.

With teacher forcing:

```text
Correct Prefix
↓
Predict Next Token
```

each next-token learning problem is better conditioned.

This usually results in:

* faster convergence
* more stable training
* better early-stage learning

---

# 🧠 Part 15 — But There Is a Catch

Training:

```text
correct previous tokens
```

Inference:

```text
model's own previous predictions
```

These are not the same.

So we train under one distribution of decoder histories...

but deploy under another.

That creates:

# **Training–Inference Mismatch**

---

# 🚨 Part 16 — The Core Mismatch

During training:

```text
<SOS>
↓
true y1
↓
true y2
↓
true y3
```

During inference:

```text
<SOS>
↓
predicted y1
↓
predicted y2
↓
predicted y3
```

Training assumes:

> clean prefixes.

Inference may contain:

> imperfect prefixes.

The model may not have been exposed enough to its own mistakes during training.

---

# ⭐ Part 17 — Exposure Bias

This phenomenon is commonly called:

# **Exposure Bias**

Meaning:

> the model is heavily exposed to ground-truth histories during training, but at inference it must operate on histories generated by itself.

So:

```text
Training
→ clean / expert history

Inference
→ self-generated history
```

That difference can hurt robustness.

---

# 📖 Part 18 — Exposure Bias Example

Correct sequence:

```text
A B C D
```

During training, model sees prefixes:

```text
<SOS>
<SOS> A
<SOS> A B
<SOS> A B C
```

But during inference, it may encounter:

```text
<SOS> A X
```

or:

```text
<SOS> Y
```

Those prefixes may be unusual or unseen during training.

So the model may not know:

> how to recover after its own mistakes.

---

# 🧠 Part 19 — Why One Wrong Prediction Can Cascade

Suppose model predicts:

```text
A X
```

instead of:

```text
A B
```

Now the decoder receives:

`X`

at the next inference step.

That changes:

* input embedding
* hidden state
* next-token probability distribution

So:

```text
Wrong Token
↓
Different Decoder State
↓
Different Next Distribution
↓
Potential Additional Error
```

This is:

> autoregressive error propagation.

---

# 🔗 Part 20 — Exposure Bias vs Context Bottleneck

These are different.

## Context Bottleneck

```text
Source
→ Decoder
```

problem.

Question:

> Can decoder access source information effectively?

## Exposure Bias

```text
Previous Target
→ Next Decoder Step
```

problem.

Question:

> Can decoder handle histories different from the ground truth?

Keep them separate.

---

# ⭐ Part 21 — Full Teacher Forcing

If every decoder step uses the true previous token:

# `teacher forcing ratio = 1.0`

or:

# `100% teacher forcing`

Then:

```text
Always ground truth
→ next decoder input
```

This usually makes training easiest.

But it also maximizes:

> training–inference mismatch.

---

# 🧠 Part 22 — No Teacher Forcing

If:

# `teacher forcing ratio = 0`

then decoder always receives:

> its own previous prediction.

Training resembles inference more closely.

But especially early in training:

> learning can become unstable or difficult.

---

# 📐 Part 23 — Teacher Forcing Ratio

We do not necessarily need:

```text
0% or 100%
```

We can use a probability:

`p_TF`

At each applicable step:

```text
with probability p_TF
→ feed ground truth

with probability 1-p_TF
→ feed model prediction
```

For example:

`p_TF = 0.7`

means roughly:

```text
70% → ground truth
30% → model prediction
```

depending on implementation.

---

# 🧮 Part 24 — Simple Example

Suppose:

`teacher forcing ratio = 0.5`

At one step, draw:

`r = 0.31`

If rule is:

`r < 0.5`

then:

> use ground truth.

Next step:

`r = 0.82`

Then:

> use model prediction.

So training can mix both behaviors.

---

# ⭐ Part 25 — Why Mix Ground Truth and Predictions?

The intuition is:

```text
Ground Truth
→ easier/stable learning

Own Predictions
→ practice inference-like conditions
```

So mixed training attempts to balance:

# **learning stability**

with:

# **inference robustness**

---

# 📖 Part 26 — Scheduled Sampling Intuition

A common idea is:

> use more teacher forcing early, then gradually reduce it.

Example:

```text
Epoch 1
→ 100%

Epoch 5
→ 80%

Epoch 10
→ 50%

Epoch 20
→ 20%
```

This is often associated with:

# **Scheduled Sampling**

The intuition:

```text
Early Training
→ model is weak
→ provide more guidance

Later Training
→ model is stronger
→ expose it more to own predictions
```

---

# ⚠️ Part 27 — Scheduled Sampling Is Not a Perfect Solution

It sounds intuitive, but it introduces its own challenges.

For example:

* training objective changes
* sampled discrete predictions can complicate optimization
* mixed histories may differ from ideal probabilistic training assumptions

So:

> scheduled sampling is a mitigation strategy, not a perfect theoretical solution.

For interview purposes, know:

* why it exists
* what problem it tries to reduce
* that it is not guaranteed to fully solve exposure bias

---

# 🧠 Part 28 — Does Teacher Forcing Change the Loss?

Usually the loss is still:

> next-token cross-entropy.

Teacher forcing changes:

> what previous token is fed into the decoder.

It does **not** fundamentally mean a different token loss.

So:

```text
Teacher Forcing
→ input strategy

Cross Entropy
→ objective
```

Do not confuse them.

---

# ⭐ Part 29 — Teacher Forcing vs Target Shifting

Related, but not identical.

## Target Shifting

Defines:

```text
decoder input
vs
decoder prediction target
```

Example:

```text
Input:
<SOS> A B

Target:
A B <EOS>
```

## Teacher Forcing

Decides:

> whether ground-truth previous target is actually used as decoder input during training.

In full batched training they often appear together, but conceptually they are distinct.

---

# 🧠 Part 30 — Teacher Forcing vs Autoregressive Decoding

Teacher forcing:

```text
Training
→ true previous token
```

Autoregressive inference:

```text
Inference
→ predicted previous token
```

So teacher forcing does not remove autoregression from the model.

The decoder still models:

# `P(y_t | y_<t, X)`

The difference is only:

> where `y_<t` comes from during training.

---

# 🔗 Part 31 — Teacher Forcing vs Greedy Decoding

Teacher forcing is:

> a training technique.

Greedy decoding is:

> an inference strategy.

Greedy:

```text
choose argmax token
↓
feed it back
```

Teacher forcing:

```text
ignore model prediction for next input
↓
feed correct previous token
```

Different stages.

---

# 🔗 Part 32 — Teacher Forcing vs Beam Search

Beam Search also belongs to:

> inference.

It keeps multiple candidate target sequences.

Teacher forcing belongs to:

> training.

So:

```text
Teacher Forcing
→ training input strategy

Greedy / Beam Search
→ inference decoding strategy
```

---

# 🧠 Part 33 — Does Teacher Forcing Solve Context Bottleneck?

No.

Teacher forcing changes:

```text
target-history input
```

It does not change:

```text
source representation
```

So a model can have:

* context bottleneck
* exposure bias

at the same time.

---

# ⭐ Part 34 — What Happens to Gradients?

Suppose at step `t`, decoder input embedding comes from:

> ground-truth token `y_(t-1)`.

Loss at step `t` still backpropagates through:

* decoder parameters
* recurrent state
* output layer
* encoder connection

Teacher forcing does not stop BPTT.

It changes the forward input history.

---

# 📐 Part 35 — Important Gradient Nuance

If instead the model uses a discrete predicted token:

```text
argmax
```

as next input, the discrete selection itself is generally:

> non-differentiable.

Standard recurrent training does not backpropagate through an argmax token choice as though it were a smooth operation.

This is one reason self-fed training can be more complicated conceptually.

---

# 🧠 Part 36 — Does the Ground-Truth Token Receive Gradient?

The token ID itself does not receive a gradient.

But its embedding is looked up from:

`E_tgt`

and the embedding parameters can be learned through normal backpropagation.

So:

```text
Ground-Truth Token ID
↓
Embedding Lookup
↓
Decoder
↓
Loss
```

The target embedding matrix can still receive gradients.

---

# ⭐ Part 37 — Full Batched Teacher Forcing

Suppose target:

```text
<BOS> A B C
```

Input embeddings:

```text
e(<BOS>)
e(A)
e(B)
e(C)
```

Labels:

```text
A
B
C
<EOS>
```

This can be arranged across a batch before forward computation.

However, for an RNN decoder:

```text
d1 → d2 → d3 → d4
```

remains sequential.

---

# 🧠 Part 38 — Does Teacher Forcing Mean Future Target Leakage?

Important nuance.

At decoder step `t`, using:

`y_(t-1)`

is legitimate.

Using:

`y_(t+1)`

would be future leakage.

Teacher forcing supplies:

> previous ground-truth targets,

not future ones.

So the autoregressive direction remains valid.

---

# 🚨 Part 39 — Future Leakage Mistake

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

Teacher forcing does not authorize:

> future target access.

---

# 📖 Part 40 — Example: Machine Translation

Target:

```text
Je suis heureux <EOS>
```

Training decoder input:

```text
<BOS> Je suis heureux
```

Labels:

```text
Je suis heureux <EOS>
```

Suppose decoder incorrectly predicts:

```text
Tu
```

for the first token.

With full teacher forcing, second-step input is still:

```text
Je
```

not:

```text
Tu
```

This keeps training on the correct target trajectory.

---

# 🧠 Part 41 — Example Without Teacher Forcing

Same model predicts:

```text
Tu
```

Without teacher forcing:

```text
Tu
↓
next decoder state
↓
next prediction
```

So the model begins exploring:

> its own generated trajectory.

This resembles inference more closely, but may be difficult when the model is immature.

---

# ⭐ Part 42 — Why Training Accuracy Can Look Misleading

Suppose teacher-forced validation reports excellent next-token accuracy.

But inference quality is poor.

Why?

Because during teacher-forced evaluation, the model repeatedly gets:

> correct history.

At deployment, it must survive:

> its own imperfect history.

So teacher-forced token accuracy may overestimate:

> actual autoregressive sequence robustness.

---

# 🧠 Part 43 — Evaluate Under Real Decoding

For realistic evaluation, eventually run:

```text
Source
↓
Encoder
↓
<BOS>
↓
Model Prediction
↓
Feed Prediction Back
↓
...
↓
<EOS>
```

Then evaluate complete outputs using task-appropriate metrics.

For translation, this might include:

* BLEU
* chrF
* COMET-like metrics in modern systems
* human evaluation

The key concept:

> evaluation should reflect inference conditions.

---

# 🚨 Part 44 — Exposure Bias Is Not the Same as Overfitting

Overfitting means:

> model performs much better on training distribution/data than unseen data.

Exposure bias means:

> model trains on ground-truth histories but infers on self-generated histories.

They can both occur, but they are different ideas.

---

# 🧠 Part 45 — Exposure Bias Is Also Not Exactly Error Propagation

Related, but distinct.

### Exposure Bias

Cause:

> mismatch between histories seen during training and inference.

### Error Propagation

Effect:

> one wrong generated token changes future states and can create more errors.

So:

```text
Exposure Bias
↓
model may be poorly prepared for mistakes
↓
Error Propagation becomes more damaging
```

---

# ⭐ Part 46 — Why Teacher Forcing Was So Useful Historically

Recurrent Seq2Seq decoders were difficult to train from scratch.

Teacher forcing allowed models to learn:

```text
correct prefix
→ correct next token
```

before asking them to handle:

> fully self-generated histories.

It made recurrent sequence training significantly more practical.

---

# 🔗 Part 47 — Teacher Forcing in Modern Models

The core idea still appears in modern autoregressive sequence training.

For example, decoder-only language models commonly train using:

> ground-truth previous tokens to predict the next token.

Conceptually:

```text
Ground Truth Prefix
→ Predict Next Token
```

This is closely related to teacher-forced autoregressive training.

Important nuance:

> modern Transformer training can compute many target positions in parallel because causal masking prevents future-token access.

That is different from recurrent decoders, whose hidden states are inherently sequential.

---

# 🧠 Part 48 — Teacher Forcing and GPT Intuition

Suppose text:

```text
I love deep learning
```

Training pairs conceptually are:

```text
I
→ love

I love
→ deep

I love deep
→ learning
```

The model is trained using:

> the correct prefix.

During generation:

```text
I
→ model predicts token
→ model uses generated token
→ continues
```

So the training–generation distinction survives even in modern autoregressive LLMs.

---

# ⭐ Part 49 — But Terminology Can Differ

In RNN Seq2Seq literature, the term:

# **Teacher Forcing**

is common.

In modern language-model training, people may more often say:

* next-token prediction
* autoregressive maximum likelihood training
* shifted labels

even though ground-truth prefixes are being used.

So do not get stuck on terminology.

Understand the underlying mechanism.

---

# 📐 Part 50 — Full Probability View

Seq2Seq objective:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

Training with teacher forcing estimates each term using:

> ground-truth prefix `y_<t^*`.

Loss:

# `L = -Σ_t log P(y_t^* | y_<t^*, X)`

This is the mathematical core.

---

# 🧠 Part 51 — Why This Objective Is Convenient

The ground-truth sequence provides:

> the correct conditioning history for every target position.

Therefore each target token produces a clean supervised learning signal.

Instead of requiring the model to first generate an entire correct prefix, we directly teach:

```text
correct prefix
→ next correct token
```

This greatly simplifies learning.

---

# ⚠️ Part 52 — But Objective and Generation Behavior Are Different

Training optimizes local conditional predictions under:

> ground-truth prefixes.

Inference quality depends on:

> complete self-generated trajectories.

Therefore:

```text
Good token-level likelihood
≠
guaranteed perfect sequence generation
```

This is another important interview point.

---

# ⭐ Part 53 — Teacher Forcing Ratio Trade-Off

High teacher forcing:

### Pros

* faster convergence
* stable training
* clean decoder histories

### Cons

* stronger train/inference mismatch
* less experience recovering from own mistakes

Low teacher forcing:

### Pros

* more inference-like histories
* more practice with own predictions

### Cons

* unstable early training
* compounding errors
* harder optimization

---

# 🧠 Part 54 — Practical Training Strategy

A practical recurrent Seq2Seq strategy might be:

```text
Early epochs
→ high teacher forcing

Later epochs
→ lower teacher forcing
```

But exact schedules depend on:

* task
* architecture
* dataset
* optimization behavior

There is no universal best ratio.

---

# 🚨 Part 55 — Do Not Assume Teacher Forcing Ratio Must Be Per Token

Implementation can vary.

A decision may be made:

* per timestep
* per sequence
* per batch
* according to a schedule

So if discussing implementation:

> specify the sampling granularity.

---

# 🧠 Part 56 — Production Failure Example

Suppose a chatbot response decoder performs:

```text
Teacher-forced validation accuracy = 94%
```

but during free generation:

* repeats phrases
* gets stuck after errors
* terminates early

A likely diagnostic question is:

> Was the model evaluated mostly with ground-truth prefixes rather than self-generated prefixes?

This is a classic training–inference mismatch signal.

---

# 🔗 Part 57 — Our Sequence Model Problem Map

We can now distinguish:

```text
Vanishing / Exploding Gradients
→ gradient propagation

LSTM / GRU
→ recurrent state improvement

Bidirectional RNN
→ source directionality

Context Vector Bottleneck
→ encoder-decoder source access

Teacher Forcing
→ decoder training history

Exposure Bias
→ train/inference history mismatch

Autoregressive Decoding
→ inference token feedback

Beam Search
→ target sequence search

Attention
→ dynamic source access
```

This map is extremely important.

---

# 🎤 Part 58 — 30-Second Interview Answer

> **Teacher forcing is a training strategy for autoregressive sequence models where the decoder is given the ground-truth previous target token instead of its own previous prediction. This makes training more stable because each next-token prediction is conditioned on the correct target history. However, during inference the ground truth is unavailable, so the decoder must condition on its own generated tokens. This creates a training–inference mismatch known as exposure bias, where the model may not be well prepared to recover from its own mistakes.**

---

# 🎤 Part 59 — Why Use Teacher Forcing?

Strong answer:

> **Without teacher forcing, especially early in training, one wrong decoder prediction becomes the next input and can push all later decoder states away from the correct sequence. Teacher forcing keeps the model on the ground-truth trajectory, giving cleaner next-token learning signals and typically making optimization faster and more stable.**

---

# 🎤 Part 60 — What Is Exposure Bias?

Strong answer:

> **Exposure bias is the mismatch caused because an autoregressive model is trained mostly on ground-truth prefixes but at inference must operate on its own generated prefixes. As a result, it may encounter erroneous histories during inference that it was rarely exposed to during training.**

---

# 🎤 Part 61 — Teacher Forcing vs Scheduled Sampling

Strong answer:

> **Teacher forcing always or frequently feeds the ground-truth previous token. Scheduled sampling gradually or probabilistically mixes ground-truth tokens with model-generated tokens so the model is exposed to more inference-like histories during training. It attempts to reduce exposure bias, though it is not a perfect solution.**

---

# 🚫 Part 62 — High-Yield Traps

### ❌ Teacher forcing gives the current correct token to predict itself

No.

It gives the **previous** correct token.

---

### ❌ Teacher forcing is an inference technique

No.

It is primarily a training strategy.

---

### ❌ Teacher forcing eliminates autoregression

No.

The model still predicts:

`P(y_t | y_<t, X)`.

---

### ❌ Teacher forcing means future target tokens can be used

No.

Only previous target history is valid.

---

### ❌ Target shifting and teacher forcing are exactly the same thing

Related, but conceptually distinct.

---

### ❌ Teacher forcing changes cross-entropy into a different loss

No.

It changes decoder inputs, not necessarily the loss function.

---

### ❌ High teacher forcing has no downside

It can worsen training–inference mismatch.

---

### ❌ No teacher forcing is always better

Early training can become much harder.

---

### ❌ Scheduled sampling completely solves exposure bias

No.

---

### ❌ Context bottleneck and exposure bias are the same issue

No.

---

### ❌ Beam Search fixes exposure bias during training

No.

Beam Search is an inference search strategy.

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

`p_TF`

```text
with probability p_TF
→ ground-truth previous token

with probability 1-p_TF
→ model-generated previous token
```

---

# 🧠 Part 63 — 10 Things You Must Know Cold

1. **Teacher forcing uses the true previous target token during training.**
2. **It does not give the model the current answer.**
3. **It stabilizes autoregressive decoder training.**
4. **It prevents early mistakes from immediately corrupting all later training inputs.**
5. **Training uses correct prefixes; inference uses generated prefixes.**
6. **That mismatch is called exposure bias.**
7. **Exposure bias and error propagation are related but different.**
8. **Teacher forcing ratio can range from 0 to 1.**
9. **Scheduled sampling mixes ground-truth and predicted histories.**
10. **Teacher forcing is a training strategy, not a source-access or decoding-search solution.**

---

# 🧠 Final Mental Model

Without teacher forcing:

```text
Correct Target:
A → B → C → D

Model:
<SOS>
  ↓
  X
  ↓
  Y
  ↓
  Z

One early error
↓
whole trajectory drifts
```

With teacher forcing:

```text
<SOS>
↓
predict A

feed TRUE A
↓
predict B

feed TRUE B
↓
predict C

feed TRUE C
↓
predict <EOS>
```

This makes training easier.

But then:

```text
TRAINING
Ground-Truth History
↓
Model

INFERENCE
Model-Generated History
↓
Model
```

which creates:

# **Exposure Bias**

So the central trade-off is:

```text
More Teacher Forcing
→ easier training
→ larger train/inference mismatch

Less Teacher Forcing
→ harder training
→ more inference-like experience
```

---

# 🚀 Where We Go Next

Teacher forcing has exposed a new question.

During inference:

> there is no teacher.

The decoder must generate:

```text
<BOS>
↓
prediction
↓
feed prediction back
↓
prediction
↓
...
↓
<EOS>
```

How exactly does this inference loop work?

What determines when decoding stops?

Why can one error affect everything afterward?

What is the difference between logits, probability distribution, token selection, and feedback?

That leads directly to our next lecture:

# `12_Inference_And_Autoregressive_Decoding`

The progression is now:

```text
Seq2Seq
↓
Context Vector Bottleneck
↓
Teacher Forcing
↓
Training–Inference Mismatch
↓
Autoregressive Inference ← NEXT
↓
Greedy Decoding
↓
Beam Search
↓
Attention Motivation
```

---

# ⭐ Golden Rule

> **Teacher forcing makes Seq2Seq training easier by teaching next-token prediction from the correct target history, but the same convenience creates exposure bias because inference must operate on the model's own generated history.**
