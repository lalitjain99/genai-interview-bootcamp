# 💡 Assignment — Teacher Forcing

> **Goal:** Apply teacher forcing concepts by tracing decoder inputs, identifying train/inference mismatches, reasoning about exposure bias, evaluating teacher forcing ratios, and diagnosing autoregressive error propagation.

---

# 🧩 Task 1 — Identify Teacher Forcing

Suppose the correct target sequence is:

```text id="tf-assignment-1"
A B C
```

At step 1, the model predicts:

```text id="tf-assignment-2"
X
```

During training, the next decoder input is still:

```text id="tf-assignment-3"
A
```

What training strategy is being used?

### ✅ Answer

# **Teacher Forcing**

The decoder receives:

> the ground-truth previous target token

instead of:

> its own previous prediction.

---

# 🔄 Task 2 — Build the Shifted Target

Target sequence:

```text id="tf-assignment-4"
A B C <EOS>
```

Create:

1. Decoder input
2. Training labels

### ✅ Answer

Decoder input:

```text id="tf-assignment-5"
<SOS> A B C
```

Labels:

```text id="tf-assignment-6"
A B C <EOS>
```

Alignment:

```text id="tf-assignment-7"
Input:   <SOS>   A   B   C
Target:    A     B   C  <EOS>
```

---

# 🧠 Task 3 — What Is Actually Forced?

At timestep `t`, which token is supplied through teacher forcing?

### Option A

`y_t`

### Option B

`y_(t-1)`

### Option C

`y_(t+1)`

### ✅ Answer

# Option B

Teacher forcing supplies:

`y_(t-1)`

to help predict:

`y_t`.

It does not reveal the current or future answer.

---

# 🚨 Task 4 — Diagnose the Wrong Implementation

A developer trains the decoder as:

```text id="tf-assignment-8"
Input:
A B C

Target:
A B C
```

What is wrong?

### ✅ Answer

The decoder is being asked to predict:

> the same token it is currently receiving.

Correct next-token training should use a shift:

```text id="tf-assignment-9"
Input:
<SOS> A B

Target:
A B C
```

Teacher forcing is based on:

> previous correct token → next token.

---

# 🧠 Task 5 — Why Teacher Forcing Helps

Correct sequence:

```text id="tf-assignment-10"
A B C D
```

Early in training the model predicts:

```text id="tf-assignment-11"
X
```

What can happen without teacher forcing?

### ✅ Answer

```text id="tf-assignment-12"
X
↓
fed back into decoder
↓
different hidden state
↓
another wrong prediction
↓
trajectory drifts
```

Teacher forcing avoids this by using:

> true `A`

as the next decoder input.

---

# 🔄 Task 6 — Trace Full Teacher Forcing

Correct target:

```text id="tf-assignment-13"
A B C <EOS>
```

Model predictions are:

```text id="tf-assignment-14"
X Y Z Q
```

What decoder inputs are used under 100% teacher forcing?

### ✅ Answer

```text id="tf-assignment-15"
Step 1 input = <SOS>
Step 2 input = A
Step 3 input = B
Step 4 input = C
```

The model's wrong predictions:

`X, Y, Z`

do not become later decoder inputs.

---

# 📐 Task 7 — Probability Interpretation

Fill the blank.

With full teacher forcing, training estimates:

`P(y_t^* | __________, X)`

### ✅ Answer

# `y_<t^*`

So:

`P(y_t^* | y_<t^*, X)`

The decoder predicts the correct next token conditioned on:

> the correct target prefix.

---

# 🧮 Task 8 — Token Loss

Suppose the correct token probability is:

`0.8`

What is the token loss?

### ✅ Answer

`L_t = -log(0.8)`

Approximately:

# `0.223`

using natural logarithm.

The important intuition:

> higher correct-token probability → lower loss.

---

# 🧮 Task 9 — Sequence Loss

Suppose valid target-token losses are:

```text id="tf-assignment-16"
0.2
0.4
0.1
0.3
```

Using sum reduction, calculate sequence loss.

### ✅ Answer

`0.2 + 0.4 + 0.1 + 0.3`

# `= 1.0`

---

# 🧠 Task 10 — Does Teacher Forcing Change Cross-Entropy?

A developer says:

> “Teacher forcing uses a special loss instead of cross-entropy.”

Correct?

### ✅ Answer

No.

Teacher forcing changes:

> what decoder input history is used.

Cross-entropy remains the usual:

> next-token training objective.

So:

```text id="tf-assignment-17"
Teacher Forcing
→ input strategy

Cross Entropy
→ loss/objective
```

---

# 🚨 Task 11 — Identify Exposure Bias

Training:

```text id="tf-assignment-18"
true y1
→ true y2
→ true y3
```

Inference:

```text id="tf-assignment-19"
predicted y1
→ predicted y2
→ predicted y3
```

What problem does this mismatch create?

### ✅ Answer

# **Exposure Bias**

The model is trained mostly under:

> ground-truth histories

but deployed under:

> self-generated histories.

---

# 🧠 Task 12 — Exposure Bias vs Error Propagation

Classify each statement.

### A

The model rarely sees its own incorrect histories during training.

### B

One incorrect generated token changes future decoder states.

### ✅ Answer

```text id="tf-assignment-20"
A → Exposure Bias
B → Error Propagation
```

They are related but distinct.

---

# 🔄 Task 13 — Error Cascade

Correct output:

```text id="tf-assignment-21"
A B C D
```

Inference produces:

```text id="tf-assignment-22"
A X ...
```

Explain why `X` can affect later predictions.

### ✅ Answer

At the next decoder step:

```text id="tf-assignment-23"
X
↓
target embedding
↓
decoder update
↓
different hidden state
↓
different next-token probabilities
```

So an early error can propagate through future decoding.

---

# 📐 Task 14 — Teacher Forcing Ratio

Suppose:

`p_TF = 0.75`

What does this mean?

### ✅ Answer

Approximately:

```text id="tf-assignment-24"
75% of applicable decisions
→ use ground-truth previous token

25%
→ use model-generated previous token
```

Exact sampling may be:

* per timestep
* per sequence
* per batch

depending on implementation.

---

# 🧮 Task 15 — Sampling Decision

Teacher forcing ratio:

`0.6`

Random sample:

`r = 0.42`

Rule:

```text id="tf-assignment-25"
if r < p_TF
→ teacher force
```

What happens?

### ✅ Answer

Since:

`0.42 < 0.6`

use:

# **ground-truth previous token**

---

# 🧮 Task 16 — Another Sampling Decision

Same ratio:

`p_TF = 0.6`

Now:

`r = 0.85`

What happens?

### ✅ Answer

Since:

`0.85 > 0.6`

use:

# **model-generated previous token**

---

# ⭐ Task 17 — Compare 100% vs 0%

Fill the table.

| Property             | 100% Teacher Forcing | 0% Teacher Forcing |
| -------------------- | -------------------- | ------------------ |
| Training history     | ?                    | ?                  |
| Early stability      | ?                    | ?                  |
| Inference similarity | ?                    | ?                  |

### ✅ Answer

| Property             | 100% Teacher Forcing | 0% Teacher Forcing |
| -------------------- | -------------------- | ------------------ |
| Training history     | Ground truth         | Model generated    |
| Early stability      | Usually high         | Often lower        |
| Inference similarity | Lower                | Higher             |

Core trade-off:

```text id="tf-assignment-26"
More guidance
vs
more inference realism
```

---

# 🧠 Task 18 — Scheduled Sampling

A training schedule is:

```text id="tf-assignment-27"
Epoch 1  → 1.0
Epoch 5  → 0.8
Epoch 10 → 0.5
Epoch 20 → 0.2
```

What strategy does this resemble?

### ✅ Answer

# **Scheduled Sampling**

The model receives:

> more guidance early

and:

> more self-generated history later.

---

# 🚨 Task 19 — Is Scheduled Sampling a Perfect Fix?

### ✅ Answer

No.

It can reduce the mismatch, but may introduce:

* optimization complexity
* mixed-history behavior
* discrete sampling issues
* objective inconsistencies

So:

> scheduled sampling is a mitigation, not a guaranteed solution.

---

# 🧠 Task 20 — Future Leakage Check

When predicting:

`y_4`

which of these is valid under teacher forcing?

### Option A

Use:

`y_1, y_2, y_3`

### Option B

Use:

`y_1, y_2, y_3, y_5`

### ✅ Answer

# Option A

Teacher forcing only uses:

> previous target tokens.

Using `y_5` would be:

> future target leakage.

---

# 🔗 Task 21 — Classify the Concept

Match each item.

### A. Teacher Forcing

### B. Attention

### C. Beam Search

### D. Context Vector Bottleneck

with:

### 1. Source-access architecture

### 2. Training target-history strategy

### 3. Sequence search during inference

### 4. Fixed source summary limitation

### ✅ Answer

```text id="tf-assignment-28"
A → 2
B → 1
C → 3
D → 4
```

---

# 🧠 Task 22 — Teacher Forcing vs Target Shifting

Which statement is correct?

### Option A

They are exactly the same thing.

### Option B

Target shifting creates previous-token/next-token alignment, while teacher forcing determines whether the previous token comes from ground truth.

### ✅ Answer

# Option B

This distinction is important.

---

# 🔄 Task 23 — Does BPTT Still Apply?

Teacher forcing is enabled.

Does the decoder still require BPTT?

### ✅ Answer

Yes.

The recurrent hidden states still depend sequentially on prior hidden states:

```text id="tf-assignment-29"
d1 → d2 → d3 → d4
```

Teacher forcing changes:

> token input source

not:

> recurrent state dependency.

---

# 🧠 Task 24 — Batched Teacher Forcing

Target:

```text id="tf-assignment-30"
A B C <EOS>
```

Can the shifted target inputs:

```text id="tf-assignment-31"
<SOS> A B C
```

be prepared for the entire batch before forward computation?

### ✅ Answer

Yes.

But for an RNN decoder:

> hidden-state computations remain sequential.

Teacher forcing does not make recurrent steps independent.

---

# 🚨 Task 25 — Argmax Gradient

Suppose during self-fed training:

```text id="tf-assignment-32"
logits
↓
argmax
↓
token ID
↓
next decoder input
```

Can ordinary backpropagation smoothly propagate through the argmax selection?

### ✅ Answer

No.

Argmax is a discrete:

> non-differentiable operation.

This makes learning through hard token decisions more complicated.

---

# 🧠 Task 26 — Diagnose Suspicious Validation Metrics

Model reports:

```text id="tf-assignment-33"
Teacher-forced token accuracy = 96%
```

But free-running generation is poor.

What might be happening?

### ✅ Answer

The model is very good at:

```text id="tf-assignment-34"
correct prefix
→ next token
```

but not necessarily good at:

```text id="tf-assignment-35"
self-generated imperfect prefix
→ next token
```

So teacher-forced metrics may overestimate:

> real autoregressive generation quality.

---

# ⭐ Task 27 — Improve Evaluation

What evaluation would better reflect production behavior?

### ✅ Answer

Run actual autoregressive decoding:

```text id="tf-assignment-36"
<BOS>
↓
model prediction
↓
feed prediction back
↓
continue
↓
<EOS>
```

Then evaluate the:

> complete generated sequence.

This tests the model under inference-like history.

---

# 🚨 Task 28 — Production Diagnosis

A recurrent translation model:

```text id="tf-assignment-37"
Training:
Teacher forcing = 100%

Production:
one wrong token often causes entire sentence to deteriorate
```

What issue should you investigate?

### ✅ Answer

Investigate:

# **exposure bias / training–inference mismatch**

The model may have insufficient experience operating under:

> its own erroneous histories.

---

# 🧠 Task 29 — Wrong Proposed Fix

A developer says:

> “Let's make the encoder bidirectional. That should solve exposure bias.”

Correct?

### ✅ Answer

No.

Bidirectional encoding changes:

> source representation.

Exposure bias concerns:

> target-history inputs during training vs inference.

Different problem.

---

# 🚫 Task 30 — Another Wrong Proposed Fix

A developer says:

> “Let's use Beam Search during training to fix teacher forcing.”

Correct?

### ✅ Answer

Not as a direct fix.

Beam Search is primarily:

> inference-time sequence search.

Teacher forcing concerns:

> training-time decoder history.

---

# ✅ True / False

| #  | Statement                                                                                                  | Answer  |
| -- | ---------------------------------------------------------------------------------------------------------- | ------- |
| 1  | Teacher forcing uses the true previous target token.                                                       | ✅ True  |
| 2  | Teacher forcing gives the current target token as input to predict itself.                                 | ❌ False |
| 3  | Teacher forcing can stabilize early decoder training.                                                      | ✅ True  |
| 4  | Teacher forcing removes autoregression from the model.                                                     | ❌ False |
| 5  | Teacher forcing usually changes decoder input history rather than the loss function.                       | ✅ True  |
| 6  | 100% teacher forcing creates no train/inference mismatch.                                                  | ❌ False |
| 7  | Exposure bias is caused by differing training and inference history distributions.                         | ✅ True  |
| 8  | Exposure bias and overfitting are identical.                                                               | ❌ False |
| 9  | One generated error can change later decoder states.                                                       | ✅ True  |
| 10 | Teacher forcing can use future target tokens.                                                              | ❌ False |
| 11 | A teacher forcing ratio can lie between 0 and 1.                                                           | ✅ True  |
| 12 | Scheduled sampling can gradually reduce teacher forcing.                                                   | ✅ True  |
| 13 | Scheduled sampling guarantees elimination of exposure bias.                                                | ❌ False |
| 14 | RNN hidden-state computation remains sequential under teacher forcing.                                     | ✅ True  |
| 15 | Free-running evaluation better reflects autoregressive inference than teacher-forced token accuracy alone. | ✅ True  |

---

# ⭐ Staff Engineer Challenge — Production Translation Model

## Scenario

Your production model is:

```text id="tf-assignment-38"
BiLSTM Encoder
↓
Attention
↓
LSTM Decoder
```

Training:

```text id="tf-assignment-39"
Teacher Forcing Ratio = 1.0
```

Observed:

```text id="tf-assignment-40"
Training loss
→ excellent

Teacher-forced validation accuracy
→ excellent

Free generation
→ repeated phrases
→ early termination
→ severe deterioration after first mistake
```

---

## Question 1 — Is attention likely to eliminate this issue?

### ✅ Answer

No.

Attention improves:

> source information accessibility.

The observed problem is more related to:

> decoder history mismatch.

A model can have excellent attention and still suffer from exposure bias.

---

## Question 2 — Why are teacher-forced metrics potentially optimistic?

### ✅ Answer

Because the model receives:

> correct previous tokens.

Production generation receives:

> self-generated previous tokens.

Therefore teacher-forced evaluation tests an easier history distribution.

---

## Question 3 — What experiment would you run first?

### ✅ Answer

Compare:

```text id="tf-assignment-41"
Teacher-forced evaluation
vs
free-running autoregressive evaluation
```

Then measure:

* sequence quality
* repetition
* `<EOS>` behavior
* recovery after deliberate errors
* degradation vs output length

---

## Question 4 — How would you test recovery ability directly?

### ✅ Answer

Force one wrong token into the decoder history at a controlled position.

Then observe:

```text id="tf-assignment-42"
Does model recover?
or
Does entire remaining sequence collapse?
```

Repeat at:

* early positions
* middle positions
* late positions

This gives a useful robustness profile.

---

## Question 5 — What training change might you test?

### ✅ Answer

Possible experiment:

```text id="tf-assignment-43"
Start with high teacher forcing
↓
gradually reduce it
↓
expose decoder to own predictions
```

For example:

* scheduled sampling
* lower fixed teacher forcing ratio

while carefully monitoring training stability.

---

## Question 6 — What trade-off do you expect?

### ✅ Answer

Reducing teacher forcing may improve:

> inference robustness

but can hurt:

* convergence speed
* training stability
* token accuracy

So:

```text id="tf-assignment-44"
High TF
→ clean learning
→ bigger mismatch

Lower TF
→ harder learning
→ more realistic histories
```

---

## Question 7 — What would make you reject a lower teacher forcing schedule?

### ✅ Answer

If it causes:

* unstable training
* significantly worse sequence metrics
* worse termination
* more repetition
* slower convergence without inference benefit

then the schedule is not useful merely because it is:

> more inference-like.

The correct choice must be validated empirically.

---

# 🎯 Final Applied Exercise

Complete the full reasoning chain:

```text id="tf-assignment-45"
Autoregressive Decoder
↓
Model Must Predict __________
↓
Early Training Predictions Are Often __________
↓
Feeding Them Back Causes __________
↓
Teacher Forcing Instead Feeds __________
↓
Training Becomes More __________
↓
But Inference Uses __________
↓
This Creates __________
↓
One Wrong Token Can Cause __________
↓
A Possible Mitigation Is __________
```

### ✅ Answer

```text id="tf-assignment-46"
Autoregressive Decoder
↓
Model Must Predict NEXT TARGET TOKEN
↓
Early Training Predictions Are Often WRONG
↓
Feeding Them Back Causes TRAJECTORY DRIFT / COMPOUNDING ERRORS
↓
Teacher Forcing Instead Feeds GROUND-TRUTH PREVIOUS TOKEN
↓
Training Becomes More STABLE
↓
But Inference Uses MODEL-GENERATED PREVIOUS TOKENS
↓
This Creates EXPOSURE BIAS
↓
One Wrong Token Can Cause ERROR PROPAGATION
↓
A Possible Mitigation Is SCHEDULED SAMPLING / LOWER TEACHER FORCING
```

---

# 🧠 Final Assignment Mental Model

Teacher forcing solves one problem:

```text id="tf-assignment-47"
Weak Model
↓
Bad Predictions
↓
Bad Training History
```

by doing:

```text id="tf-assignment-48"
Correct Previous Token
↓
Cleaner Decoder State
↓
Better Next-Token Learning
```

But creates another:

```text id="tf-assignment-49"
TRAINING
Ground-Truth History

≠

INFERENCE
Self-Generated History

↓
Exposure Bias
```

And once inference begins:

```text id="tf-assignment-50"
Wrong Token
↓
Different History
↓
Different Hidden State
↓
Potential Error Cascade
```

So always remember:

# **Teacher forcing improves learning conditions, but does not reproduce deployment conditions.**

---

# ⭐ Golden Rule

> **When evaluating teacher forcing, ask two separate questions: does it make next-token learning easier, and does the resulting decoder remain robust when the teacher disappears and the model must condition on its own mistakes?**
