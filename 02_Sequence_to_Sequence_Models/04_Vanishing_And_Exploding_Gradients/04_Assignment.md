# 💡 Assignment — Vanishing and Exploding Gradients

> **Goal:** Apply the concepts of gradient stability to realistic RNN training situations.
> This assignment focuses on **diagnosis, calculation, debugging, and architecture decisions** rather than repeating interview definitions.

---

# 🧩 Task 1 — Predict the Gradient Behavior

Suppose an RNN has an effective backward multiplier of approximately:

`0.7`

per time step.

A gradient must travel backward across:

`8` time steps.

### Questions

1. What is the approximate gradient multiplier after 8 steps?
2. Is this showing vanishing or exploding behavior?
3. What happens if the sequence becomes much longer?

### ✅ Answer

We calculate:

`0.7^8`

`≈ 0.0576`

So only about:

`5.76%`

of the original gradient magnitude remains.

This indicates:

> **vanishing-gradient tendency**

If the sequence becomes much longer:

```text
0.7^T → increasingly small
```

So early time steps receive weaker learning signals.

---

# 💥 Task 2 — Diagnose an Exploding Gradient

During training, you observe the following global gradient norms:

```text
Epoch 1 → 3.2
Epoch 2 → 4.1
Epoch 3 → 8.7
Epoch 4 → 55
Epoch 5 → 7,400
Epoch 6 → NaN
```

### Questions

1. What is the likely problem?
2. What practical technique would you try immediately?
3. Would increasing the hidden-state size solve this problem?

### ✅ Answer

### 1. Likely Problem

The gradient norm is increasing rapidly:

```text
3.2 → 4.1 → 8.7 → 55 → 7400
```

This strongly suggests:

> **exploding gradients**

### 2. Immediate Mitigation

A common practical technique is:

> **gradient clipping**

especially gradient norm clipping.

For example:

```text
max_gradient_norm = 5
```

If the gradient norm exceeds `5`, scale it down before the optimizer step.

### 3. Increase Hidden Size?

No.

Increasing hidden size increases:

> model capacity

but does not directly solve unstable gradient propagation.

---

# 🧠 Task 3 — Long-Term Dependency Failure

You train a vanilla RNN on sequences like:

```text
START_ACCOUNT_TYPE = "Premium"

... 300 time steps ...

Predict whether premium rules apply
```

The model performs well when the relevant information occurs within the last 10 steps, but poorly when it occurs 200–300 steps earlier.

### Question

What does this behavior strongly suggest?

### ✅ Answer

It suggests difficulty learning:

> **long-term dependencies**

A likely reason is vanishing gradients.

The final loss must propagate backward through hundreds of recurrent transformations:

```text
Loss
↓
h300
↓
h299
↓
...
↓
h1
```

If the gradient shrinks repeatedly, the early steps receive almost no learning signal.

So the model learns nearby relationships but struggles with distant ones.

---

# 🔢 Task 4 — Compare Two Recurrent Dynamics

Consider two simplified scalar RNNs.

### Model A

Each backward step multiplies the gradient by:

`0.8`

### Model B

Each backward step multiplies it by:

`1.2`

Assume a path of:

`20` steps.

### Questions

1. Calculate the approximate multiplier for Model A.
2. Calculate the approximate multiplier for Model B.
3. Diagnose each model.

### ✅ Answer

### Model A

`0.8^20 ≈ 0.0115`

So only about:

`1.15%`

of the original gradient remains.

Diagnosis:

> vanishing-gradient tendency.

---

### Model B

`1.2^20 ≈ 38.34`

The gradient is amplified more than:

`38×`

Diagnosis:

> exploding-gradient tendency.

---

# 🔥 Task 5 — Activation Saturation

Suppose an RNN uses:

`tanh`

and during training many pre-activation values become:

```text
a_t = 7
a_t = 8
a_t = -9
```

### Questions

1. What happens to `tanh(a_t)`?
2. What happens to `tanh'(a_t)`?
3. Why is this problematic for BPTT?

### ✅ Answer

For large positive or negative values:

```text
tanh(a_t) ≈ +1 or -1
```

The activation is:

> saturated.

Since:

`tanh'(z) = 1 - tanh²(z)`

when:

`tanh(z) ≈ ±1`

then:

`tanh'(z) ≈ 0`

During BPTT, these very small derivatives are repeatedly multiplied into the gradient.

That contributes to:

> vanishing gradients.

---

# 🔍 Task 6 — Find the Faulty Explanation

A developer says:

> “The model has vanishing gradients because every value in `W_hh` is below 1.”

### What is incomplete about this explanation?

### ✅ Answer

`W_hh` is a:

> matrix

not a single scalar.

A matrix can:

* amplify some directions
* shrink other directions
* rotate representations

Also, recurrent gradient propagation depends on more than:

`W_hh`

It includes activation derivatives as well.

The more precise explanation is:

> gradient stability depends on the repeated product of recurrent Jacobians.

For a tanh RNN, each Jacobian contains something like:

`diag(tanh'(a_t)) W_hh`

So looking at individual matrix elements alone is insufficient.

---

# 🧮 Task 7 — Gradient Clipping by Norm

Suppose a gradient vector is:

```text
g = [6, 8]
```

and the maximum gradient norm allowed is:

`5`

### Questions

1. What is the original norm?
2. What scale factor should be applied?
3. What is the clipped gradient?

### ✅ Answer

Original norm:

`||g|| = sqrt(6² + 8²)`

`= sqrt(36 + 64)`

`= 10`

Maximum allowed norm:

`5`

Scale factor:

`5 / 10 = 0.5`

Therefore:

```text
g_clipped = [3, 4]
```

and:

`||g_clipped|| = 5`

The direction remains unchanged.

---

# ⚠️ Task 8 — Will Gradient Clipping Help?

For each situation, decide whether gradient clipping is likely to help.

### Situation A

Gradient norm:

`80,000`

### Situation B

Gradient norm:

`0.00000003`

### Situation C

Gradient norm:

`120`

with sudden loss spikes.

### ✅ Answer

### A

✅ Yes.

This is extremely large.

Gradient clipping can reduce the update magnitude.

---

### B

❌ No.

The gradient is already tiny.

Clipping does not amplify small gradients.

---

### C

✅ Likely useful.

The combination of large gradients and sudden instability suggests exploding gradients.

---

# 🧠 Task 9 — Capacity vs Learnability

Two engineers disagree.

### Engineer A

> “Our hidden state has 2048 dimensions, so the RNN should easily remember something from 500 steps ago.”

### Engineer B

> “That does not guarantee the model can learn the dependency.”

### Who is more correct?

### ✅ Answer

Engineer B.

A large hidden state provides:

> representational capacity.

But learning long-term dependencies requires successful:

> gradient propagation and credit assignment.

If gradients vanish before reaching early states:

```text
Large Memory Capacity
+
Poor Credit Assignment
=
Long-Term Learning Still Fails
```

So:

> capacity is necessary in some cases, but it is not sufficient.

---

# 🔗 Task 10 — Connect to BPTT

Suppose the final loss is:

`L_100`

and an important state is:

`h_2`

### Question

Write the conceptual gradient path connecting them.

### ✅ Answer

The gradient must propagate:

```text
L_100
↓
h_100
↓
h_99
↓
h_98
↓
...
↓
h_2
```

Mathematically, the gradient includes a product like:

`dL/dh_100`

`× dh_100/dh_99`

`× dh_99/dh_98`

`× ...`

`× dh_3/dh_2`

The large number of repeated multiplications is what creates the risk of:

* vanishing gradients
* exploding gradients

---

# 🔧 Task 11 — Debug a Training Run

You observe:

```text
Training loss → slowly decreases
Validation performance → improves on short sequences
Performance on long sequences → almost unchanged
Early-time-step gradients → ~10^-11
Late-time-step gradients → ~10^-2
```

### Questions

1. What is the strongest diagnosis?
2. Why might looking only at the global loss have hidden the issue?
3. What architecture would you consider next?

### ✅ Answer

### 1. Diagnosis

The strongest diagnosis is:

> vanishing gradients across long temporal paths.

---

### 2. Why Loss Can Hide It

The model can still learn:

> short-range relationships.

Those improvements can reduce the overall training loss.

So a decreasing loss does not guarantee that:

> long-range dependencies are being learned.

---

### 3. Next Architecture

A natural next architecture is:

> LSTM

or:

> GRU

because they introduce mechanisms designed to improve long-term information and gradient flow.

---

# 🧩 Task 12 — Orthogonal Initialization Reasoning

You initialize:

`W_hh`

as an orthogonal matrix.

### Question

Why might this help compared with a randomly scaled recurrent matrix?

### ✅ Answer

Orthogonal transformations approximately preserve vector norms:

`||Wx|| ≈ ||x||`

So initially, the linear recurrent transformation is less likely to:

* strongly shrink signals
* strongly amplify signals

This can improve gradient stability.

However:

> it is not a complete solution.

The recurrent path still includes:

* activation derivatives
* learned changes to `W_hh`
* repeated nonlinear transformations

---

# 🔥 Task 13 — Why Not Simply Use ReLU Everywhere?

A developer proposes:

> “Since `tanh` saturates, let's replace it with ReLU and vanishing gradients disappear.”

### Explain why this is incomplete.

### ✅ Answer

ReLU avoids saturation on its positive side because its derivative is:

`1`

for positive inputs.

That can help gradient flow.

But recurrent use of ReLU introduces other problems.

Repeated recurrent multiplication still exists:

```text
W_hh
×
W_hh
×
W_hh
...
```

So gradients can still:

* shrink
* explode

Also, ReLU neurons can become inactive for negative inputs.

Therefore:

> changing activation alone does not fundamentally solve the recurrent long-term dependency problem.

---

# 🧠 Task 14 — Forward Memory vs Backward Learning

Suppose an RNN's final hidden state actually contains some information originating from:

`t = 1`.

However, gradients reaching `t = 1` during training are nearly zero.

### Can the model still have a long-term learning problem?

### ✅ Answer

Yes.

These are separate questions.

### Forward

Can information survive?

```text
t1
→ hidden states
→ t100
```

### Backward

Can the loss teach the early computation what should be preserved?

```text
Loss
→ ...
→ t1
```

If backward gradients vanish:

> the network may struggle to learn the correct long-term memory policy.

Therefore:

```text
Forward Retention
≠
Backward Credit Assignment
```

---

# 🧮 Task 15 — Learning Rate vs Exploding Gradient

Suppose:

`gradient = 100,000`

### Model A

Learning rate:

`0.01`

### Model B

Learning rate:

`0.000001`

### Questions

1. What approximate update magnitude does each produce?
2. Does Model B actually solve the exploding-gradient problem?

### ✅ Answer

Ignoring sign:

### Model A

`0.01 × 100,000`

`= 1,000`

Huge update.

---

### Model B

`0.000001 × 100,000`

`= 0.1`

Much smaller update.

So lowering the learning rate can:

> reduce the damage caused by a large gradient.

But the underlying gradient is still:

`100,000`

Therefore:

> Model B reduces the effect but does not solve the underlying exploding-gradient dynamics.

---

# 📊 Task 16 — Interpret Gradient Monitoring

You monitor gradients at four temporal distances from the loss:

| Distance from Loss | Gradient Norm |
| ------------------ | ------------: |
| 1 step             |         `0.8` |
| 10 steps           |        `0.12` |
| 50 steps           |      `0.0004` |
| 100 steps          |   `0.0000003` |

### What does this pattern tell you?

### ✅ Answer

The gradient systematically decreases as temporal distance increases.

This suggests:

> gradient signal is vanishing across the recurrent chain.

It also explains why the model may learn:

* nearby relationships

but struggle with:

* long-range relationships.

---

# 💥 Task 17 — Exploding Gradient or Large Useful Gradient?

During one batch:

`gradient norm = 40`

but the next batches show:

```text
35
38
29
41
33
```

The loss is stable and steadily decreasing.

### Should you immediately conclude that gradients are exploding?

### ✅ Answer

No.

A relatively large gradient does not automatically mean:

> exploding gradient.

Exploding gradients usually involve unstable amplification patterns such as:

```text
10
100
10,000
1e9
NaN
```

Diagnosis should consider:

* trend
* training stability
* loss behavior
* parameter updates
* numerical issues

not just one absolute number.

---

# 🧠 Task 18 — Same Global Gradient, Different Temporal Behavior

Consider two RNN models.

Both have:

`global gradient norm = 5`

### Model A

Gradients remain meaningful throughout the sequence.

### Model B

Almost all gradient comes from the final 5 time steps; early states receive almost zero gradient.

### Question

Are the models equally healthy?

### ✅ Answer

No.

The global norm alone hides:

> where the gradient originates.

Model B may still have a severe:

> long-range vanishing-gradient problem.

This is why debugging sequence models may require examining gradients by:

* layer
* parameter group
* temporal depth

rather than only global norm.

---

# ✅ True / False — Reasoning Traps

| # | Statement | Answer |
|---|---|
| 1 | Vanishing gradients refer to backward learning signals becoming very small. | ✅ True |
| 2 | If gradients vanish, hidden states must also become zero. | ❌ False |
| 3 | Repeated Jacobian multiplication is central to recurrent gradient instability. | ✅ True |
| 4 | `tanh` saturation can contribute to vanishing gradients. | ✅ True |
| 5 | `tanh` is the only cause of vanishing gradients. | ❌ False |
| 6 | Gradient clipping primarily helps exploding gradients. | ✅ True |
| 7 | Gradient clipping makes tiny gradients larger. | ❌ False |
| 8 | Increasing hidden size automatically fixes long-range learning. | ❌ False |
| 9 | Orthogonal initialization can improve initial recurrent gradient stability. | ✅ True |
| 10 | Lowering learning rate changes the underlying recurrent Jacobian product. | ❌ False |
| 11 | Long sequences create deeper temporal computational graphs. | ✅ True |
| 12 | A small gradient always indicates a bug. | ❌ False |
| 13 | A large gradient always means exploding gradients. | ❌ False |
| 14 | Long-term dependency problems involve both memory retention and credit assignment. | ✅ True |
| 15 | Vanishing/exploding gradients can occur outside RNNs too. | ✅ True |

---

# 🧮 Calculation Challenge

Suppose a simplified recurrent gradient multiplier is:

`0.9`

and the dependency spans:

`50` steps.

### Question 1

Estimate:

`0.9^50`

### ✅ Answer

Approximately:

`0.00515`

So only about:

`0.515%`

of the original gradient remains.

---

### Question 2

What if the multiplier were:

`1.1`?

Calculate approximately:

`1.1^50`

### ✅ Answer

Approximately:

`117.39`

The gradient would be amplified by more than:

`117×`

in this simplified scalar example.

---

### Question 3

What lesson does this illustrate?

### ✅ Answer

Values that look close to:

`1`

can still produce dramatic effects when multiplied:

> many times.

That is why recurrent depth matters so much.

---

# ⭐ Staff Engineer Challenge — Long-Horizon Event Model

## Scenario

You are building a risk-detection RNN for financial transactions.

Each customer generates:

`1 transaction every minute`

and some fraud patterns develop over:

`24 hours`.

So the model may need to relate events separated by:

`1,440 time steps`.

During training:

```text
short-range accuracy = strong
long-range accuracy  = weak

gradient norm near t=1440 = 0.4
gradient norm near t=1    = 2 × 10^-12
```

---

## Question 1 — Diagnose the Core Problem

### ✅ Answer

The strongest evidence points to:

> severe vanishing gradients across the long recurrent path.

The final part of the sequence receives useful gradients.

But the earliest states receive almost none.

Therefore long-range credit assignment is failing.

---

## Question 2 — Would Gradient Clipping Be Your Primary Fix?

### ✅ Answer

No.

Gradient clipping limits:

> excessively large gradients.

Here the problem is:

> gradients becoming too small.

Clipping would not restore the vanished learning signal.

---

## Question 3 — Would You Increase Hidden Size First?

### ✅ Answer

Not as the primary fix.

Increasing hidden size addresses:

> representation capacity.

The observed issue is:

> gradient propagation.

So I would first focus on architecture and training dynamics.

---

## Question 4 — What Would You Investigate?

### ✅ Answer

I would inspect:

* gradient norm vs temporal depth
* activation saturation
* recurrent weight initialization
* recurrent Jacobian behavior
* sequence length
* TBPTT configuration if used
* normalization
* learning rate
* whether vanilla RNN is appropriate

I would also compare:

* LSTM
* GRU
* temporal convolutional models
* attention-based models

depending on latency and sequence-length requirements.

---

## Question 5 — Suppose You Use Gradient Clipping and Training Stops Producing NaNs, but Long-Range Accuracy Is Still Poor. Why?

### ✅ Answer

Gradient clipping may have fixed:

> exploding-gradient instability.

But it does not fix:

> vanishing gradients.

So the model can become numerically stable while still being unable to assign credit across long temporal distances.

This is an important distinction:

```text
Stable Training
≠
Successful Long-Term Learning
```

---

## Question 6 — What Architectural Idea Do We Need Next?

### ✅ Answer

We need a recurrent architecture that provides:

> a more stable memory path through time

and lets the network learn:

* what to preserve
* what to forget
* what new information to write
* what information to expose

This leads directly to:

# **LSTM**

---

# 🎯 Final Applied Exercise

A vanilla RNN is trained on sequences of length:

`500`

You observe:

```text
Training loss: decreasing
Gradient norm at t=500: 0.7
Gradient norm at t=400: 0.2
Gradient norm at t=250: 0.003
Gradient norm at t=50: 1e-9
Long-range accuracy: poor
Short-range accuracy: strong
```

Answer the following.

### 1. Is training completely broken?

### ✅ Answer

No.

The model is clearly learning something because:

* training loss decreases
* short-range accuracy is strong

The problem is specifically:

> long-range learning.

---

### 2. What is the likely issue?

### ✅ Answer

Vanishing gradients.

---

### 3. Why is the gradient near `t=500` healthy while near `t=50` it is tiny?

### ✅ Answer

The gradient must cross many more recurrent transformations to reach:

`t=50`.

Repeated recurrent Jacobian multiplication progressively weakens the signal.

---

### 4. Would gradient clipping solve this?

### ✅ Answer

No.

The gradient is already too small.

---

### 5. Would larger hidden size guarantee a fix?

### ✅ Answer

No.

It increases capacity, not necessarily gradient flow.

---

### 6. What architectural direction should be considered?

### ✅ Answer

A gated recurrent architecture such as:

> **LSTM or GRU**

should be evaluated.

---

# 🧠 Final Reasoning Chain

Complete the chain:

```text
Long RNN sequence
↓
BPTT creates __________
↓
Gradients repeatedly multiply by __________
↓
Repeated contraction causes __________
↓
Repeated expansion causes __________
↓
Vanishing hurts __________
↓
Exploding hurts __________
↓
Gradient clipping mainly helps __________
↓
Long-term architectural improvement motivates __________
```

### ✅ Answer

```text
Long RNN sequence
↓
BPTT creates LONG GRADIENT PATHS
↓
Gradients repeatedly multiply by RECURRENT JACOBIANS
↓
Repeated contraction causes VANISHING GRADIENTS
↓
Repeated expansion causes EXPLODING GRADIENTS
↓
Vanishing hurts LONG-RANGE CREDIT ASSIGNMENT
↓
Exploding hurts TRAINING STABILITY
↓
Gradient clipping mainly helps EXPLODING GRADIENTS
↓
Long-term architectural improvement motivates LSTM / GRU
```

---

# 🧠 Final Assignment Mental Model

```text
FORWARD

Important Information
↓
Many Recurrent Steps
↓
Later Prediction
```

Training requires:

```text
BACKWARD

Later Loss
↓
Many Recurrent Jacobians
↓
Earlier Computations
```

Then:

```text
Repeated Contraction
→ Vanishing
→ Early Steps Barely Learn
```

or:

```text
Repeated Expansion
→ Exploding
→ Training Becomes Unstable
```

Practical response:

```text
Exploding
→ Gradient Clipping

General Stability
→ Initialization + Normalization + Careful Optimization

Long-Term Learning
→ Better Memory Architecture
→ LSTM / GRU
```

---

# ⭐ Golden Rule

> **When debugging gradient problems, do not ask only “Is the gradient large or small?” Ask how its magnitude changes across depth or time, because long-range learning depends on whether useful error signals survive the entire computational path.**
