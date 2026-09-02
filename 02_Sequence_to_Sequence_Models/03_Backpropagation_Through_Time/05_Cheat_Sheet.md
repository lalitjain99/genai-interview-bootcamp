# ⚡ Cheat Sheet — Backpropagation Through Time

> **Goal:** Rapid revision of the most important BPTT concepts, equations, gradient-flow rules, shared-parameter behavior, truncation, and interview traps.

---

# 🌟 1. BPTT in One Line

> **BPTT = ordinary backpropagation applied to an RNN after unrolling it through time.**

It still uses:

> **chain rule**

---

# 🔓 2. Why Unroll an RNN?

Compact RNN:

```text id="bptt-cheat-1"
       ┌───────────┐
       │           │
x_t → RNN → h_t ───┘
```

Unrolled:

```text id="bptt-cheat-2"
h0 → RNN → h1 → RNN → h2 → RNN → h3
      ↑            ↑            ↑
     x1           x2           x3
```

Once unrolled:

> it becomes an ordinary computational graph.

---

# 🔄 3. Forward vs Backward

Forward:

```text id="bptt-cheat-3"
t1 → t2 → t3 → ... → tT
```

Backward:

```text id="bptt-cheat-4"
Loss
↓
hT
↓
h(T-1)
↓
...
↓
h1
```

Important:

> BPTT does NOT mean reversing the input sequence.

---

# 📐 4. Vanilla RNN Forward Equations

Pre-activation:

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

Hidden state:

`h_t = tanh(a_t)`

Optional output:

`y_t = W_hy h_t + b_y`

---

# 🧠 5. Weight Notation

```text id="bptt-cheat-5"
W_xh
→ x to h
→ Input-to-Hidden
```

```text id="bptt-cheat-6"
W_hh
→ h to h
→ Hidden-to-Hidden
```

```text id="bptt-cheat-7"
W_hy
→ h to y
→ Hidden-to-Output
```

---

# 🔁 6. Shared Parameters Across Time

The same:

* `W_xh`
* `W_hh`
* `b_h`

are reused at every time step.

```text id="bptt-cheat-8"
t1 → SAME parameters
t2 → SAME parameters
t3 → SAME parameters
```

So:

> unrolling creates many uses of the same parameter, not many independent parameters.

---

# ⭐ 7. Core Gradient Rule

If one variable affects the loss through multiple paths:

> **sum all gradient contributions.**

For an RNN hidden state:

```text id="bptt-cheat-9"
h_t
├──→ current output / loss
└──→ future hidden state
```

Therefore:

```text id="bptt-cheat-10"
Gradient at h_t
=
Local Contribution
+
Future Contribution
```

---

# 📐 8. Hidden-State Gradient

Conceptually:

`dL/dh_t`

`= local loss gradient`

`+ future hidden-state gradient`

A common recursive form is:

`dL/dh_t = dL_t/dh_t + (dL/dh_{t+1})(dh_{t+1}/dh_t)`

Key idea:

> current hidden state receives error information from both now and the future.

---

# 🔗 9. Why Future Loss Can Affect Earlier States

Example:

```text id="bptt-cheat-11"
h1 → h2 → h3 → L3
```

Even though the loss is at time `3`:

`h1`

still affects it through:

```text id="bptt-cheat-12"
h1 → h2 → h3 → L3
```

So gradients can travel:

> from later predictions back toward earlier sequence states.

---

# 📐 10. Hidden-to-Hidden Derivative

For:

`h_t = tanh(a_t)`

with:

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

the derivative from:

`h_{t-1}` to `h_t`

contains:

* `W_hh`
* `tanh'(a_t)`

Conceptually:

`dh_t/dh_{t-1} ≈ tanh'(a_t) × W_hh`

Exact matrix ordering depends on Jacobian convention.

---

# 🧠 11. Why Long Sequences Are Hard

Suppose:

`h_T`

depends on:

`h_1`.

Then the gradient involves:

```text id="bptt-cheat-13"
dh_T/dh_(T-1)
×
dh_(T-1)/dh_(T-2)
×
...
×
dh_2/dh_1
```

This means:

> repeated multiplication across time.

---

# 📉 12. Vanishing Gradient Intuition

If each backward step roughly multiplies by:

`0.5`

then after:

`10 steps`

we get:

`0.5^10 ≈ 0.00098`

So:

```text id="bptt-cheat-14"
Gradient
↓
smaller
↓
smaller
↓
≈ 0
```

Early time steps receive:

> very little useful learning signal.

---

# 💥 13. Exploding Gradient Intuition

If each backward step roughly multiplies by:

`2`

then after:

`10 steps`

we get:

`2^10 = 1024`

So:

```text id="bptt-cheat-15"
Gradient
↓
larger
↓
larger
↓
huge
```

Training can become unstable.

---

# ⚠️ 14. Don't Oversimplify Vanishing / Exploding

Avoid:

```text id="bptt-cheat-16"
small weight
→ vanish

large weight
→ explode
```

More precise:

> repeated products of recurrent Jacobians determine gradient behavior.

These include:

* recurrent weights
* activation derivatives

---

# 🔁 15. Shared Weight Gradient

Because:

`W_hh`

is reused at every time step:

# `dL/dW_hh = Σ temporal contributions`

For vanilla RNN:

`dL/dW_hh = Σ_t (dL/da_t) h_{t-1}^T`

Meaning:

```text id="bptt-cheat-17"
Contribution at t1
+
Contribution at t2
+
Contribution at t3
+
...
↓
One dL/dW_hh
```

---

# 📐 16. Input Weight Gradient

Similarly:

# `dL/dW_xh = Σ_t (dL/da_t) x_t^T`

Again:

> one shared matrix, many temporal contributions.

---

# 📐 17. Bias Gradient

Since the same bias is reused:

# `dL/db_h = Σ_t dL/da_t`

---

# 🔗 18. CNN vs RNN Gradient Sharing

CNN:

```text id="bptt-cheat-18"
Same Filter
→ many spatial uses

Backward
→ sum spatial contributions
```

RNN:

```text id="bptt-cheat-19"
Same W_hh
→ many temporal uses

Backward
→ sum temporal contributions
```

Shortcut:

```text id="bptt-cheat-20"
CNN → SPACE

RNN → TIME
```

---

# 🧠 19. Forward Sharing → Backward Accumulation

Best memory rule:

```text id="bptt-cheat-21"
FORWARD

One Parameter
→ Used Many Times
```

```text id="bptt-cheat-22"
BACKWARD

Many Paths
→ One Parameter Gradient
```

---

# ⚠️ 20. BPTT Does Not Update Parameters

BPTT computes:

`dL/dW`

Optimizer performs:

`W_new`

So:

```text id="bptt-cheat-23"
BPTT
→ Gradient Computation

Optimizer
→ Parameter Update
```

---

# 🧠 21. BPTT Workflow

```text id="bptt-cheat-24"
Input Sequence
↓
Forward RNN
↓
Hidden States
↓
Predictions
↓
Loss
↓
Backward Through Time
↓
Shared Parameter Gradients
↓
Optimizer
↓
Updated Weights
```

---

# 🎯 22. Credit Assignment

Credit assignment asks:

> **Which earlier computation was responsible for a later error?**

Example:

```text id="bptt-cheat-25"
Important event at t=5
↓
...
↓
Prediction at t=100
```

BPTT sends gradient backward so the earlier computation can receive:

> credit or blame.

Long-range credit assignment is difficult.

---

# 🧠 23. Long-Term Dependency Has Two Sides

### Forward Problem

Can important information survive?

```text id="bptt-cheat-26"
h1 → h2 → ... → hT
```

### Backward Problem

Can useful gradients survive?

```text id="bptt-cheat-27"
Loss → hT → ... → h1
```

So:

```text id="bptt-cheat-28"
Long-Term Dependency
=
Forward Information Retention
+
Backward Credit Assignment
```

---

# 💾 24. Training Memory vs Inference

### Streaming Inference

Usually needs:

```text id="bptt-cheat-29"
Current Input
+
Latest Hidden State
```

### Full BPTT Training

Needs enough historical activations/state to backpropagate:

```text id="bptt-cheat-30"
h1
h2
h3
...
hT
```

So:

> training memory grows with sequence length.

---

# ✂️ 25. Truncated BPTT — TBPTT

For very long sequences, instead of backpropagating through everything:

```text id="bptt-cheat-31"
t1 → ... → t10000
```

use shorter chunks:

```text id="bptt-cheat-32"
t1 ... t100
↓
Backward

t101 ... t200
↓
Backward
```

This is:

> **Truncated Backpropagation Through Time**

---

# 🧠 26. Detach Hidden State

Between chunks:

```text id="bptt-cheat-33"
Chunk 1
↓
h100
↓
Chunk 2
```

You can carry the value:

`h100`

forward.

But detach its old graph.

Remember:

```text id="bptt-cheat-34"
Carry STATE VALUE
✅

Carry OLD GRADIENT HISTORY
❌
```

---

# ⚖️ 27. TBPTT Trade-Off

### Benefit

* lower memory
* lower backward compute
* practical for long streams

### Cost

* limited gradient horizon
* harder long-range credit assignment

So:

```text id="bptt-cheat-35"
Training Practicality ↑

Gradient Horizon ↓
```

---

# ⭐ 28. Forward Memory Horizon ≠ Gradient Horizon

This is very important.

A hidden state can carry information beyond a truncation boundary.

But the loss cannot directly backpropagate beyond that detached boundary.

So:

```text id="bptt-cheat-36"
Forward Information
can continue

Backward Gradient
may stop
```

---

# ⚠️ 29. BPTT vs Bidirectional RNN

### BPTT

> training mechanism.

### Bidirectional RNN

> architecture that reads sequence in both directions.

Remember:

```text id="bptt-cheat-37"
BPTT
→ How RNN LEARNS

Bidirectional RNN
→ How RNN READS CONTEXT
```

---

# ⚠️ 30. BPTT vs Reversing Input

Wrong:

> “BPTT feeds `x_T ... x_1`.”

Correct:

Forward may remain:

```text id="bptt-cheat-38"
x1 → x2 → x3
```

Backward:

```text id="bptt-cheat-39"
Loss → h3 → h2 → h1
```

---

# 🧮 31. Scalar Toy Example

Suppose:

`h_t = w h_{t-1}`

Then:

`h3 = w³ h0`

If:

`L = h3`

then:

`dL/dh0 = w³`

If:

`w = 0.5`

gradient shrinks.

If:

`w = 2`

gradient grows.

This toy example shows:

> repeated multiplication through recurrence.

---

# 🧮 32. Shared Weight Toy Example

Suppose:

`h1 = w h0`

`h2 = w h1`

Then:

`h2 = w² h0`

If:

`L = h2`

then:

`dL/dw = 2w h0`

The factor `2` appears because:

> the same `w` influenced the loss through two recurrent uses.

---

# 🚫 33. High-Yield Interview Traps

### ❌ BPTT is different mathematics from backprop

No.

> Same chain rule.

---

### ❌ Every time step gets its own `W_hh`

No.

> Same shared matrix.

---

### ❌ `dL/dh_t` comes only from current loss

No.

> future states can contribute.

---

### ❌ BPTT reverses the input sequence

No.

---

### ❌ BPTT directly updates weights

No.

> optimizer does.

---

### ❌ Vanishing gradients are caused only by `tanh`

Too simplistic.

---

### ❌ TBPTT must reset hidden state

No.

> state can be carried; graph can be detached.

---

### ❌ TBPTT makes all dependencies shorter than its window

Too strong.

Forward state may carry older information.

---

### ❌ BPTT and bidirectional RNN are related concepts

They solve different problems.

---

# 📐 34. Formula Flash Card

### Forward

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

`h_t = tanh(a_t)`

---

### Hidden-State Gradient

```text id="bptt-cheat-40"
dL/dh_t
=
Local Contribution
+
Future Contribution
```

---

### Recurrent Weight

`dL/dW_hh = Σ_t (dL/da_t) h_{t-1}^T`

---

### Input Weight

`dL/dW_xh = Σ_t (dL/da_t) x_t^T`

---

### Bias

`dL/db_h = Σ_t dL/da_t`

---

# 🧠 35. Seven Things to Know Cold

1. **BPTT = backprop on unrolled RNN**
2. **Chain rule is unchanged**
3. **Gradient moves late → early**
4. **Hidden state gets local + future gradient**
5. **Shared parameters collect temporal contributions**
6. **Long paths cause repeated Jacobian multiplication**
7. **Repeated multiplication can vanish or explode**

---

# 🎤 36. 30-Second Interview Answer

> **Backpropagation Through Time is ordinary backpropagation applied to an RNN after unrolling its recurrence across time steps. The gradient moves from later hidden states toward earlier ones through the chain rule. Because recurrent parameters such as `W_hh` are reused at every time step, their final gradients are the sum of contributions from all temporal uses. For long sequences, gradients involve products of many recurrent Jacobians, which can cause vanishing or exploding gradients and make long-range credit assignment difficult.**

---

# 🧠 37. BPTT Story in One Diagram

```text id="bptt-cheat-41"
RNN Shares Weights Across Time
↓
Unroll Recurrence
↓
Ordinary Computational Graph
↓
Compute Loss
↓
Backpropagate Late → Early
↓
Sum Temporal Parameter Contributions
↓
Long Gradient Path
↓
Repeated Multiplication
↓
Vanishing / Exploding Gradient Risk
```

---

# ⭐ Final Mental Model

```text id="bptt-cheat-42"
FORWARD

x1 → h1 → h2 → h3 → ... → hT → Loss
```

```text id="bptt-cheat-43"
BACKWARD

Loss
↓
hT
↓
h(T-1)
↓
...
↓
h1
```

At each step:

```text id="bptt-cheat-44"
Gradient continues backward
+
Shared weights receive a contribution
```

Finally:

```text id="bptt-cheat-45"
All Contributions
↓
dL/dW_xh
dL/dW_hh
db_h
↓
Optimizer
↓
Updated RNN
```

---

# ⭐ Golden Rule

> **BPTT = unroll the RNN, apply ordinary chain rule backward through time, and sum gradient contributions from every temporal use of the shared recurrent parameters.**
