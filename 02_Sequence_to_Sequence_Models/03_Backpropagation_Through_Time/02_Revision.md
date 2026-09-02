# 📝 Revision — Backpropagation Through Time

> **Goal:** Quickly revise what BPTT is, why RNNs need it, how gradients flow through time, how shared recurrent weights receive gradients, and why long sequences create vanishing/exploding gradient problems.

---

# 🌟 1. Why Do We Need BPTT?

In an RNN:

```text id="bptt-rev-1"
x1 → h1 → h2 → h3 → ... → hT
```

the hidden state at one time step affects later hidden states.

If the final prediction is wrong, the loss may depend indirectly on:

```text id="bptt-rev-2"
hT
↑
h(T-1)
↑
...
↑
h1
```

So the learning signal must move backward across the recurrent chain.

This is:

> **Backpropagation Through Time — BPTT**

---

# 🧠 2. BPTT in One Line

> **BPTT = ordinary backpropagation applied to an RNN after unrolling it through time.**

It is not a new type of calculus.

It still uses:

> **chain rule**

---

# 🔓 3. Why Unroll the RNN?

Compact RNN:

```text id="bptt-rev-3"
       ┌───────────┐
       │           │
x_t → RNN → h_t ───┘
```

Unrolled:

```text id="bptt-rev-4"
h0 → RNN → h1 → RNN → h2 → RNN → h3
      ↑            ↑            ↑
     x1           x2           x3
```

Now recurrence becomes:

> an ordinary computational graph.

So we can apply backpropagation normally.

---

# 🔄 4. Forward vs Backward Direction

Forward:

```text id="bptt-rev-5"
t1 → t2 → t3 → ... → tT
```

Backward:

```text id="bptt-rev-6"
tT → t(T-1) → ... → t1
```

Important:

> BPTT does not mean the input sequence itself is fed backward.

It means:

> gradients move backward through the unrolled computation graph.

---

# 📐 5. Vanilla RNN Forward Equation

At time `t`:

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

Then:

`h_t = tanh(a_t)`

Where:

```text id="bptt-rev-7"
W_xh
→ x to h
→ Input-to-Hidden weights

W_hh
→ h to h
→ Hidden-to-Hidden recurrent weights
```

Optional output:

`y_t = W_hy h_t + b_y`

where:

`W_hy`

means:

> Hidden-to-Output weights.

---

# 🧠 6. Why Shared Weights Matter

The same:

```text id="bptt-rev-8"
W_xh
W_hh
b_h
```

are reused at every time step.

Example:

```text id="bptt-rev-9"
t1 → W_hh
t2 → SAME W_hh
t3 → SAME W_hh
```

Therefore during backward:

> every temporal use contributes to the gradient of the same parameter.

---

# 🔗 7. Connection to CNN Backpropagation

CNN:

```text id="bptt-rev-10"
Same Filter
→ many spatial positions
```

Backward:

```text id="bptt-rev-11"
many spatial gradient contributions
↓
one filter gradient
```

RNN:

```text id="bptt-rev-12"
Same W_hh
→ many time steps
```

Backward:

```text id="bptt-rev-13"
many temporal gradient contributions
↓
one dL/dW_hh
```

Shortcut:

```text id="bptt-rev-14"
CNN
→ sharing across SPACE

RNN
→ sharing across TIME
```

---

# 🧠 8. Hidden-State Gradient Has Multiple Paths

Suppose:

```text id="bptt-rev-15"
h1 → L1
↓
h2 → L2
↓
h3 → L3
```

Consider:

`h2`

It influences:

* current loss `L2`
* future state `h3`, which affects `L3`

So:

> gradient at `h2` must include both paths.

Conceptually:

```text id="bptt-rev-16"
Gradient at h_t
=
Local Loss Contribution
+
Gradient Coming Back From Future States
```

This is one of the most important BPTT ideas.

---

# 📐 9. Recursive Hidden-State Gradient

Conceptually:

`dL/dh_t`

contains:

`direct contribution from L_t`

plus:

`future contribution through h_{t+1}`

So:

`dL/dh_t = dL_t/dh_t + (dL/dh_{t+1})(dh_{t+1}/dh_t)`

The exact notation may vary, but the idea is:

> **current hidden-state gradient = local gradient + future gradient**

---

# 🔄 10. Final-Loss-Only Case

Suppose only final step has a loss:

```text id="bptt-rev-17"
h1 → h2 → h3 → L
```

Then:

```text id="bptt-rev-18"
L
↓
h3
↓
h2
↓
h1
```

So:

`dL/dh1`

must travel through:

* `h3 → h2`
* `h2 → h1`

This introduces repeated derivative multiplication.

---

# 📐 11. Hidden-to-Hidden Derivative

Recall:

`h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`

The derivative:

`dh_t/dh_{t-1}`

contains:

* `W_hh`
* `tanh'`

Conceptually:

`dh_t/dh_{t-1} ≈ tanh'(a_t) × W_hh`

Exact matrix ordering depends on derivative convention.

The key point is:

> every backward time step multiplies through the recurrent transformation and activation derivative.

---

# ⭐ 12. Long Gradient Path

For a final hidden state `h_T` and an early hidden state `h_k`:

`∂h_T/∂h_k`

contains:

```text id="bptt-rev-19"
∂h_T/∂h_(T-1)
×
∂h_(T-1)/∂h_(T-2)
×
...
×
∂h_(k+1)/∂h_k
```

So long-range learning requires:

> repeated Jacobian multiplication.

This is the mathematical source of the gradient stability problem.

---

# 🧮 13. Scalar Intuition

Suppose each backward step roughly multiplies the gradient by:

`0.5`

After 5 steps:

`0.5^5 = 0.03125`

After 10:

`0.5^10 ≈ 0.00098`

Gradient becomes tiny.

That is:

> **vanishing gradient**

---

# 💥 14. Exploding Case

Suppose each step roughly multiplies by:

`2`

After 5 steps:

`2^5 = 32`

After 10:

`2^10 = 1024`

Gradient becomes huge.

That is:

> **exploding gradient**

---

# ⚠️ 15. Don't Oversimplify the Cause

Avoid saying only:

```text id="bptt-rev-20"
small weight
→ vanishing

large weight
→ exploding
```

The more precise story involves repeated products of:

* recurrent Jacobians
* recurrent weights
* activation derivatives

So:

> gradient behavior depends on the overall recurrent transformation.

---

# 📐 16. Gradient of `W_hh`

`W_hh`

is reused at every time step.

Therefore:

# `dL/dW_hh = Σ_t contribution from time t`

For vanilla RNN:

`dL/dW_hh = Σ_t (dL/da_t) h_{t-1}^T`

Conceptually:

```text id="bptt-rev-21"
Gradient Contribution t1
+
Gradient Contribution t2
+
Gradient Contribution t3
+
...
↓
One dL/dW_hh
```

---

# 📐 17. Gradient of `W_xh`

Similarly:

`W_xh`

is reused at every step.

So:

# `dL/dW_xh = Σ_t (dL/da_t) x_t^T`

Again:

> all temporal uses contribute.

---

# 📐 18. Bias Gradient

Bias is also shared across time.

So:

# `dL/db_h = Σ_t dL/da_t`

In batched training, gradients are also aggregated appropriately across examples.

---

# 🧠 19. Forward Sharing → Backward Accumulation

This is the easiest way to remember shared-weight gradients:

```text id="bptt-rev-22"
FORWARD

One Parameter
→ Used Many Times
```

```text id="bptt-rev-23"
BACKWARD

Many Contributions
→ One Parameter Gradient
```

For RNN:

> many temporal uses → one shared gradient.

---

# ⚠️ 20. Mathematical Accumulation vs Framework `.grad` Accumulation

These are different.

### Mathematical accumulation

Multiple graph paths contribute to the same parameter gradient.

This is required by:

> chain rule.

### Framework accumulation

A framework may accumulate gradients across multiple backward calls until gradients are reset.

This is:

> implementation behavior.

Do not confuse them.

---

# 🧠 21. BPTT Does Not Update Weights

BPTT computes:

> gradients.

Optimizer performs:

> parameter updates.

So:

```text id="bptt-rev-24"
BPTT
→ dL/dW
```

```text id="bptt-rev-25"
Optimizer
→ W_new
```

Same distinction as ordinary neural-network training.

---

# 🔄 22. Typical BPTT Workflow

```text id="bptt-rev-26"
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
Gradients for Shared Weights
↓
Optimizer
↓
Updated Parameters
```

---

# 💾 23. Training Memory vs Streaming Inference

During inference:

```text id="bptt-rev-27"
x_t + h_{t-1}
↓
h_t
```

we mainly need the latest state for ongoing processing.

During full BPTT training, backward needs intermediate information from:

```text id="bptt-rev-28"
h1
h2
h3
...
hT
```

So:

```text id="bptt-rev-29"
Training
→ more activation/state memory

Streaming inference
→ can maintain mainly current state
```

---

# ✂️ 24. Truncated BPTT

For very long sequences, full BPTT can be expensive.

Instead, we may train in shorter chunks:

```text id="bptt-rev-30"
t1 ... t100
↓
Backprop

t101 ... t200
↓
Backprop
```

This is:

> **Truncated Backpropagation Through Time — TBPTT**

---

# 🧠 25. What Does Detaching the State Mean?

Between chunks:

```text id="bptt-rev-31"
Chunk 1
↓
h100
↓
Chunk 2
```

we may carry the numerical value of `h100` forward but detach it from the old computation graph.

Remember:

```text id="bptt-rev-32"
Carry State VALUE Forward
✅

Carry Infinite Gradient History
❌
```

---

# ⚖️ 26. TBPTT Trade-Off

Benefits:

* less memory
* shorter backward passes
* practical for long sequences

Cost:

> gradients cannot travel beyond the truncation boundary.

So:

```text id="bptt-rev-33"
Practical Training ↑

Long-Range Credit Assignment ↓
```

---

# 🧠 27. Forward Memory vs Backward Credit Assignment

This is subtle.

With TBPTT, hidden state can still carry some information forward:

```text id="bptt-rev-34"
Chunk 1
→ h
→ Chunk 2
```

But gradient may not flow backward across that boundary.

So:

```text id="bptt-rev-35"
Forward Information Flow
≠
Backward Gradient Flow
```

---

# 🎯 28. Credit Assignment

Credit assignment asks:

> Which earlier computation deserves credit or blame for a later prediction?

Example:

```text id="bptt-rev-36"
Important x1
↓
...
↓
Prediction at t100
```

To learn the relationship, gradient must communicate the error back toward:

`x1`

and the parameters that processed it.

Long-range credit assignment is one of the main difficulties in recurrent networks.

---

# 🧠 29. Two Long-Term Problems

Long-term dependency problems have two sides.

### Forward Problem

Can important information survive:

```text id="bptt-rev-37"
h1 → h2 → ... → hT
```

### Backward Problem

Can the learning signal survive:

```text id="bptt-rev-38"
Loss → hT → ... → h1
```

So:

```text id="bptt-rev-39"
Long-Term Dependency
=
Information Retention
+
Credit Assignment
```

---

# 🔥 30. `tanh` and BPTT

`tanh` derivative:

`1 - tanh²(z)`

Its magnitude is at most:

`1`

and often much smaller in saturated regions.

Therefore repeated multiplication through `tanh'` can contribute to:

> vanishing gradients.

But:

> `W_hh` also participates.

So do not say:

> “tanh alone causes vanishing gradients.”

---

# ⚠️ 31. BPTT vs Bidirectional RNN

Do not confuse these.

### BPTT

> training algorithm.

Gradients move backward through time.

### Bidirectional RNN

> architecture.

Processes sequence in:

* forward direction
* backward direction

So:

```text id="bptt-rev-40"
BPTT
→ How the RNN LEARNS

Bidirectional RNN
→ How the RNN READS context
```

---

# ⚠️ 32. BPTT vs Reversing the Sequence

BPTT:

```text id="bptt-rev-41"
Input:
x1 → x2 → x3

Gradient:
Loss → h3 → h2 → h1
```

It does NOT mean:

```text id="bptt-rev-42"
x3 → x2 → x1
```

is fed as input.

---

# 🧮 33. Tiny Toy Example

Suppose:

`h_t = w h_{t-1}`

Then after 3 steps:

`h3 = w³ h0`

If:

`L = h3`

then:

`dL/dh0 = w³`

If:

`w = 0.5`

then:

`0.5³ = 0.125`

If sequence had 20 steps:

`0.5^20`

is extremely small.

If:

`w = 2`

then:

`2^20`

is extremely large.

This simplified example demonstrates:

> repeated multiplication across time.

---

# ⚡ 34. High-Yield BPTT Traps

### ❌ BPTT is a completely new learning algorithm

No.

> It is backprop + chain rule on an unrolled RNN.

---

### ❌ Every time step has separate `W_hh`

No.

> Same recurrent parameter is reused.

---

### ❌ Shared parameter receives only one time-step gradient

No.

> Contributions from all relevant uses are summed.

---

### ❌ BPTT updates weights

No.

> BPTT computes gradients; optimizer updates weights.

---

### ❌ Backward through time means reverse the input

No.

---

### ❌ Hidden-state gradient comes only from current loss

No.

It may also receive:

> future-state gradient.

---

### ❌ Vanishing gradient is only because weights are small

Too simplistic.

---

### ❌ TBPTT resets all forward memory

Not necessarily.

State value may be carried forward while:

> gradient history is detached.

---

# 📐 35. Formula Flash Card

### Forward

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

`h_t = tanh(a_t)`

---

### Hidden-State Gradient

Conceptually:

`Gradient at h_t`

`= local loss gradient + future gradient`

---

### Shared Recurrent Weight

`dL/dW_hh = Σ_t (dL/da_t) h_{t-1}^T`

---

### Input Weight

`dL/dW_xh = Σ_t (dL/da_t) x_t^T`

---

### Bias

`dL/db_h = Σ_t dL/da_t`

---

# 🎤 36. 30-Second Interview Revision

> **Backpropagation Through Time is ordinary backpropagation applied to an RNN after it is unrolled across its sequence steps. Gradients move from later time steps toward earlier hidden states using the chain rule. Because the same recurrent parameters are reused across time, their gradients accumulate contributions from every temporal use. For long sequences, the backward path contains repeated recurrent Jacobian multiplications, which can make gradients vanish or explode and makes long-range credit assignment difficult.**

---

# 🧠 37. Seven Things You Must Know Cold

1. **BPTT = backprop on an unrolled RNN.**
2. **Chain rule is still the core mathematics.**
3. **Hidden-state gradient receives local + future contributions.**
4. **RNN weights are shared across time.**
5. **Shared-weight gradients sum across temporal uses.**
6. **Long paths create repeated derivative multiplication.**
7. **Repeated multiplication leads toward vanishing/exploding gradients.**

---

# 🧠 Final Mental Model

```text id="bptt-rev-43"
FORWARD

x1 → h1 → h2 → h3 → ... → hT → Loss
```

```text id="bptt-rev-44"
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

At every step:

```text id="bptt-rev-45"
Gradient moves farther into the past
+
Shared parameter receives a contribution
```

Finally:

```text id="bptt-rev-46"
Temporal Contributions
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

# 🔗 Where This Leads

The most important discovery from BPTT is:

```text id="bptt-rev-47"
Long Sequence
↓
Long Gradient Path
↓
Repeated Multiplication
↓
Gradient Shrinks or Grows
↓
Vanishing / Exploding Gradients
```

That is the exact problem we will study next.

---

# ⭐ Golden Rule

> **BPTT unrolls recurrence into an ordinary computational graph, sends gradients backward through hidden-state dependencies, and sums the contributions created by every temporal use of the shared RNN parameters.**
