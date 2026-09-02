# 📘 Lecture — Backpropagation Through Time

## 🌟 Introduction

In the previous lecture, we built a Recurrent Neural Network from first principles.

At every time step:

```text id="bptt-intro-1"
Current Input x_t
+
Previous Hidden State h_{t-1}
↓
RNN Cell
↓
New Hidden State h_t
```

Mathematically:

`h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`

And we learned something extremely important:

> **the same `W_xh` and `W_hh` are reused at every time step.**

Now a new question appears.

Suppose our RNN processes:

```text id="bptt-intro-2"
I → love → AI
```

and produces the wrong prediction at the end.

The final loss may depend on:

```text id="bptt-intro-3"
x3
↑
h3
↑
h2
↑
h1
↑
x1
```

But the same recurrent matrix `W_hh` was used repeatedly:

```text id="bptt-intro-4"
h1 → h2
     W_hh

h2 → h3
     SAME W_hh
```

So how does one final loss teach a weight matrix that was used many times?

Do we calculate separate gradients like:

```text id="bptt-intro-5"
dL/dW_hh at t1
dL/dW_hh at t2
dL/dW_hh at t3
```

and somehow combine them?

And how can the loss at the end teach something that happened several time steps earlier?

To answer these questions, we use:

# **Backpropagation Through Time — BPTT**

The name sounds like a completely new algorithm.

It is not.

> **BPTT is ordinary backpropagation applied to an RNN after the recurrent computation has been unrolled across time.**

That single statement is the foundation of this lecture.

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to:

### 🧠 Conceptual Understanding

* Explain why an RNN must be unrolled before reasoning about backpropagation.
* Explain why BPTT is ordinary chain-rule backpropagation rather than a fundamentally different algorithm.
* Explain how a loss at a later time step can affect earlier hidden states.
* Distinguish:

  * gradient through hidden states
  * gradient with respect to shared RNN parameters
* Explain why shared parameters receive gradient contributions from multiple time steps.

### 📐 Mathematical Understanding

* Write the forward equations of a vanilla RNN.
* Construct a small unrolled computational graph.
* Derive how `dL/dh_t` receives both:

  * a direct contribution from the current output
  * an indirect contribution through future hidden states.
* Derive the recursive hidden-state gradient relation.
* Understand how `dL/dW_hh` and `dL/dW_xh` accumulate across time.
* Explain why repeated Jacobian multiplication appears during BPTT.

### 🧮 Applied Understanding

* Trace gradients through a 3-step RNN.
* Explain why early time steps can receive weaker or stronger gradients.
* Connect RNN shared-weight gradients with CNN parameter sharing.
* Understand what **truncated BPTT** means at a high level.

### ⚠️ Limitation Awareness

* Understand why long recurrent chains make training difficult.
* Build the mathematical intuition for:

  * vanishing gradients
  * exploding gradients

These will become the central topic of our next lecture.

---

# 📖 Part 1 — Let's Start From the Problem

Consider a simple sentiment model.

Input:

```text id="bptt-story-1"
I → love → this → movie
```

The RNN processes each token:

```text id="bptt-story-2"
x1 → h1
x2 → h2
x3 → h3
x4 → h4
```

Suppose we use:

`h4`

to predict sentiment:

```text id="bptt-story-3"
h4
↓
Classifier
↓
Positive
```

But the correct answer was:

> Negative

So we calculate some loss:

`L`

Now the optimizer eventually needs gradients such as:

`dL/dW_xh`

and:

`dL/dW_hh`

But here is the difficulty.

`W_hh` was not used once.

It was used repeatedly.

---

# 🧠 Part 2 — Follow One Shared Weight

Imagine `W_hh` as one shared recurrent transformation.

Forward processing:

```text id="bptt-story-4"
h0
↓
W_hh used
↓
h1
↓
W_hh used AGAIN
↓
h2
↓
W_hh used AGAIN
↓
h3
↓
W_hh used AGAIN
↓
h4
```

It is the same parameter each time.

So when the prediction is wrong, the loss can blame `W_hh` through:

> every place where `W_hh` participated in producing the final result.

That should remind you of CNN backpropagation.

---

# 🔗 Part 3 — Connection to CNN Backpropagation

In a CNN:

```text id="bptt-cnn-1"
Same Filter
↓
Location 1
Location 2
Location 3
Location 4
```

The filter is used many times.

During backward:

```text id="bptt-cnn-2"
Gradient from Location 1
+
Gradient from Location 2
+
Gradient from Location 3
+
Gradient from Location 4
↓
One Filter Gradient
```

RNN does the same kind of thing, but across:

> **time**

instead of space.

```text id="bptt-rnn-1"
Same W_hh
↓
Time 1
Time 2
Time 3
Time 4
```

Backward:

```text id="bptt-rnn-2"
Gradient contribution from t1
+
Gradient contribution from t2
+
Gradient contribution from t3
+
Gradient contribution from t4
↓
One dL/dW_hh
```

But RNNs introduce an additional complication:

> hidden states are recursively dependent on previous hidden states.

That creates long gradient paths.

---

# 📖 Part 4 — Why Do We “Unroll” the RNN?

The compact RNN looks like:

```text id="bptt-unroll-1"
       ┌─────────────┐
       │             │
x_t → RNN Cell → h_t ┘
```

Trying to backpropagate through a loop is hard to visualize.

So we expand the recurrence across time.

For three steps:

```text id="bptt-unroll-2"
h0 ──→ [RNN] ──→ h1 ──→ [RNN] ──→ h2 ──→ [RNN] ──→ h3
        ↑                  ↑                  ↑
        x1                 x2                 x3
```

Now this looks like an ordinary computational graph.

And ordinary computational graphs can be trained with:

> **backpropagation + chain rule**

So BPTT begins with:

# **Unroll the RNN through time.**

---

# 🧠 Part 5 — BPTT Is Not a New Type of Calculus

This distinction is important.

Ordinary neural network:

```text id="bptt-normal-1"
Layer 1
↓
Layer 2
↓
Layer 3
↓
Loss
```

Backpropagate through:

> layers.

Unrolled RNN:

```text id="bptt-normal-2"
Time 1
↓
Time 2
↓
Time 3
↓
Loss
```

Backpropagate through:

> time steps.

The mathematics is still:

# **Chain Rule**

So:

```text id="bptt-normal-3"
Backpropagation
through network depth
```

becomes:

```text id="bptt-normal-4"
Backpropagation
through recurrent time depth
```

Hence:

> **Backpropagation Through Time**

---

# 📐 Part 6 — Forward Equations

Let's use a simple vanilla RNN.

At time step `t`:

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

Then:

`h_t = tanh(a_t)`

Suppose every hidden state produces an output:

`o_t = W_hy h_t + b_y`

and a corresponding loss:

`L_t`

Total sequence loss could be:

`L = L_1 + L_2 + ... + L_T`

or often an average.

For now, using a sum makes the dependency easier to see.

---

# 🧠 Part 7 — Start With a Tiny 3-Step RNN

Let's use:

```text id="bptt-3step-1"
x1 → h1 → y1 → L1
      ↓
x2 → h2 → y2 → L2
      ↓
x3 → h3 → y3 → L3
```

Total loss:

`L = L1 + L2 + L3`

Now ask:

> What influences `h2`?

`h2` influences:

* `L2` directly through `y2`
* `h3`, which then influences `L3`

Therefore `h2` affects:

```text id="bptt-3step-2"
L2
+
L3
```

This is where BPTT becomes interesting.

---

# ⭐ Part 8 — A Hidden State Has More Than One Gradient Path

Suppose we want:

`dL/dh2`

There are two paths.

### Path 1 — Current-Time Loss

```text id="bptt-path-1"
h2
↓
y2
↓
L2
```

### Path 2 — Future-Time Loss

```text id="bptt-path-2"
h2
↓
h3
↓
y3
↓
L3
```

Therefore:

`dL/dh2`

must include:

> gradient from every downstream path.

Conceptually:

`dL/dh2 = contribution from L2 + contribution through h3`

This is simply the computational-graph rule:

> **if one variable influences the loss through multiple paths, sum all gradient contributions.**

---

# 🔗 Part 9 — This Is the Same Rule We Already Know

Remember residual connections?

If:

```text id="bptt-branch-1"
x
├──→ path A
└──→ path B
```

then during backward:

```text id="bptt-branch-2"
gradient from A
+
gradient from B
↓
gradient for x
```

The same rule appears in RNNs.

`h_t` can influence:

* the current output
* the next hidden state

So its gradient receives:

> both contributions.

Nothing magical has happened.

---

# 📐 Part 10 — Deriving the Hidden-State Gradient

Suppose:

`L = L1 + L2 + L3`

For `h3`, there is no future hidden state in our small example.

So:

`dL/dh3`

comes from:

`L3`

through the output.

Now look at `h2`.

It influences:

* `L2`
* `h3`

So:

`dL/dh2 = dL2/dh2 + (dL/dh3)(dh3/dh2)`

Similarly:

`dL/dh1 = dL1/dh1 + (dL/dh2)(dh2/dh1)`

This is the essential recursive gradient relationship in BPTT.

---

# 🧠 Part 11 — The Most Important BPTT Equation Intuitively

Don't memorize the symbols first.

Think:

```text id="bptt-intuition-1"
Gradient at Current Hidden State
=
Gradient from Current Prediction
+
Gradient Coming Back From the Future
```

Or:

# `Current gradient = local contribution + future contribution`

This is one of the most important mental models in BPTT.

---

# 📐 Part 12 — What Is `dh_t/dh_{t-1}`?

Recall:

`h_t = tanh(a_t)`

where:

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

We want:

`dh_t/dh_{t-1}`

There are two transformations:

```text id="bptt-jacobian-1"
h_{t-1}
↓
W_hh
↓
a_t
↓
tanh
↓
h_t
```

By chain rule, the hidden-to-hidden derivative contains:

* recurrent weight transformation `W_hh`
* derivative of `tanh`

Conceptually:

`dh_t/dh_{t-1} ≈ tanh'(a_t) × W_hh`

The exact matrix orientation depends on the vector/Jacobian convention, but the important structural point is:

> **each backward time step multiplies the gradient through the recurrent weights and activation derivative.**

This is the seed of the vanishing/exploding gradient problem.

---

# 🧠 Part 13 — Follow a Gradient Backward

Suppose the only loss is at the final step:

`L = L3`

Then:

```text id="bptt-grad-flow-1"
L3
↓
h3
↓
h2
↓
h1
```

For `h2`:

`dL/dh2 = dL/dh3 × dh3/dh2`

For `h1`:

`dL/dh1 = dL/dh3 × dh3/dh2 × dh2/dh1`

Notice what appears:

> repeated multiplication.

For a long sequence:

```text id="bptt-grad-flow-2"
dL/dh1
=
dL/dhT
×
dhT/dh(T-1)
×
dh(T-1)/dh(T-2)
×
...
×
dh2/dh1
```

This is the central mathematical reason long RNNs become difficult to train.

---

# 🧮 Part 14 — Scalar Toy Example

Before thinking about matrices, let's use a scalar recurrence.

Suppose:

`h_t = tanh(w h_{t-1} + u x_t)`

Ignore bias for simplicity.

Imagine, only for intuition, that along the backward path:

`tanh'(...) ≈ 0.5`

and:

`w = 0.8`

Then each backward step contributes roughly:

`0.5 × 0.8 = 0.4`

Suppose a gradient must travel backward 5 steps.

Then approximately:

`0.4^5 = 0.01024`

A gradient of:

`1`

at the end becomes roughly:

`0.01`

five steps earlier.

After 10 steps:

`0.4^10 ≈ 0.000105`

So the learning signal becomes tiny.

This is the intuition behind:

> **vanishing gradients**

We will study this precisely in the next lecture.

---

# 💥 Part 15 — What If the Repeated Factor Is Larger Than 1?

Suppose the effective backward multiplier is:

`1.5`

Then after 5 steps:

`1.5^5 ≈ 7.59`

After 10 steps:

`1.5^10 ≈ 57.67`

After 20 steps:

`1.5^20 ≈ 3325`

Now the gradient becomes enormous.

This is:

> **exploding gradient**

So repeated multiplication is dangerous in both directions.

---

# 📖 Part 16 — But We Are Not Done: How Does `W_hh` Get Its Gradient?

So far we tracked gradients through hidden states.

Now consider:

`W_hh`

Recall:

`W_hh` appears in every recurrent step.

For three time steps:

```text id="bptt-whh-1"
h0 ──W_hh──→ h1
h1 ──W_hh──→ h2
h2 ──W_hh──→ h3
```

But these are all uses of:

> **the same parameter**

Therefore:

`dL/dW_hh`

must include contributions from all three uses.

Conceptually:

# `dL/dW_hh = contribution_t1 + contribution_t2 + contribution_t3`

This is shared-parameter gradient accumulation across time.

---

# 📐 Part 17 — Gradient Contribution From One Time Step

At time `t`:

`a_t = W_hh h_{t-1} + ...`

Suppose we know:

`dL/da_t`

Then the gradient contribution to `W_hh` from this time step is conceptually an outer product:

`(dL/da_t) × h_{t-1}^T`

So over all time steps:

# `dL/dW_hh = Σ_t (dL/da_t) h_{t-1}^T`

This equation is extremely important.

It tells us:

> each time step where `W_hh` was used contributes to the final shared gradient.

---

# 🧠 Part 18 — Intuition Behind That Formula

At one step:

```text id="bptt-whh-intuition-1"
Previous State h_{t-1}
↓
W_hh
↓
Current Pre-activation a_t
↓
...
↓
Loss
```

The weight gradient asks:

> “How much did this recurrent transformation at this time step contribute to the loss?”

Then because the same `W_hh` appears at all time steps:

```text id="bptt-whh-intuition-2"
Contribution at t1
+
Contribution at t2
+
Contribution at t3
+
...
↓
Shared W_hh Gradient
```

Again:

> forward sharing → backward accumulation.

---

# 🔗 Part 19 — Exact Connection With CNN

CNN:

```text id="bptt-cnn-rnn-1"
One Filter
used at many spatial positions
```

Gradient:

`sum over spatial uses`

RNN:

```text id="bptt-cnn-rnn-2"
One W_hh
used at many temporal positions
```

Gradient:

`sum over temporal uses`

So:

```text id="bptt-cnn-rnn-3"
CNN
Forward sharing across SPACE
→ backward accumulation across SPACE

RNN
Forward sharing across TIME
→ backward accumulation across TIME
```

This should feel very familiar now.

---

# 📐 Part 20 — What About `W_xh`?

Exactly the same principle.

At every step:

`a_t = W_xh x_t + ...`

Therefore:

# `dL/dW_xh = Σ_t (dL/da_t) x_t^T`

Again:

> one shared matrix, many temporal uses, summed gradient contributions.

---

# 📐 Part 21 — What About Bias?

Bias:

`b_h`

is also reused at every time step.

So:

# `dL/db_h = Σ_t dL/da_t`

and if processing a batch, we also usually aggregate appropriately across:

> batch examples.

---

# 🧠 Part 22 — Important Distinction: Two Types of Accumulation

This is something we also discussed during CNN backprop.

### Mathematical Gradient Accumulation

Because the same parameter appears multiple times in the computational graph:

```text id="bptt-two-1"
t1 use
+
t2 use
+
t3 use
↓
one gradient
```

This is required by calculus.

---

### Framework `.grad` Accumulation

Libraries such as PyTorch can also accumulate `.grad` values across multiple backward calls unless gradients are reset.

That is:

> a separate implementation behavior.

Do not confuse the two.

---

# 📖 Part 23 — Let's Trace a 3-Step RNN Completely

Suppose:

```text id="bptt-trace-1"
h0
↓
x1 → h1
      ↓
x2 → h2
      ↓
x3 → h3
      ↓
     y3
      ↓
      L
```

Only the final time step has a loss.

Forward:

`h1 = f(x1, h0)`

`h2 = f(x2, h1)`

`h3 = f(x3, h2)`

`L = loss(y3)`

Backward starts from:

`dL/dy3`

Then:

```text id="bptt-trace-2"
dL/dy3
↓
dL/dh3
↓
dL/dh2
↓
dL/dh1
```

At each hidden step we also compute contributions to:

* `W_xh`
* `W_hh`
* `b_h`

Then all contributions for each shared parameter are summed.

That entire process is:

> **BPTT**

---

# ⭐ Part 24 — BPTT With Loss at Every Time Step

Now suppose every time step has a loss:

```text id="bptt-eachloss-1"
h1 → L1
↓
h2 → L2
↓
h3 → L3
```

For `h2`:

> it receives gradient from `L2`

and also:

> gradient coming backward from `h3`, which may include `L3`.

So:

```text id="bptt-eachloss-2"
dL/dh2
=
local gradient from L2
+
future gradient through h3
```

For `h1`:

```text id="bptt-eachloss-3"
dL/dh1
=
local gradient from L1
+
future gradient through h2
```

This recursive structure is essential.

---

# 🧠 Part 25 — Why “Through Time”?

The name sometimes causes confusion.

The gradient does not literally travel backward through physical time.

We have simply created a computational graph where:

```text id="bptt-time-1"
earlier sequence position
→ later sequence position
```

Backpropagation traverses that dependency graph in reverse:

```text id="bptt-time-2"
later position
→ earlier position
```

Hence:

> backpropagation **through the unrolled time dimension**.

---

# ⚠️ Part 26 — BPTT Does Not Update Weights at Every Time Step

A common misconception:

```text id="bptt-update-1"
t1 → compute gradient → immediately update W
t2 → compute gradient → update W again
t3 → ...
```

That is not what standard BPTT means.

Typically:

```text id="bptt-update-2"
Forward through sequence
↓
Compute loss
↓
Backward through unrolled graph
↓
Accumulate gradients for shared parameters
↓
Optimizer step
```

The optimizer then updates the parameter once for that training step/batch.

Exact training setups can vary, but conceptually:

> **BPTT computes gradients; the optimizer performs the update.**

---

# 🔗 Part 27 — Backpropagation vs BPTT

Ordinary backprop:

```text id="bptt-vs-1"
Loss
↓
Layer 3
↓
Layer 2
↓
Layer 1
```

BPTT:

```text id="bptt-vs-2"
Loss
↓
Time T
↓
Time T-1
↓
...
↓
Time 1
```

But mathematically:

```text id="bptt-vs-3"
Both
=
Chain Rule on a Computational Graph
```

So:

> BPTT is not a fundamentally different learning rule.

---

# ⏳ Part 28 — What Happens With Very Long Sequences?

Imagine:

`T = 10,000`

Then the unrolled computational graph has:

> 10,000 recurrent steps.

Backward may need to traverse a very long chain.

This creates two major problems.

### Problem 1 — Gradient Stability

Repeated multiplications can cause:

* vanishing gradients
* exploding gradients

### Problem 2 — Memory and Compute

Training must retain enough intermediate information from the forward pass for backward computation.

Longer sequences therefore require:

* more computation
* more activation/state memory

---

# 💾 Part 29 — Why Training Uses More Memory Than Streaming Inference

During streaming inference:

```text id="bptt-memory-1"
x_t + h_{t-1}
↓
h_t
```

Once `h_t` is obtained, older intermediate states may not be needed for simple forward inference.

But during training with full BPTT:

> backward needs intermediate hidden states.

So we often need:

```text id="bptt-memory-2"
h1
h2
h3
...
hT
```

or enough information to reconstruct them.

Therefore:

```text id="bptt-memory-3"
Inference
→ can keep mainly current state

Training
→ needs history for backward
```

This is an important systems distinction.

---

# 📖 Part 30 — Can We Avoid Backpropagating Through the Entire History?

Suppose a sequence has:

`100,000`

steps.

Full BPTT over all 100,000 steps may be:

* computationally expensive
* memory-intensive
* unstable

A practical approach is:

> **Truncated Backpropagation Through Time**

or:

# **TBPTT**

---

# ✂️ Part 31 — Truncated BPTT Intuition

Instead of backpropagating through:

```text id="bptt-truncate-1"
t1 → t2 → ... → t100000
```

we process shorter chunks.

For example:

```text id="bptt-truncate-2"
t1 → ... → t100
           │
      Backprop here

t101 → ... → t200
             │
        Backprop here
```

The hidden state can be carried forward between chunks, while the gradient graph is cut/detached at chosen boundaries.

---

# 🧠 Part 32 — What Does “Detach Hidden State” Mean?

Suppose:

```text id="bptt-detach-1"
Chunk 1
↓
h100
↓
Chunk 2
```

We may pass the numerical value of:

`h100`

into Chunk 2.

But detach it from the previous computational graph.

Conceptually:

```text id="bptt-detach-2"
Carry VALUE forward ✅

Carry GRADIENT history indefinitely ❌
```

This limits how far backward gradients travel.

---

# ⚖️ Part 33 — TBPTT Trade-Off

Benefits:

* lower memory
* cheaper backward pass
* practical for long streams

But cost:

> dependencies longer than the truncation window may receive limited or no direct gradient signal across the detached boundary.

So TBPTT trades:

```text id="bptt-trade-1"
Training Practicality
↑

Very Long Gradient Horizon
↓
```

---

# ⚠️ Part 34 — Hidden-State Memory vs Gradient Memory

This is subtle but extremely important.

With truncated BPTT:

> hidden state may still carry information forward beyond the truncation boundary.

But:

> gradients do not propagate backward beyond that detached boundary.

So:

```text id="bptt-trade-2"
Forward information flow
can continue

Backward credit assignment
is truncated
```

These are not the same thing.

---

# 🧠 Part 35 — What Is Credit Assignment?

Suppose an important input occurs at:

`t = 10`

and affects the correct output at:

`t = 100`.

Training needs to answer:

> “Which earlier computation deserves credit or blame for the final result?”

This is called:

> **credit assignment**

BPTT performs temporal credit assignment by sending gradients backward through the recurrent chain.

Long-range credit assignment is difficult because:

* paths are long
* gradients may vanish/explode
* truncation may cut the gradient path

This is a deep reason long-term dependency learning is difficult.

---

# 📐 Part 36 — Compact Mathematical View

For:

`h_t = f(h_{t-1}, x_t; θ)`

where:

`θ`

represents shared RNN parameters.

Suppose only the final loss:

`L_T`

exists.

The effect of an earlier hidden state `h_k` on the final hidden state involves:

`∂h_T/∂h_k`

which expands through chain rule:

`∂h_T/∂h_k`

`= ∂h_T/∂h_{T-1}`

`× ∂h_{T-1}/∂h_{T-2}`

`× ...`

`× ∂h_{k+1}/∂h_k`

The key thing to notice is:

> **a product of many Jacobians.**

That product is the heart of the long-term gradient problem.

---

# 🧠 Part 37 — You Do Not Need to Fear the Word “Jacobian”

A Jacobian is simply the multivariable version of:

> derivative.

If hidden state is a vector:

```text id="bptt-jacobian-2"
h_t = [h1, h2, ..., hH]
```

then:

`∂h_t/∂h_{t-1}`

describes:

> how changes in each previous hidden component affect each current hidden component.

For our understanding, the important thing is not memorizing the full matrix.

It is understanding:

> BPTT repeatedly multiplies these transformations as gradients move backward through time.

---

# 📖 Part 38 — A Useful Analogy

Imagine a message passed through 100 people.

Each person slightly modifies it.

Forward:

```text id="bptt-analogy-1"
Person 1
→ Person 2
→ Person 3
→ ...
→ Person 100
```

Now Person 100 realizes the final message is wrong.

We want to determine:

> which earlier changes caused the error.

So information about the mistake must travel backward:

```text id="bptt-analogy-2"
Person 100
→ Person 99
→ ...
→ Person 1
```

If the correction signal becomes weaker at every handoff:

> early people receive almost no useful feedback.

If it becomes amplified:

> the correction becomes unstable.

That is essentially the challenge BPTT faces.

---

# 🔥 Part 39 — Why `tanh` Matters

Vanilla RNN commonly uses:

`tanh`.

Its derivative:

`tanh'(z) = 1 - tanh²(z)`

The derivative lies between:

`0` and `1`

for finite real inputs, reaching at most 1 and becoming close to zero in saturated regions.

Therefore recurrent gradient paths may repeatedly multiply by values:

> smaller than 1.

That contributes to:

> vanishing gradients.

But remember:

> `W_hh` is also part of the repeated Jacobian.

So the full story depends on both:

* recurrent weights
* activation derivatives

---

# ⚠️ Part 40 — Don't Oversimplify to “tanh Causes Vanishing Gradient”

That statement is too simplistic.

The recurrent gradient behaves according to repeated products involving:

```text id="bptt-not-simple-1"
W_hh
×
activation derivatives
```

So vanishing/exploding behavior is determined by:

> the overall recurrent Jacobian dynamics.

`tanh` saturation contributes, but it is not the entire story.

---

# 🧠 Part 41 — Why Early Inputs May Not Learn

Suppose:

```text id="bptt-early-1"
x1 contains important information
```

but output loss occurs at:

`t = 100`.

For `x1`-related parameters to learn that relationship, gradient information must travel:

```text id="bptt-early-2"
L100
↓
h100
↓
h99
↓
...
↓
h1
```

If the gradient becomes approximately zero before reaching `h1`:

> the model receives almost no learning signal telling it how `x1` should have influenced the future.

So long-term dependencies are not only a:

> “memory problem”

They are also a:

> **learning / credit-assignment problem.**

---

# ⭐ Part 42 — This Distinction Is Crucial

People often say:

> “RNN forgets old information.”

That is only part of the story.

There are two related problems.

### Forward Problem

Can information survive through many hidden-state updates?

### Backward Problem

Can a useful gradient signal reach the earlier steps that need to learn?

So:

```text id="bptt-two-problems-1"
Long-Term Dependency Problem
=
Forward information retention
+
Backward credit assignment
```

This will be extremely important when we understand LSTM.

---

# 📐 Part 43 — Full BPTT Workflow

A typical conceptual workflow:

```text id="bptt-workflow-1"
1. Input Sequence
↓
2. Forward Recurrent Pass
↓
3. Store Intermediate Hidden States
↓
4. Compute Predictions
↓
5. Compute Sequence Loss
↓
6. Start From dL
↓
7. Backpropagate From Late → Early Time Steps
↓
8. Sum Gradients for Shared Parameters
↓
9. Optimizer Updates Parameters
```

---

# 🔄 Part 44 — Forward vs Backward Direction

Forward:

```text id="bptt-direction-1"
t1 → t2 → t3 → t4
```

Backward:

```text id="bptt-direction-2"
t4 → t3 → t2 → t1
```

Do not interpret that as reversing the input sequence.

The backward pass simply traverses:

> the computational dependencies in reverse.

---

# ⚠️ Part 45 — BPTT vs Bidirectional RNN

These are completely different concepts.

### BPTT

Training algorithm:

> gradients move backward through the unrolled computational graph.

### Bidirectional RNN

Architecture:

> processes the input sequence in both forward and reverse directions to obtain past + future context.

So:

```text id="bptt-bi-1"
BPTT
→ HOW RNN learns

Bidirectional RNN
→ HOW sequence is encoded
```

Do not confuse them.

---

# ⚠️ Part 46 — BPTT vs Reverse-Time Input

Another misconception:

> “BPTT means we feed the sequence backward.”

No.

Input may still be processed:

```text id="bptt-reverse-1"
x1 → x2 → x3
```

BPTT means gradient traversal:

```text id="bptt-reverse-2"
Loss
→ h3
→ h2
→ h1
```

Those are different things.

---

# 🧠 Part 47 — Why Shared Weights Make BPTT Special

Imagine an ordinary 3-layer network:

```text id="bptt-shared-1"
Layer 1 → W1

Layer 2 → W2

Layer 3 → W3
```

Each layer has separate parameters.

An unrolled RNN looks like:

```text id="bptt-shared-2"
Time 1 → W

Time 2 → SAME W

Time 3 → SAME W
```

Therefore gradients from different temporal depths all update:

> the same parameter.

This is the defining shared-parameter characteristic of BPTT.

---

# 🧮 Part 48 — Small Scalar Gradient Example

Consider simplified recurrence:

`h_t = w h_{t-1}`

Ignore input and activation only for this toy example.

Suppose:

`h3 = w h2`

`h2 = w h1`

`h1 = w h0`

Therefore:

`h3 = w³ h0`

Suppose:

`L = h3`

Then:

`dL/dh0 = w³`

If:

`w = 0.5`

then:

`dL/dh0 = 0.125`

For 10 steps:

`0.5^10 ≈ 0.00098`

Gradient vanishes.

If:

`w = 2`

then after 10 steps:

`2^10 = 1024`

Gradient explodes.

This toy example ignores many real RNN details but clearly shows:

> repeated recurrent multiplication.

---

# 🎯 Part 49 — What Should You Remember Before the Next Lecture?

Do not memorize a dozen derivative equations.

Understand this chain:

```text id="bptt-remember-1"
RNN Reuses Weights Across Time
↓
Unroll It
↓
Now It Is a Computational Graph
↓
Apply Ordinary Chain Rule
↓
Hidden-State Gradients Flow Backward Across Time
↓
Shared Parameters Receive Contributions From Every Use
↓
Long Paths Mean Repeated Multiplication
↓
Gradient Can Shrink or Grow
```

That is BPTT.

---

# ⚡ Quick Recall Table

| Concept            | Meaning                                        |
| ------------------ | ---------------------------------------------- |
| BPTT               | Backpropagation on an unrolled RNN             |
| Unrolling          | Expanding recurrence across time               |
| `dL/dh_t`          | Gradient reaching hidden state `t`             |
| Local gradient     | Contribution from current-time loss            |
| Future gradient    | Contribution arriving through later states     |
| Shared weights     | Same parameter used at multiple times          |
| `dL/dW_hh`         | Sum of temporal contributions                  |
| Long gradient path | Repeated recurrent Jacobian multiplication     |
| Vanishing gradient | Gradient becomes very small                    |
| Exploding gradient | Gradient becomes very large                    |
| TBPTT              | Limit backward horizon to shorter chunks       |
| Credit assignment  | Deciding which earlier steps caused later loss |

---

# 📐 Mathematical Summary

Forward:

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

`h_t = tanh(a_t)`

---

Hidden gradient conceptually:

`dL/dh_t`

`= local loss contribution`

`+ future hidden-state contribution`

---

Shared recurrent weight gradient:

`dL/dW_hh = Σ_t (dL/da_t) h_{t-1}^T`

---

Input weight gradient:

`dL/dW_xh = Σ_t (dL/da_t) x_t^T`

---

Bias gradient:

`dL/db_h = Σ_t dL/da_t`

---

Long-range hidden dependency:

`∂h_T/∂h_k`

contains a product of recurrent Jacobians:

`∂h_T/∂h_{T-1} × ... × ∂h_{k+1}/∂h_k`

---

# 🎤 30-Second Interview Answer

> **Backpropagation Through Time is the application of ordinary backpropagation to an RNN after the recurrent computation is unrolled across time. Gradients flow from later time steps toward earlier hidden states using the chain rule. Because the same recurrent parameters are reused at every time step, their final gradients are the sum of contributions from all temporal uses. For long sequences, this creates products of many recurrent Jacobians, which can cause gradients to vanish or explode and make long-term credit assignment difficult.**

---

# 🧠 Final Mental Model

Forward:

```text id="bptt-final-1"
x1 → h1 → h2 → h3 → ... → hT → Loss
```

Backward:

```text id="bptt-final-2"
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

```text id="bptt-final-3"
Gradient continues backward
+
Gradient contributes to shared weights
```

And finally:

```text id="bptt-final-4"
All temporal contributions
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

# 🚀 Where We Go Next

We have now discovered something worrying.

For information at time `1` to learn from an error at time `100`, the gradient may need to pass through:

> nearly 100 recurrent transformations.

That means repeated multiplication by:

* recurrent weights
* activation derivatives

Depending on those products:

```text id="bptt-next-1"
Gradient
→ 0
```

or:

```text id="bptt-next-2"
Gradient
→ enormous
```

So our next lecture is:

# `04_Vanishing_And_Exploding_Gradients`

There we will answer:

> **Why exactly do gradients vanish or explode, how can we recognize each problem, why are long-term dependencies affected, and what can we do about it?**

That problem will then naturally create the need for:

> **LSTM**

---

# ⭐ Golden Rule

> **BPTT is ordinary chain-rule backpropagation through an RNN unrolled in time: gradients flow backward through hidden states, while every temporal use of a shared recurrent parameter contributes to one final parameter gradient.**
