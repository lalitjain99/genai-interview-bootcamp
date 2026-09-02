# 📘 Lecture 05 — Xavier Initialization

> **Core Idea:** Xavier Initialization chooses the initial weight scale based on the size of the layer so that activations and gradients stay at a more stable magnitude as they move through a deep network.

---

# 🎯 Learning Objectives

By the end of this lecture, you will understand:

- Why naive random initialization is not enough
- What Xavier Initialization is
- What `fan_in` and `fan_out` mean
- The intuition behind Xavier Initialization
- Why Xavier tries to preserve variance
- Xavier Normal vs Xavier Uniform
- Why Xavier works well with `tanh` and sigmoid-style activations
- Why Xavier is not usually the best choice for ReLU
- How Xavier helps reduce vanishing and exploding gradients
- How Xavier connects to He Initialization

---

# 1️⃣ Why Do We Need Xavier Initialization?

We already learned:

Zero Initialization  
↓  
Symmetry Problem  
↓  
Random Initialization  
↓  
Symmetry Broken  
↓  
But Weight Scale Matters

Random initialization gives different neurons different starting values.

That is good.

But random values can still be badly scaled.

For example:

Very Small Random Weights  
↓  
Activations Shrink  
↓  
Gradients Shrink  
↓  
Vanishing Gradient Risk

Very Large Random Weights  
↓  
Activations Grow  
↓  
Gradients Grow  
↓  
Exploding Gradient Risk

So we need something better than simply saying:

> "Use random numbers."

We need:

Randomness  
+  
Controlled Scale  
↓  
Better Initialization

That is the idea behind **Xavier Initialization**.

---

# 2️⃣ What Is Xavier Initialization?

Xavier Initialization is a weight initialization method that chooses the scale of the initial random weights based on the number of neurons connected to a layer.

It is also called:

- **Glorot Initialization**
- **Xavier/Glorot Initialization**

The main goal is:

> Keep the variance of activations and gradients reasonably stable across layers.

Conceptually:

Input Signal  
↓  
Layer 1  
↓  
Layer 2  
↓  
Layer 3  
↓  
Output

We do not want:

Signal → Smaller → Smaller → Smaller → 0

and we do not want:

Signal → Larger → Larger → Larger → Huge

We want something closer to:

Signal → Similar Scale → Similar Scale → Similar Scale

---

# 3️⃣ What Problem Is Xavier Trying to Solve?

Suppose a neuron receives many inputs.

The weighted sum is:

z = w₁x₁ + w₂x₂ + ... + wₙxₙ

If there are many inputs and each weight is large, the weighted sum can become too large.

If each weight is extremely small, the weighted sum can become too small.

Therefore, the correct weight scale depends partly on:

> How many values are being combined in the layer?

This leads to the idea of:

`fan_in`

and

`fan_out`

---

# 4️⃣ What Is fan_in?

`fan_in` means:

> The number of inputs coming into a neuron or layer.

Example:

Suppose a neuron receives 4 inputs:

x₁  
x₂  
x₃  
x₄  
↓  
Neuron

Then:

fan_in = 4

Another example:

100 input neurons  
↓  
Hidden Layer

Then:

fan_in = 100

---

# 5️⃣ What Is fan_out?

`fan_out` means:

> The number of outputs or neurons that a layer feeds into.

Suppose a layer contains 100 neurons and connects to a next layer containing 50 neurons.

Then:

fan_in = 100  
fan_out = 50

Conceptually:

100 neurons  
↓  
Current connection  
↓  
50 neurons

---

# 6️⃣ Why Do fan_in and fan_out Matter?

Imagine two layers.

### Layer A

fan_in = 10

### Layer B

fan_in = 1000

If we use the exact same random weight scale for both layers, Layer B is combining far more input terms.

That can cause the variance of the output to become much larger.

So weight scale should adapt to the size of the layer.

Xavier does exactly that.

---

# 7️⃣ The Main Xavier Idea

Xavier Initialization tries to choose weight variance so that:

Variance of Input  
≈  
Variance of Output

Conceptually:

Var(Input)  
≈  
Var(Output)

Why?

Because if variance shrinks layer by layer:

Var₁  
> Var₂  
> Var₃  
> Var₄  
> ...  
> 0

the network can suffer from vanishing signals and gradients.

If variance grows layer by layer:

Var₁  
< Var₂  
< Var₃  
< Var₄  
< ...  
< Huge

the network can become unstable.

So Xavier tries to keep variance relatively stable.

---

# 8️⃣ Xavier Variance Formula

A common Xavier formulation uses:

Variance(W) = 2 / (fan_in + fan_out)

So the standard deviation is approximately:

Std(W) = sqrt(2 / (fan_in + fan_out))

This gives us the scale of the random weights.

The important idea is not memorizing the formula alone.

Remember:

More Connections  
↓  
Smaller Individual Weight Scale

Fewer Connections  
↓  
Larger Individual Weight Scale

---

# 9️⃣ Why Does Xavier Use Both fan_in and fan_out?

There are two directions to think about:

Forward Propagation  
↓  
Activations

Backward Propagation  
↓  
Gradients

For forward propagation, `fan_in` matters because it affects how many input terms are being combined.

For backward propagation, `fan_out` also matters because of how gradients flow backward through the network.

So Xavier balances both.

That is why a common formula uses:

fan_in + fan_out

instead of only one of them.

### 🧠 Why Does Xavier Use `fan_in + fan_out` Instead of `fan_in × fan_out`?

This is an important question.

At first, it may seem natural to think:

fan_in × fan_out

because:

fan_in × fan_out = total number of weights in the layer

For example:

fan_in  = 4  
fan_out = 50

Total weights:

4 × 50 = 200

So why doesn't Xavier use:

1 / 200

to calculate the variance?

The reason is:

> **Activation and gradient variance depend on how many values are summed for each neuron, not on the total number of weights in the complete layer.**

---

## Forward Propagation Perspective

Consider one neuron.

Its weighted sum is:

z = w₁x₁ + w₂x₂ + ... + wₙxₙ

If the neuron receives `fan_in` inputs, then it is adding approximately:

fan_in

different weighted values.

Assuming the inputs and weights are roughly independent, the variance behaves approximately like:

Var(z)
≈
fan_in × Var(w) × Var(x)

Suppose we want the output variance to remain similar to the input variance:

Var(z) ≈ Var(x)

Then:

fan_in × Var(w) ≈ 1

Therefore:

Var(w) ≈ 1 / fan_in

So for forward propagation:

fan_in
↓
Number of terms being summed by one neuron
↓
Controls activation variance

Notice that:

fan_in × fan_out

does not appear here.

Why?

Because one neuron does not use every weight in the layer.

It only uses its own incoming weights.

---

## Example

Suppose:

fan_in  = 4  
fan_out = 50

There are:

4 × 50 = 200 total weights

But each hidden neuron receives only:

4 weights

For example:

Neuron 1:

z₁ = w₁₁x₁ + w₁₂x₂ + w₁₃x₃ + w₁₄x₄

Neuron 2:

z₂ = w₂₁x₁ + w₂₂x₂ + w₂₃x₃ + w₂₄x₄

...

Neuron 50 also receives only 4 inputs.

So each neuron is summing:

4 terms

not:

200 terms

Therefore, the forward-pass variance depends on:

fan_in = 4

not:

fan_in × fan_out = 200

---

## Backward Propagation Perspective

During backpropagation, gradients also pass through multiple connections.

A simplified relationship is:

Gradient_previous
≈
Weight × Gradient_next

For a neuron in the previous layer, gradients may arrive through approximately:

fan_out

different outgoing paths.

So the gradient variance behaves approximately like:

Var(gradient_previous)
≈
fan_out × Var(w) × Var(gradient_next)

To keep gradient variance stable:

fan_out × Var(w) ≈ 1

Therefore:

Var(w) ≈ 1 / fan_out

So we get two desirable conditions:

Forward propagation:

Var(w) ≈ 1 / fan_in

Backward propagation:

Var(w) ≈ 1 / fan_out

---

## The Problem

Unless:

fan_in = fan_out

we cannot satisfy both conditions perfectly with one single weight variance.

For example:

fan_in  = 50  
fan_out = 100

Forward preference:

Var(w) ≈ 1 / 50
       = 0.02

Backward preference:

Var(w) ≈ 1 / 100
       = 0.01

We need one variance for the weight matrix.

So Xavier chooses a compromise.

---

## Xavier's Balanced Solution

Xavier uses:

Var(W)
=
2 / (fan_in + fan_out)

For:

fan_in  = 50  
fan_out = 100

we get:

Var(W)
=
2 / 150

≈ 0.0133

So:

Backward target      Xavier       Forward target

     0.01   ←        0.0133       →   0.02

Xavier chooses a value between the forward and backward requirements.

---

## Why Addition?

Another way to write the Xavier formula is:

Var(W)
=
1 / ((fan_in + fan_out) / 2)

Here:

(fan_in + fan_out) / 2

is the average of the incoming and outgoing connectivity.

So Xavier effectively says:

Forward requirement
        +
Backward requirement
        ↓
Balance both directions
        ↓
Use average connectivity
        ↓
Var(W) = 2 / (fan_in + fan_out)

---

## Why Not Multiplication?

Suppose we instead used:

Var(W)
=
1 / (fan_in × fan_out)

For:

fan_in  = 4  
fan_out = 50

we would get:

Var(W)
=
1 / 200

=
0.005

But the approximate forward requirement is:

1 / fan_in
=
1 / 4
=
0.25

and the backward requirement is:

1 / fan_out
=
1 / 50
=
0.02

So:

0.005

is smaller than both.

That would make the weights unnecessarily small and could cause signals to shrink too aggressively.

---

## Larger Example

Suppose:

fan_in  = 1000  
fan_out = 1000

### If we used multiplication

Var(W)
=
1 / (1000 × 1000)

=
0.000001

Standard deviation:

Std(W)
=
sqrt(0.000001)

=
0.001

### Xavier

Var(W)
=
2 / (1000 + 1000)

=
0.001

Standard deviation:

Std(W)
≈
0.0316

The product-based approach would initialize the weights much smaller.

That could cause activations and gradients to shrink dramatically in a deep network.

---

# 🎯 Important Distinction

fan_in × fan_out
↓
Total number of weights
↓
Useful for parameter counting

But:

fan_in
↓
Forward signal variance

fan_out
↓
Backward gradient variance

Therefore:

fan_in + fan_out
↓
Balances forward and backward requirements
↓
Xavier Initialization

---

# ⭐ Golden Intuition

> **Xavier does not care about how many weights exist in the whole layer. It cares about how many values participate in the forward and backward signal flow for each neuron.**

That is why:

fan_in × fan_out

is useful for counting parameters,

while:

fan_in + fan_out

is useful for choosing the Xavier weight scale.

---

# 🔟 Simple Numerical Example

Suppose:

fan_in = 100  
fan_out = 100

Then:

Variance(W) = 2 / (100 + 100)

Variance(W) = 2 / 200

Variance(W) = 0.01

So:

Std(W) = sqrt(0.01)

Std(W) = 0.1

Therefore, the random weights should have a scale around:

0.1

Not:

100

and not:

0.0000001

This is **controlled randomness**.

---

# 1️⃣1️⃣ Another Example

Suppose:

fan_in = 400  
fan_out = 100

Then:

Variance(W) = 2 / (400 + 100)

Variance(W) = 2 / 500

Variance(W) = 0.004

So:

Std(W) ≈ 0.063

Notice:

More total connections  
↓  
Smaller weight scale

This helps prevent signals from becoming too large.

---

# 1️⃣2️⃣ Xavier Normal Initialization

Xavier can generate weights from a **Normal Distribution**.

Conceptually:

W ~ Normal(mean = 0, variance = 2 / (fan_in + fan_out))

The distribution is centered around zero.

Example shape:

             *
          *     *
        *         *
------*-------------*------
      negative     positive

Most weights are near zero, while larger positive or negative values are less common.

The exact numbers depend on `fan_in` and `fan_out`.

---

# 1️⃣3️⃣ Xavier Uniform Initialization

Xavier can also use a **Uniform Distribution**.

Instead of drawing weights from a bell-shaped distribution, weights are sampled evenly from a range.

Conceptually:

-a ≤ W ≤ +a

For Xavier Uniform, the range is commonly:

a = sqrt(6 / (fan_in + fan_out))

So:

W ~ Uniform(-a, +a)

Again, the main idea is:

> The range depends on the size of the layer.

---

# 1️⃣4️⃣ Xavier Normal vs Xavier Uniform

| Xavier Normal | Xavier Uniform |
|---|---|
| Uses Normal Distribution | Uses Uniform Distribution |
| Bell-shaped sampling | Even sampling within a range |
| Scale depends on fan_in/fan_out | Scale depends on fan_in/fan_out |
| Same main objective | Same main objective |

Both aim to keep activation and gradient scales more stable.

The core idea is more important than which distribution is used.

---

# 1️⃣5️⃣ Why Is the Mean Usually Zero?

Weights are typically initialized around zero.

For example:

-0.05  
+0.02  
-0.01  
+0.04

Why?

Because we want:

- Positive weights
- Negative weights
- Different neurons
- No strong initial directional bias

But remember:

> "Around zero" is very different from "all exactly zero."

All exactly zero:

0  
0  
0  
0

causes symmetry.

Random values centered around zero:

-0.03  
+0.01  
+0.04  
-0.02

break symmetry.

---

# 1️⃣6️⃣ Why Does Xavier Work Well with Tanh?

Tanh has output approximately in the range:

-1 to +1

and around zero, it behaves relatively smoothly.

Xavier attempts to keep activations from becoming too large or too small.

That helps keep tanh values away from extreme saturation regions.

Conceptually:

Proper Weight Scale  
↓  
Reasonable z Values  
↓  
Tanh stays more in useful region  
↓  
Healthier Gradients

---

# 1️⃣7️⃣ Xavier and Sigmoid

Xavier is also commonly associated with sigmoid-style networks because it was designed to help maintain signal scale through layers.

However, sigmoid still has an important limitation:

Its derivative is at most:

0.25

and becomes very small when saturated.

So Xavier can help improve the starting conditions, but it does not magically remove all vanishing-gradient issues of deep sigmoid networks.

Important distinction:

Xavier  
↓  
Helps Initialization

But:

Sigmoid Saturation  
↓  
Can Still Cause Vanishing Gradients

---

# 1️⃣8️⃣ Why Isn't Xavier Usually Preferred for ReLU?

ReLU behaves differently from tanh and sigmoid.

ReLU:

x > 0 → x  
x < 0 → 0

That means many negative activations are removed.

Conceptually:

Input Activations  
↓  
ReLU  
↓  
Negative Half Becomes 0

This changes the variance of the signal.

Xavier does not specifically compensate for this ReLU behavior.

Because of that, another initialization method was designed:

> **He Initialization**

He Initialization uses a larger variance that is better suited to ReLU-family activations.

---

# 1️⃣9️⃣ Xavier vs He — High-Level Preview

| Xavier | He |
|---|---|
| Commonly used with tanh/sigmoid-style activations | Designed for ReLU-family activations |
| Uses fan_in and fan_out in common form | Commonly uses fan_in |
| Controls signal variance | Compensates for ReLU behavior |
| Glorot Initialization | Kaiming Initialization |

We will study He Initialization separately.

---

# 2️⃣0️⃣ How Xavier Helps Forward Propagation

Suppose input activations have a reasonable variance.

Without proper initialization:

Layer 1  
↓  
Smaller Variance  
↓  
Layer 2  
↓  
Even Smaller Variance  
↓  
Layer 3  
↓  
Almost Zero

or:

Layer 1  
↓  
Larger Variance  
↓  
Layer 2  
↓  
Even Larger  
↓  
Layer 3  
↓  
Huge

Xavier tries to produce:

Layer 1  
↓  
Reasonable Scale  
↓  
Layer 2  
↓  
Reasonable Scale  
↓  
Layer 3  
↓  
Reasonable Scale

---

# 2️⃣1️⃣ How Xavier Helps Backpropagation

Backward propagation also involves repeated multiplication.

If gradients shrink layer by layer:

Gradient  
↓  
Smaller  
↓  
Smaller  
↓  
Vanishing

If gradients grow:

Gradient  
↓  
Larger  
↓  
Larger  
↓  
Exploding

By choosing an appropriate initial weight scale, Xavier helps reduce the chance that gradients immediately shrink or grow dramatically.

Important wording:

> Xavier helps reduce the risk of vanishing/exploding gradients at initialization.

It does not guarantee that these problems can never happen during training.

---

# 2️⃣2️⃣ Xavier Is About Variance Preservation

This is one of the most important interview points.

Do not describe Xavier simply as:

> "It gives random weights."

Random Initialization already does that.

The more complete idea is:

> **Xavier Initialization chooses the variance of random weights based on layer connectivity to preserve signal variance across layers.**

That is the key difference.

---

# 2️⃣3️⃣ Random Initialization vs Xavier

| Random Initialization | Xavier Initialization |
|---|---|
| Breaks symmetry | Breaks symmetry |
| Random values | Random values |
| Arbitrary scale possible | Controlled scale |
| Does not consider layer size | Considers fan_in/fan_out |
| Can cause unstable signal scale | Tries to preserve variance |

So Xavier is not replacing randomness.

It is improving it.

Random  
↓  
Controlled Random  
↓  
Xavier

---

# 2️⃣4️⃣ Why Is Xavier Better Than Using a Fixed Small Number?

Suppose someone initializes every layer with:

Std = 0.01

That may work for one network.

But imagine:

Layer A → 10 inputs  
Layer B → 100 inputs  
Layer C → 10,000 inputs

Using the same scale for every layer ignores the network structure.

Xavier adapts the weight scale according to the layer size.

So:

Different Layer Sizes  
↓  
Different Appropriate Scales

This is a major advantage.

---

# 2️⃣5️⃣ Does Xavier Eliminate Vanishing Gradients?

No.

This is an important interview trap.

Xavier can help maintain healthy signal scale at the start of training.

But vanishing gradients can still happen because of:

- Deep architectures
- Saturating activation functions
- Training dynamics
- Other architectural choices

So say:

✅ Xavier **reduces the risk**

not:

❌ Xavier **guarantees elimination**

---

# 2️⃣6️⃣ Does Xavier Eliminate Exploding Gradients?

Again:

No guarantee.

It helps prevent poor initialization from immediately producing extremely large signal scales.

But exploding gradients can still occur during training.

Other techniques may still be needed, such as:

- Gradient clipping
- Normalization
- Appropriate optimizer settings
- Architectural choices

---

# 2️⃣7️⃣ Xavier in PyTorch

PyTorch provides Xavier initialization directly.

Example:

```python
import torch.nn as nn
import torch.nn.init as init

layer = nn.Linear(100, 50)

init.xavier_uniform_(layer.weight)