# 📘 Lecture 06 — He Initialization

> **Core Idea:** He Initialization chooses a larger, ReLU-aware weight scale so that activations do not shrink too aggressively when ReLU sets many negative values to zero.

---

# 🎯 Learning Objectives

By the end of this lecture, you will understand:

- What He Initialization is
- Why Xavier is not always ideal for ReLU
- Why ReLU changes signal variance
- Why He Initialization commonly uses `fan_in`
- Why the variance is:

Var(W) = 2 / fan_in

- He Normal vs He Uniform
- Why He works well with ReLU-family activations
- How He helps reduce vanishing/exploding gradient risk
- How He differs from Xavier Initialization
- How to use He Initialization in PyTorch and TensorFlow

---

# 1️⃣ Why Do We Need He Initialization?

We already learned:

Zero Initialization  
↓  
Symmetry Problem  
↓  
Random Initialization  
↓  
Symmetry Broken  
↓  
But Scale Matters  
↓  
Xavier Initialization  
↓  
Controlled Variance

Xavier works well when we want to preserve signal variance across layers.

But ReLU introduces an important behavior:

For ReLU:

ReLU(x) = max(0, x)

So:

x > 0 → ReLU(x) = x  
x < 0 → ReLU(x) = 0

That means negative activations are removed.

Conceptually:

Input Activations  
↓  
Positive Values → Kept  
Negative Values → Set to 0

This changes the variance of the signal.

So ReLU needs a slightly different initialization strategy.

That is the motivation for:

> **He Initialization**

---

# 2️⃣ What Is He Initialization?

He Initialization is a weight initialization method designed primarily for **ReLU-family activation functions**.

It is also called:

- **Kaiming Initialization**
- **He/Kaiming Initialization**

Its main goal is:

> Maintain a healthy activation scale when ReLU removes a significant portion of negative activations.

A common He variance is:

Var(W) = 2 / fan_in

The corresponding standard deviation is:

Std(W) = sqrt(2 / fan_in)

---

# 3️⃣ Quick Revision — What Is `fan_in`?

`fan_in` means:

> Number of inputs coming into each output neuron in a layer.

Example:

100 input neurons  
↓  
50 output neurons

Then:

fan_in = 100  
fan_out = 50

Total number of weights:

100 × 50 = 5000

Important:

fan_in × fan_out  
↓  
Total number of weights

But:

fan_in  
↓  
Used in the common He initialization formula

---

# 4️⃣ Why Is Xavier Not Ideal for ReLU?

Recall the common Xavier variance:

Var(W) = 2 / (fan_in + fan_out)

Xavier tries to balance forward and backward signal variance.

That works well for activations such as `tanh`, where the signal is not systematically throwing away one side of the distribution.

But ReLU behaves differently.

Suppose the pre-activation values are roughly centered around zero:

-1.2  
+0.8  
-0.5  
+1.1  
-0.7  
+0.4

After ReLU:

0  
0.8  
0  
1.1  
0  
0.4

Many values become zero.

So part of the original signal disappears.

Conceptually:

Before ReLU:

Positive + Negative Values

After ReLU:

Positive Values + Many Zeros

This reduces the amount of signal being propagated.

---

# 5️⃣ Why Does ReLU Reduce Variance?

Suppose the pre-activation distribution is roughly symmetric around zero.

That means approximately:

50% positive values  
50% negative values

ReLU does this:

Positive half → kept  
Negative half → zeroed

So approximately half the values stop contributing.

Very roughly:

Before ReLU:

Signal Energy ≈ 100%

After ReLU:

Signal Energy ≈ 50%

This is not an exact statement for every dataset or layer, but it gives the right intuition.

Therefore, if we use weights that are too small, the signal can shrink quickly across many ReLU layers.

---

# 6️⃣ He's Main Idea

He Initialization compensates for the signal reduction caused by ReLU.

Instead of using a smaller Xavier-like variance, it commonly uses:

Var(W) = 2 / fan_in

Notice the factor:

2

That factor helps compensate for the fact that ReLU discards roughly half of the activations.

Conceptually:

ReLU removes about half the signal  
↓  
Increase initial weight variance  
↓  
Preserve healthier activation scale

---

# 7️⃣ Where Does `2 / fan_in` Come From?

Let's build the intuition.

For one neuron:

z = w₁x₁ + w₂x₂ + ... + wₙxₙ

Approximately:

Var(z)
≈
fan_in × Var(w) × Var(x)

Without ReLU, to roughly preserve variance:

fan_in × Var(w) ≈ 1

So:

Var(w) ≈ 1 / fan_in

But ReLU removes roughly half of the signal.

So after ReLU, the variance is approximately reduced.

To compensate:

fan_in × Var(w) × 1/2 ≈ 1

Now solve for Var(w):

Var(w) ≈ 2 / fan_in

That is the central intuition behind He Initialization.

---

# 8️⃣ Step-by-Step Derivation Intuition

Start with:

Var(output before activation)
≈
fan_in × Var(w) × Var(input)

Then ReLU keeps roughly half the signal.

So approximately:

Var(output after ReLU)
≈
1/2 × fan_in × Var(w) × Var(input)

We want:

Var(output after ReLU)
≈
Var(input)

Therefore:

1/2 × fan_in × Var(w) ≈ 1

Multiply both sides by 2:

fan_in × Var(w) ≈ 2

Therefore:

Var(w) ≈ 2 / fan_in

This is the key result.

---

# 9️⃣ Numerical Example

Suppose:

fan_in = 100

Then:

Var(W)
=
2 / 100

=
0.02

Standard deviation:

Std(W)
=
sqrt(0.02)

≈
0.141

So for He Normal:

W ~ Normal(0, 0.141²)

or more simply:

Mean = 0  
Std ≈ 0.141

---

# 🔟 Another Example

Suppose:

fan_in = 1000

Then:

Var(W)
=
2 / 1000

=
0.002

Standard deviation:

Std(W)
=
sqrt(0.002)

≈
0.0447

Notice:

More Inputs  
↓  
Smaller Individual Weight Scale

This is similar to Xavier's general philosophy.

But He uses a ReLU-aware scale.

---

# 1️⃣1️⃣ He Normal Initialization

He Normal samples weights from a Normal distribution.

Conceptually:

W ~ Normal(
    mean = 0,
    variance = 2 / fan_in
)

Standard deviation:

Std(W)
=
sqrt(2 / fan_in)

Example:

fan_in = 100

Then:

Std ≈ 0.141

So random weights might look like:

+0.08  
-0.13  
+0.21  
-0.04  
+0.11  
...

The values remain random.

He only controls their scale.

---

# 1️⃣2️⃣ He Uniform Initialization

He can also use a Uniform distribution.

For a uniform distribution:

W ~ Uniform(-a, +a)

A common He Uniform range is:

a = sqrt(6 / fan_in)

Why?

Because a uniform distribution from `-a` to `+a` has:

Variance = a² / 3

We want:

Variance = 2 / fan_in

So:

a² / 3
=
2 / fan_in

Therefore:

a²
=
6 / fan_in

and:

a
=
sqrt(6 / fan_in)

---

# 1️⃣3️⃣ He Uniform Example

Suppose:

fan_in = 100

Then:

a
=
sqrt(6 / 100)

=
sqrt(0.06)

≈
0.245

So weights are sampled from:

-0.245 to +0.245

That means:

W ~ Uniform(-0.245, +0.245)

---

# 1️⃣4️⃣ He Normal vs He Uniform

| He Normal | He Uniform |
|---|---|
| Normal distribution | Uniform distribution |
| Mean centered around 0 | Symmetric bounded range |
| Std = `sqrt(2 / fan_in)` | Range = `±sqrt(6 / fan_in)` |
| No strict hard min/max | Fixed minimum and maximum |
| Same variance objective | Same variance objective |

Both are based on:

Var(W) = 2 / fan_in

---

# 1️⃣5️⃣ Why Does He Commonly Use Only `fan_in`?

This is an important question.

Xavier commonly uses:

fan_in + fan_out

because it tries to balance both:

Forward activations  
+  
Backward gradients

He's common derivation is focused primarily on preserving the **forward activation variance through ReLU**.

The forward variance depends on:

fan_in

because each output neuron sums `fan_in` weighted inputs.

So the common formula is:

Var(W)
=
2 / fan_in

This is often referred to as:

`fan_in` mode

---

# 1️⃣6️⃣ Does He Completely Ignore Backpropagation?

No.

That would be an oversimplification.

The initialization still affects gradient flow.

If forward activations remain at a healthy scale, it also helps the overall training dynamics.

But the standard He derivation is primarily motivated by preserving activation variance through ReLU.

So remember:

Xavier  
↓  
Balances fan_in and fan_out

He  
↓  
Commonly uses fan_in for ReLU-aware forward variance preservation

---

# 1️⃣7️⃣ Why the Factor 2?

This is probably the most important He interview question.

Without ReLU, a simple variance-preserving idea would be:

Var(W) ≈ 1 / fan_in

But ReLU approximately removes half the signal.

So we compensate by doubling the variance:

1 / fan_in  
↓  
ReLU loses roughly half  
↓  
Multiply by 2  
↓  
2 / fan_in

That is why the factor `2` appears.

---

# 1️⃣8️⃣ Example: Xavier vs He

Suppose:

fan_in = 100  
fan_out = 100

### Xavier

Var(W)
=
2 / (100 + 100)

=
2 / 200

=
0.01

Std:

sqrt(0.01)
=
0.1

### He

Var(W)
=
2 / 100

=
0.02

Std:

sqrt(0.02)
≈
0.141

So He uses a larger scale.

Comparison:

Xavier Std ≈ 0.10  
He Std     ≈ 0.141

Why?

Because He compensates for ReLU zeroing many negative activations.

---

# 1️⃣9️⃣ Why Would Xavier Cause Shrinking in a Deep ReLU Network?

Suppose:

fan_in = fan_out

Then Xavier gives roughly:

Var(W)
≈
1 / fan_in

But ReLU removes a significant portion of the signal afterward.

So the sequence can become:

Xavier Scale  
↓  
ReLU  
↓  
Some Signal Lost  
↓  
Next Layer  
↓  
ReLU  
↓  
More Signal Lost  
↓  
Activations gradually shrink

He increases the initial variance:

2 / fan_in

to compensate.

---

# 2️⃣0️⃣ He and ReLU

He Initialization is particularly associated with:

- ReLU
- Leaky ReLU
- related ReLU-family activations

The standard ReLU case is the easiest:

ReLU  
↓  
Negative values become zero  
↓  
Need larger initialization variance  
↓  
He Initialization

---

# 2️⃣1️⃣ What About Leaky ReLU?

Leaky ReLU does not completely zero negative values.

Instead:

x > 0 → x  
x < 0 → αx

where `α` is a small positive slope.

Example:

α = 0.01

So negative values still pass some signal.

Because of that, the ideal scaling can be adjusted based on the negative slope.

A more general gain factor can be used.

For now, remember:

> He/Kaiming Initialization supports ReLU-family activations, including Leaky ReLU, with appropriate adjustment.

---

# 2️⃣2️⃣ Does He Work with Sigmoid or Tanh?

It can technically be used, but it is generally not the natural choice.

He is designed around ReLU's behavior.

For:

- `tanh`
- sigmoid-style activations

Xavier is generally a more suitable default.

So the practical rule is:

Tanh / Sigmoid  
↓  
Xavier

ReLU / Leaky ReLU  
↓  
He

---

# 2️⃣3️⃣ Does He Solve Vanishing Gradients Completely?

No.

He Initialization helps provide a better starting scale for ReLU networks.

It reduces the risk that activations and gradients shrink rapidly because of poor initialization.

But vanishing gradients can still happen because of:

- Very deep networks
- Architecture
- Dying ReLU
- Training dynamics
- Other optimization issues

So say:

✅ He reduces the risk

not:

❌ He guarantees no vanishing gradients

---

# 2️⃣4️⃣ Does He Prevent Exploding Gradients?

Again, no guarantee.

He chooses a principled initial variance.

That helps avoid overly large or overly small initial signal scales.

But gradients can still become unstable later during training.

Other techniques may still be useful:

- Gradient clipping
- Normalization
- Careful learning rates
- Better architectures

---

# 2️⃣5️⃣ Xavier vs He — Core Difference

The easiest way to understand the difference is:

Xavier asks:

> How can we balance signal variance across general layers?

He asks:

> How can we preserve signal variance when ReLU removes many negative activations?

So:

Xavier  
↓  
General variance balancing

He  
↓  
ReLU-aware variance compensation

---

# 2️⃣6️⃣ Xavier vs He — Formula Comparison

| Initialization | Common Variance |
|---|---|
| Xavier | `2 / (fan_in + fan_out)` |
| He | `2 / fan_in` |

For equal-sized layers:

fan_in = fan_out = n

Xavier:

2 / (n + n)

=
1 / n

He:

2 / n

So:

He Variance  
≈  
2 × Xavier Variance

for equal-width layers.

That larger scale compensates for ReLU's behavior.

---

# 2️⃣7️⃣ Xavier vs He — Activation Choice

| Activation | Common Initialization |
|---|---|
| Sigmoid | Xavier |
| Tanh | Xavier |
| ReLU | He |
| Leaky ReLU | He/Kaiming with proper gain |

A useful interview rule:

> **Xavier for tanh-style activations, He for ReLU-style activations.**

---

# 2️⃣8️⃣ He in PyTorch

PyTorch calls He Initialization:

> **Kaiming Initialization**

For Kaiming Normal:

```python
import torch.nn as nn
import torch.nn.init as init

layer = nn.Linear(100, 50)

init.kaiming_normal_(
    layer.weight,
    mode="fan_in",
    nonlinearity="relu"
)