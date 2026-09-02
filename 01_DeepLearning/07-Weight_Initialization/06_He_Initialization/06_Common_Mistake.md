# 🚫 Common Mistakes — He Initialization

> **Core Idea:** He Initialization is about choosing the **right ReLU-aware scale for random weights**. The most common mistakes come from confusing variance, weight count, `fan_in`, `fan_out`, and the role of ReLU.

---

## ❌ Mistake 1: Thinking He Initialization is only "random weights"

### Wrong ❌

> He Initialization just means using random values.

### Correct ✅

Randomness only breaks symmetry.

He adds **controlled scaling** designed for ReLU-family activations.

So:

Random Initialization  
↓  
Break Symmetry

He Initialization  
↓  
Break Symmetry  
+  
ReLU-Aware Scale

---

## ❌ Mistake 2: Thinking `2 / fan_in` is the value of every weight

### Wrong ❌

Suppose:

fan_in = 100

Then:

2 / fan_in = 0.02

So every weight should be:

0.02

### Correct ✅

`0.02` is the **target variance** of the weight distribution.

For He Normal:

Variance = 0.02

Therefore:

Std  
=  
sqrt(0.02)

≈  
0.141

Then different random weights are sampled.

Example:

+0.08  
-0.13  
+0.04  
+0.19  
-0.06

---

## ❌ Mistake 3: Confusing `fan_in × fan_out` with He variance

### Wrong ❌

> Since `fan_in × fan_out` gives the number of weights, He should divide by that value.

### Correct ✅

For a fully connected layer:

fan_in × fan_out  
↓  
Total Number of Weights

But He scaling depends on signal flow.

Common forward He:

Var(W) = 2 / fan_in

Backward-preserving form:

Var(W) = 2 / fan_out

So parameter count and parameter scale are different concepts.

---

## ❌ Mistake 4: Forgetting why the factor `2` exists

### Wrong ❌

> The `2` in He Initialization is just a constant chosen experimentally.

### Correct ✅

The factor `2` comes from ReLU's behavior.

For a roughly symmetric pre-activation distribution:

Approximately half positive  
Approximately half negative

ReLU:

Positive → Kept  
Negative → 0

So roughly part of the signal is removed.

He compensates by increasing the variance.

Conceptually:

1 / fan_in  
↓  
ReLU removes part of signal  
↓  
Compensate  
↓  
2 / fan_in

---

## ❌ Mistake 5: Saying ReLU always removes exactly 50% of activations

### Wrong ❌

> ReLU always zeros exactly half of all neurons.

### Correct ✅

The "half" idea is an **approximation** used in the derivation.

It assumes the pre-activation distribution is roughly symmetric around zero.

In a real trained network, the percentage of positive and negative activations may differ.

So say:

> **Approximately half under common initialization assumptions.**

Not:

> **Exactly 50% always.**

---

## ❌ Mistake 6: Thinking `2 / fan_in` perfectly preserves both forward and backward variance

### Wrong ❌

> He with `2 / fan_in` guarantees stable activations and gradients in both directions.

### Correct ✅

`2 / fan_in` primarily targets:

> **Forward activation variance**

For backward gradient variance, the corresponding condition is approximately:

Var(W) = 2 / fan_out

So:

fan_in  
↓  
Forward Preservation

fan_out  
↓  
Backward Preservation

---

## ❌ Mistake 7: Ignoring `fan_out` completely

### Wrong ❌

> `fan_out` has no relevance in He Initialization.

### Correct ✅

`fan_out` matters when reasoning about **backpropagation**.

Simplified backward condition:

Var(gradient_previous)
≈
fan_out × Var(W) × 1/2 × Var(gradient_next)

To approximately preserve gradient variance:

Var(W) ≈ 2 / fan_out

So `fan_out` is important for backward signal flow.

---

## ❌ Mistake 8: Thinking `fan_in` and `fan_out` modes are always equivalent

### Wrong ❌

> It doesn't matter whether I choose `fan_in` or `fan_out`.

### Correct ✅

They are equivalent only when:

fan_in ≈ fan_out

Example:

fan_in = 100  
fan_out = 100

Then:

2 / fan_in = 0.02

and:

2 / fan_out = 0.02

But if:

fan_in = 100  
fan_out = 500

then:

Forward:

2 / 100 = 0.02

Backward:

2 / 500 = 0.004

These are very different.

---

## ❌ Mistake 9: Thinking He is always better than Xavier

### Wrong ❌

> He Initialization is newer, so it should always replace Xavier.

### Correct ✅

The choice depends strongly on the activation function.

Useful default:

Tanh / Sigmoid  
↓  
Xavier

ReLU / Leaky ReLU  
↓  
He / Kaiming

He is designed specifically around ReLU-family behavior.

---

## ❌ Mistake 10: Using Xavier and He formulas interchangeably

### Wrong ❌

> Xavier and He are basically the same formula.

### Correct ✅

They use different variance rules.

Xavier:

Var(W) = 2 / (fan_in + fan_out)

He:

Var(W) = 2 / fan_in

for the common forward-preserving ReLU case.

For equal-width layers:

fan_in = fan_out = n

Xavier:

Var(W) = 1 / n

He:

Var(W) = 2 / n

So He uses a larger variance.

---

## ❌ Mistake 11: Thinking He completely eliminates Vanishing Gradients

### Wrong ❌

> If I use He Initialization with ReLU, gradients cannot vanish.

### Correct ✅

He provides a better **starting scale**.

It helps reduce vanishing-gradient risk, but problems can still occur because of:

- Very deep networks
- Dying ReLU
- Architecture
- Training dynamics
- Other optimization issues

So say:

✅ Reduces risk

not:

❌ Guarantees elimination

---

## ❌ Mistake 12: Thinking He completely prevents Exploding Gradients

### Wrong ❌

> He Initialization guarantees gradients will never explode.

### Correct ✅

He chooses a principled initial scale.

But exploding gradients can still appear later during training.

Possible additional techniques include:

- Gradient clipping
- Normalization
- Appropriate learning rates
- Better architecture

---

## ❌ Mistake 13: Confusing He Normal with He Uniform

### Wrong ❌

> He Normal and He Uniform sample weights in exactly the same way.

### Correct ✅

They use different distributions.

### He Normal

Mean = 0

Std:

sqrt(2 / fan_in)

### He Uniform

Weights are sampled from:

-a to +a

where:

a = sqrt(6 / fan_in)

Different distribution shapes, same basic variance goal.

---

## ❌ Mistake 14: Treating He Normal as having a fixed min/max

### Wrong ❌

> He Normal produces weights only inside a fixed range.

### Correct ✅

A Normal distribution is theoretically unbounded.

Most values will be near zero, but there is no strict minimum or maximum.

He Uniform, on the other hand, has a fixed range.

---

## ❌ Mistake 15: Thinking He starts weights closer to the final solution

### Wrong ❌

> He works because it guesses values closer to the optimal weights.

### Correct ✅

He has no knowledge of the final optimal parameters.

Its purpose is to give the network a **healthy starting signal scale**.

Think:

Stable Signal Flow

not:

Close to Final Solution

---

# 🧠 Don't Confuse These Concepts

| Concept | Meaning |
|---|---|
| `fan_in` | Number of incoming connections |
| `fan_out` | Number of outgoing connections |
| `fan_in × fan_out` | Total number of weights |
| `2 / fan_in` | Common forward He variance |
| `2 / fan_out` | Backward-preserving He variance |
| Factor `2` | Compensates for ReLU gating |
| He Normal | Normal-distributed random weights |
| He Uniform | Uniformly distributed random weights |

---

# ⚡ Quick Mental Model

ReLU  
↓  
Negative Activations → 0  
↓  
Part of Signal Removed  
↓  
Need Compensation  
↓  
He Initialization

Forward:

fan_in  
↓  
2 / fan_in  
↓  
Healthy Activation Scale

Backward:

fan_out  
↓  
2 / fan_out  
↓  
Healthy Gradient Scale

---

# 🎯 Interview Trap

### Weak Answer ❌

> "He Initialization uses `2 / fan_in` because ReLU needs bigger weights."

This is incomplete.

### Strong Answer ✅

> **He Initialization uses a larger variance because ReLU zeroes negative activations, reducing the propagated signal. Under common assumptions, this leads to a forward-preserving variance of `2 / fan_in`. During backpropagation, a corresponding `2 / fan_out` condition can be used to preserve gradient variance.**

---

# ⭐ Golden Rule

> **He Initialization is not just "bigger Xavier." It is a ReLU-aware initialization strategy whose scaling comes directly from how ReLU affects forward activations and backward gradients.**