# 📘 Lecture 04 — Vanishing & Exploding Gradients

> **Core Idea:** In deep neural networks, gradients are repeatedly multiplied as they move backward through layers. If those values are consistently smaller than 1, gradients can shrink toward zero. If they are consistently larger than 1, gradients can grow very large.

---

# 🎯 Learning Objectives

By the end of this lecture, you will understand:

- What the vanishing gradient problem is.
- What the exploding gradient problem is.
- Why gradients are multiplied during backpropagation.
- Why deep networks are more vulnerable.
- How activation functions affect gradient flow.
- How poor weight initialization contributes to the problem.
- What symptoms you observe during training.
- Why Xavier and He Initialization are designed to help.

---

# 1️⃣ First, What Is a Gradient?

A gradient tells us:

> **How much should a parameter change to reduce the loss?**

For a weight:

```text
Gradient = ∂Loss / ∂Weight
```

Gradient Descent updates the weight using:

```text
w_new = w_old - learning_rate × gradient
```

So the size of the gradient matters.

---

## If the gradient is reasonable

```text
Gradient = 0.2
```

The weight receives a meaningful update.

---

## If the gradient is extremely small

```text
Gradient = 0.00000001
```

The weight barely changes.

---

## If the gradient is extremely large

```text
Gradient = 100000
```

The update can become unstable.

---

# 2️⃣ Where Does the Problem Appear?

Consider a deep neural network:

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
Layer 4
  ↓
Output
```

During forward propagation:

```text
Input → Layer 1 → Layer 2 → Layer 3 → Layer 4 → Output
```

During backpropagation:

```text
Output → Layer 4 → Layer 3 → Layer 2 → Layer 1
```

The gradient must travel backward through every layer.

This is where repeated multiplication becomes important.

---

# 3️⃣ Why Are Gradients Multiplied?

Backpropagation uses the **Chain Rule**.

Suppose:

```text
x → Layer 1 → Layer 2 → Loss
```

We want:

```text
∂Loss / ∂x
```

Using the Chain Rule:

```text
∂Loss / ∂x
=
∂Loss / ∂Layer2
×
∂Layer2 / ∂Layer1
×
∂Layer1 / ∂x
```

Notice what happens:

> Gradients are multiplied together.

For deeper networks:

```text
Gradient
=
g1 × g2 × g3 × g4 × ... × gn
```

This repeated multiplication is the key reason gradients can vanish or explode.

---

# 4️⃣ Vanishing Gradient Problem

A **vanishing gradient** occurs when gradients become smaller and smaller as they propagate backward.

Suppose each layer contributes approximately:

```text
0.5
```

Then:

```text
1 layer:

0.5
```

Two layers:

```text
0.5 × 0.5
= 0.25
```

Four layers:

```text
0.5 × 0.5 × 0.5 × 0.5
= 0.0625
```

Ten layers:

```text
0.5^10
≈ 0.00098
```

Twenty layers:

```text
0.5^20
≈ 0.00000095
```

The gradient keeps shrinking.

---

# 📉 Visualizing Vanishing Gradients

```text
Output Layer
Gradient = 1
      ↓
Layer 5
0.5
      ↓
Layer 4
0.25
      ↓
Layer 3
0.125
      ↓
Layer 2
0.0625
      ↓
Layer 1
0.03125
```

As the network becomes deeper:

```text
Gradient → 0
```

This is the **Vanishing Gradient Problem**.

---

# 5️⃣ Why Is Vanishing Gradient Bad?

If the gradient becomes almost zero:

```text
w_new = w_old - learning_rate × tiny_gradient
```

Example:

```text
learning_rate = 0.01
gradient      = 0.000001
```

Then:

```text
update = 0.01 × 0.000001
       = 0.00000001
```

The weight barely changes.

---

## Result

Earlier layers may learn extremely slowly.

```text
Output Layers → Learning
Middle Layers → Slow Learning
Early Layers  → Almost No Learning
```

This is particularly damaging because early layers are responsible for learning useful lower-level representations.

---

# 6️⃣ Exploding Gradient Problem

Now suppose gradients are repeatedly multiplied by values larger than 1.

For example:

```text
2 × 2 × 2 × 2
= 16
```

Ten layers:

```text
2^10
= 1024
```

Twenty layers:

```text
2^20
= 1,048,576
```

The gradient becomes extremely large.

This is the:

> **Exploding Gradient Problem**

---

# 📈 Visualizing Exploding Gradients

```text
Output Layer
Gradient = 1
      ↓
Layer 5
2
      ↓
Layer 4
4
      ↓
Layer 3
8
      ↓
Layer 2
16
      ↓
Layer 1
32
```

With enough layers:

```text
Gradient → Huge Value
```

---

# 7️⃣ Why Is Exploding Gradient Bad?

Suppose:

```text
learning_rate = 0.01
gradient      = 100000
```

Then:

```text
update
=
0.01 × 100000
=
1000
```

The weight suddenly changes by a huge amount.

Training can become unstable.

---

## Symptoms

You may observe:

```text
Loss suddenly becomes very large

Loss oscillates wildly

Weights become extremely large

Training becomes unstable

NaN / Inf values may appear
```

---

# 8️⃣ Why Are Deep Networks More Vulnerable?

In shallow networks, there are fewer multiplications.

Example:

```text
0.5 × 0.5
= 0.25
```

Not too bad.

But in deep networks:

```text
0.5 × 0.5 × 0.5 × ... × 0.5
```

the effect compounds.

Similarly:

```text
1.5 × 1.5 × 1.5 × ... × 1.5
```

can grow rapidly.

---

# 🧠 Depth Amplifies the Problem

```text
More Layers
    ↓
More Gradient Multiplications
    ↓
More Chance of
Shrinking / Growing
    ↓
Vanishing / Exploding
```

---

# 9️⃣ Role of Activation Functions

Activation functions also affect gradients.

Recall that during backpropagation we multiply by the derivative of the activation function.

Conceptually:

```text
Gradient
    ×
Activation Derivative
    ×
Weight
```

If the activation derivative is consistently small, the gradient can shrink.

---

# 🔵 Sigmoid and Vanishing Gradients

Sigmoid:

```text
σ(x) = 1 / (1 + e^-x)
```

Its derivative is:

```text
σ'(x) = σ(x)(1 - σ(x))
```

The maximum derivative is only:

```text
0.25
```

So during backpropagation:

```text
gradient
×
0.25
×
0.25
×
0.25
×
...
```

can shrink quickly.

---

## Example

```text
0.25^5
≈ 0.00098
```

Therefore, deep networks using sigmoid can be particularly vulnerable to vanishing gradients.

---

# 🟣 Tanh

Tanh generally has stronger gradients around zero than sigmoid, but its derivative can also become very small when the input is strongly positive or strongly negative.

So tanh can also suffer from vanishing gradients.

---

# 🟢 ReLU

ReLU:

```text
ReLU(x) = max(0, x)
```

Derivative:

```text
If x > 0 → 1
If x < 0 → 0
```

For positive activations:

```text
Derivative = 1
```

So ReLU often helps reduce the vanishing-gradient problem compared with sigmoid/tanh.

But ReLU has its own issue:

```text
Negative input
      ↓
Derivative = 0
      ↓
No gradient through that unit
```

You have already seen this as the **dying ReLU problem**.

---

# 🔟 Role of Weight Initialization

During backpropagation, gradients are also multiplied by weights.

Simplified idea:

```text
Gradient_previous
=
Gradient_current
×
Weight
×
Activation Derivative
```

So weight scale matters.

---

## If weights are too small

Example:

```text
weight ≈ 0.1
```

Repeated multiplication:

```text
0.1 × 0.1 × 0.1 × ...
```

can shrink gradients.

```text
Too Small Weights
       ↓
Vanishing Gradient Risk
```

---

## If weights are too large

Example:

```text
weight ≈ 5
```

Repeated multiplication:

```text
5 × 5 × 5 × ...
```

can grow gradients rapidly.

```text
Too Large Weights
       ↓
Exploding Gradient Risk
```

---

# 1️⃣1️⃣ Connection to Random Initialization

This explains why naive random initialization is not enough.

Suppose we randomly choose weights from:

```text
-100 to +100
```

Symmetry is broken.

✅ Good.

But the scale is huge.

❌ Bad.

Similarly:

```text
-0.000001 to +0.000001
```

also breaks symmetry.

✅ Good.

But the scale may be too small.

❌ Bad.

---

# 🎯 Therefore

```text
Randomness
    ↓
Break Symmetry

Proper Scale
    ↓
Control Signal Flow
```

We need both.

---

# 1️⃣2️⃣ Forward Propagation Also Matters

Vanishing/exploding problems are not only about gradients.

Poor initialization can first affect **activations during forward propagation**.

Example:

```text
Input
  ↓
Small Weights
  ↓
Small Activation
  ↓
Smaller Activation
  ↓
Even Smaller Activation
```

Or:

```text
Input
  ↓
Large Weights
  ↓
Large Activation
  ↓
Larger Activation
  ↓
Huge Activation
```

Then backpropagation inherits these unstable scales.

---

# 🔄 Forward + Backward View

```text
Poor Initialization
        ↓
Unstable Activation Scale
        ↓
Forward Propagation Problems
        ↓
Unstable Gradient Scale
        ↓
Backpropagation Problems
```

---

# 1️⃣3️⃣ Vanishing vs Exploding Gradient

| Feature | Vanishing Gradient | Exploding Gradient |
|---|---|---|
| Gradient size | Extremely small | Extremely large |
| Main effect | Learning becomes slow | Training becomes unstable |
| Early layers | Barely update | Can receive huge updates |
| Typical behavior | Training stagnates | Loss may oscillate/diverge |
| Common cause | Repeated factors < 1 | Repeated factors > 1 |
| Weight scale | Too small can contribute | Too large can contribute |

---

# 1️⃣4️⃣ How Can We Detect Vanishing Gradients?

Possible signs:

```text
Training progresses very slowly

Early layers barely change

Gradient values near zero

Loss stops improving significantly

Later layers learn more than early layers
```

If you inspect gradients layer-by-layer, you might see:

```text
Output Layer → 0.4

Layer 4      → 0.08

Layer 3      → 0.01

Layer 2      → 0.0005

Layer 1      → 0.000001
```

This is a strong indication of vanishing gradients.

---

# 1️⃣5️⃣ How Can We Detect Exploding Gradients?

Possible signs:

```text
Loss suddenly jumps

Loss becomes NaN

Weights become huge

Gradients become extremely large

Training becomes unstable
```

Example:

```text
Output Layer → 1

Layer 4      → 5

Layer 3      → 25

Layer 2      → 125

Layer 1      → 625
```

This suggests exploding gradients.

---

# 1️⃣6️⃣ How Do We Fix These Problems?

Several techniques can help.

For now, focus on these:

### ✅ Proper Weight Initialization

```text
Xavier Initialization
He Initialization
```

These choose weight scales more carefully.

---

### ✅ Better Activation Functions

Instead of relying heavily on sigmoid in deep hidden layers:

```text
ReLU
Leaky ReLU
```

can often provide better gradient flow.

---

### ✅ Batch Normalization

We already studied BatchNorm.

It helps stabilize the scale of activations during training.

That can make optimization more stable.

---

### ✅ Gradient Clipping

For exploding gradients, one common technique is:

> Limit gradients if they become too large.

Conceptually:

```text
Gradient = 1000
Threshold = 5

After clipping:

Gradient ≈ 5
```

We will not go deep into this technique here, but remember:

> Gradient clipping is mainly used to control exploding gradients.

---

# 1️⃣7️⃣ Why Xavier Initialization?

Suppose a layer receives many input values.

If we simply initialize every weight with the same arbitrary random scale, the variance of the output can change as it moves through layers.

Xavier initialization tries to choose the initial weight variance based on the number of neurons connected to the layer.

The goal is approximately:

```text
Input Variance
      ≈
Output Variance
```

So signals do not rapidly shrink or grow.

---

# 1️⃣8️⃣ Why He Initialization?

ReLU behaves differently because negative activations become zero.

```text
Negative Input
     ↓
ReLU
     ↓
0
```

This changes the amount of signal being passed forward.

He Initialization adjusts the scale of the weights specifically for ReLU-family activations.

Conceptually:

```text
ReLU Network
     ↓
Need Different Variance
     ↓
He Initialization
```

We will derive this properly in the upcoming lecture.

---

# 1️⃣9️⃣ Important Clarification

Do not think:

> "Vanishing gradients happen only because weights are small."

That is incomplete.

Several factors interact:

```text
Network Depth
      +
Weight Scale
      +
Activation Derivatives
      +
Architecture
      ↓
Gradient Behavior
```

Similarly, exploding gradients are not caused only by large weights.

The full gradient is a product of many terms.

---

# 2️⃣0️⃣ Simple Mathematical View

For a deep network, think of the gradient as:

```text
Gradient
≈
w1 × w2 × w3 × ... × wn
×
f'(z1) × f'(z2) × ... × f'(zn)
```

Now imagine many of these factors are:

```text
< 1
```

Then:

```text
Gradient → 0
```

Vanishing gradient.

---

If many are:

```text
> 1
```

Then:

```text
Gradient → Very Large
```

Exploding gradient.

That is the central intuition.

---

# 🧠 Complete Mental Model

```text
Deep Neural Network
        ↓
Backpropagation
        ↓
Chain Rule
        ↓
Repeated Multiplication
        ↓
┌───────────────────────┐
│                       │
Factors mostly < 1      Factors mostly > 1
│                       │
↓                       ↓
Gradient Shrinks        Gradient Grows
│                       │
↓                       ↓
Vanishing Gradient      Exploding Gradient
│                       │
↓                       ↓
Slow Learning           Unstable Training
└───────────────────────┘
        ↓
Need Better Signal Control
        ↓
Proper Initialization
        ↓
Xavier / He
```

---

# 🎤 Interview Perspective

If an interviewer asks:

> **Why do vanishing and exploding gradients happen?**

A strong answer is:

> **During backpropagation, gradients are propagated using the chain rule, which multiplies weights and activation derivatives across many layers. In deep networks, if these factors are repeatedly smaller than 1, the gradient can shrink toward zero, causing vanishing gradients. If they are repeatedly larger than 1, gradients can grow very large, causing exploding gradients. Proper initialization, suitable activation functions, normalization, and techniques such as gradient clipping can help control these problems.**

---

# ⚡ Vanishing Gradient in One Line

> **Gradients become progressively smaller as they move backward, so earlier layers learn very slowly.**

---

# ⚡ Exploding Gradient in One Line

> **Gradients become progressively larger as they move backward, causing unstable and excessively large parameter updates.**

---

# 📌 Key Takeaways

- Backpropagation uses the **Chain Rule**.
- Gradients are repeatedly multiplied across layers.
- Repeated factors smaller than 1 can cause **vanishing gradients**.
- Repeated factors larger than 1 can cause **exploding gradients**.
- Deep networks are more vulnerable because they involve more multiplications.
- Sigmoid and tanh can contribute to vanishing gradients because their derivatives can be small.
- Poorly scaled initialization can contribute to both vanishing and exploding gradients.
- Vanishing gradients cause slow or stalled learning in earlier layers.
- Exploding gradients can make training unstable.
- Proper initialization is one of the most important ways to maintain healthy activation and gradient scales.
- This directly motivates **Xavier and He Initialization**.

---

# ⭐ Golden Rule

```text
Backpropagation
      ↓
Repeated Multiplication
      ↓
Keep the Scale Healthy
      ↓
Avoid
Vanishing / Exploding
```

> **Good weight initialization is not about starting close to the final solution. It is about giving the network a stable starting point where signals and gradients can flow through many layers without rapidly shrinking or exploding.**

---

# ➡️ Next Topic

## 📘 Xavier Initialization

Now that we understand the problem, the next question is:

> **How can we mathematically choose the initial weight scale so that activations and gradients remain reasonably stable across layers?**

That is exactly what **Xavier Initialization** tries to solve.