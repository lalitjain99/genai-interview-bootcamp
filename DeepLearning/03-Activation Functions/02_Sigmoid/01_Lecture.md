# Lecture 3 — Sigmoid Activation Function
## *The First Activation Function That Made Neural Networks Learn*

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to answer:

- Why couldn't the Step Function train a Neural Network?
- Why do Neural Networks need gradients?
- What problem was Sigmoid trying to solve?
- Why is Sigmoid differentiable?
- Why did Sigmoid become the first successful trainable activation function?
- Why is Sigmoid widely used for Binary Classification?
- Why does Sigmoid suffer from the Vanishing Gradient Problem?
- Why don't modern Deep Learning models use Sigmoid in hidden layers?


# Why Researchers Invented Sigmoid

---

# Chapter 1 — Where We Left Off

Let's continue our journey.

So far, we have built a Perceptron.

The Perceptron computes a weighted sum,

```text
z = Wx + b
```

and passes it through the **Step Function**.

The Step Function solved one major problem.

It converted a numerical score into a decision.

```text
Weighted Sum
        │
        ▼
Step Function
        │
        ▼
0 or 1
```

For the first time, a machine could make binary decisions.

This was revolutionary.

But researchers quickly discovered another problem.

The Perceptron could make decisions...

**but it could not learn from its mistakes.**

---

# 🤔 Think About It

Suppose a Neural Network predicts

```text
Dog
```

The correct answer is

```text
Cat
```

Clearly, the prediction is wrong.

The obvious solution seems simple.

> Change the weights.

But another question immediately appears.

- Which weight?
- Increase it?
- Decrease it?
- By how much?

The network has no way of answering these questions.

---

# Chapter 2 — Learning Requires Feedback

Imagine a student who scores **42 marks** in an exam.

The teacher simply says,

> "Improve."

The student immediately asks,

> Improve what?

- Mathematics?
- English?
- Science?
- Writing speed?

Without feedback,

improvement becomes nothing more than guessing.

Neural Networks faced exactly the same problem.

A wrong prediction alone isn't enough.

The network needs feedback explaining

**how each weight contributed to the error.**

Only then can it improve.

---

# 💡 First Principle

Learning requires more than knowing **whether you're wrong.**

It requires knowing

> **How should I change?**

This missing piece became the foundation of modern Deep Learning.

---

# Chapter 3 — Researchers Needed Direction

Researchers realized something important.

Instead of asking,

> "Is the prediction wrong?"

they asked a much better question.

> "If I change a weight slightly, how much does the output change?"

This single question introduced one of the most important ideas in Machine Learning.

**The Derivative.**

---

## Intuition Behind a Derivative

A derivative simply measures

```text
Small Change in Weight
            │
            ▼
How Much Output Changes
```

Think of it as feedback.

Large derivative

→ This weight has a strong influence.

Small derivative

→ This weight barely affects the prediction.

Now the optimizer finally has guidance.

Instead of randomly changing weights,

it knows

- which direction to move,
- and how much to move.

---

# Chapter 4 — Why the Step Function Could Never Learn

Let's revisit the Step Function.

Imagine standing on the flat part of its graph.

```text
        1
        │──────────────
        │
────────┼──────────────
        │
        │
        0
```

Take one tiny step.

Did the output change?

No.

Take another tiny step.

Still no change.

The slope is

```text
0
```

Now stand exactly at the threshold.

```text
0
↓
1
```

The graph jumps instantly.

The slope doesn't even exist.

So the derivative of the Step Function is

```text
0
Almost Everywhere

Undefined
At the Threshold
```

---

# 🚨 The Chain Reaction

No derivative means

```text
No Derivative
        │
        ▼
No Gradient
        │
        ▼
No Gradient Descent
        │
        ▼
No Backpropagation
        │
        ▼
No Learning
```

The Step Function could make decisions,

but it gave the optimizer **no useful feedback.**

That made training impossible.

---

# Chapter 5 — Designing a Better Activation Function

Researchers now had a clear goal.

They didn't want to replace the Step Function completely.

They wanted to improve it.

Imagine you're one of those researchers.

What properties would your ideal activation function have?

It should

✅ Make decisions

✅ Introduce non-linearity

✅ Be smooth

✅ Be differentiable

✅ Allow gradients to flow

✅ Be computationally efficient

In other words,

it should behave like the Step Function,

but also support learning.

Could such a function exist?

Yes.

Researchers invented one.

Its name was

# **Sigmoid**


# Chapter 6 — Meet the Sigmoid Function

Researchers finally had an activation function that satisfied most of their requirements.

Mathematically,

```text
                1
σ(z) = ----------------
        1 + e^(-z)
```

At first glance,

this equation looks intimidating.

Don't worry.

You don't need to memorize it.

Instead, let's understand **what it actually does.**

---

# 🤔 Think About It

Suppose the weighted sum is

```text
z = 100
```

Sigmoid outputs approximately

```text
0.99995
```

Now suppose

```text
z = -100
```

The output becomes

```text
0.00004
```

And if

```text
z = 0
```

Sigmoid outputs

```text
0.5
```

Interesting...

Unlike the Step Function,

Sigmoid doesn't suddenly jump from 0 to 1.

Instead,

it changes **smoothly**.

---

# 📈 Visualizing Sigmoid

```text
Output

1.0 |                           ●
    |                        ●
0.8 |                     ●
    |                  ●
0.6 |               ●
0.5 |------------●----------------
0.4 |         ●
    |      ●
0.2 |   ●
    | ●
0.0 +------------------------------------→ z
      -∞         0                +∞
```

Notice something important.

The output is **always between 0 and 1.**

It never goes below 0.

It never exceeds 1.

---

# 💡 First Principle

Sigmoid transforms

```text
(-∞ , +∞)
```

into

```text
(0 , 1)
```

Regardless of how large or small the weighted sum becomes,

the output always remains within this range.

---

# Chapter 7 — Why Does Sigmoid Look Like a Probability?

One of the most common interview questions is

> "Why is Sigmoid used in Binary Classification?"

Many candidates immediately answer,

> "Because Sigmoid outputs probabilities."

That answer is **not entirely correct.**

---

Sigmoid doesn't actually understand probability.

It simply outputs a value between

```text
0

and

1
```

Because of this range,

we can **interpret** the output as confidence.

For example,

```text
0.97
```

can be interpreted as

```text
97% confidence
```

that the input belongs to the positive class.

Likewise,

```text
0.08
```

can be interpreted as

```text
8% confidence
```

for the positive class.

---

# 📦 Real World Example

Imagine we're building a disease prediction model.

The network computes

```text
z = 5.3
```

Passing it through Sigmoid gives

```text
σ(z) = 0.995
```

We may interpret this as

```text
99.5% confidence

Patient is likely to have the disease.
```

Similarly,

```text
σ(z) = 0.04
```

means

```text
Only 4% confidence.

Patient is unlikely to have the disease.
```

This is exactly why Sigmoid became the standard choice for

- Disease Prediction
- Spam Detection
- Fraud Detection
- Binary Classification

---

# Chapter 8 — Why Sigmoid Changed Deep Learning

Now let's compare it with the Step Function.

The Step Function looked like a cliff.

```text
0
│
│______
       │
       │
       1
```

A tiny movement changed nothing.

Until suddenly,

everything changed.

---

Sigmoid behaves differently.

```text
        ●
      ●
    ●
  ●
●
```

Every small movement

creates a small change in output.

That means

the derivative exists everywhere.

---

# 🚀 What Does This Mean?

Now the optimizer finally receives useful feedback.

Instead of hearing

```text
I don't know where to move.
```

the derivative now says

```text
Move slightly left.

Good.

Now a little more.

Stop.

Move right.

Tiny adjustment.
```

The optimizer finally has a direction.

Gradient Descent becomes possible.

Backpropagation becomes possible.

Neural Networks can finally learn.

This was one of the biggest breakthroughs in AI.

---

# Chapter 9 — The Hidden Problem

Researchers celebrated.

For the first time,

deep neural networks could actually be trained.

Everything looked perfect.

Until networks became deeper.

---

Imagine a Neural Network with many hidden layers.

During Backpropagation,

gradients are multiplied layer after layer.

Suppose the gradient flowing through one Sigmoid neuron is

```text
0.2
```

After another layer,

```text
0.2 × 0.2 = 0.04
```

After another,

```text
0.04 × 0.2 = 0.008
```

After several layers,

```text
0.000001
```

The gradient has almost disappeared.

---

# 📉 The Vanishing Gradient Problem

```text
Layer 10
Gradient = 0.25

        │

Layer 9
Gradient = 0.06

        │

Layer 8
Gradient = 0.015

        │

Layer 7
Gradient = 0.003

        │

Layer 6
Gradient ≈ 0
```

Eventually,

the optimizer can barely update the weights.

Learning slows dramatically.

Sometimes,

it almost stops.

This became known as the

# **Vanishing Gradient Problem**

---

# 🤔 Why Does This Happen?

The derivative of Sigmoid is always less than 1.

During Backpropagation,

gradients from every layer are multiplied together.

Repeated multiplication by small numbers causes the gradient to shrink rapidly.

```text
Small Gradient
        │
        ▼
Multiply Across Layers
        │
        ▼
Smaller Gradient
        │
        ▼
Eventually Almost Zero
```

This makes it difficult for early layers to learn.

---

# 🌍 Engineering Insight

Imagine training GPT using Sigmoid in every hidden layer.

GPT contains dozens of Transformer blocks.

During Backpropagation,

gradients would keep shrinking as they move backward.

The earliest layers would receive almost no learning signal.

Training would become painfully slow.

This is one of the major reasons modern architectures replaced Sigmoid with

- ReLU
- GELU
- Swish

for hidden layers.

---

# 🎯 Interview Insight

If an interviewer asks,

> **Why did Sigmoid replace the Step Function?**

A strong answer is:

> The Step Function could make binary decisions but was not differentiable, making Gradient Descent impossible. Sigmoid introduced a smooth, differentiable activation that enabled gradients to flow, allowing Backpropagation and making Neural Networks trainable.

---

If asked,

> **Why don't modern Deep Learning models use Sigmoid in hidden layers?**

A strong answer is:

> Although Sigmoid is differentiable, its gradients become very small for large positive or negative inputs. During Backpropagation these small gradients multiply across layers, causing the Vanishing Gradient Problem and slowing training in deep networks.

---

# 🎯 Key Takeaways

- Sigmoid is a smooth approximation of the Step Function.
- It maps any input to a value between **0 and 1**.
- Its outputs can be interpreted as confidence scores.
- It is differentiable everywhere.
- Differentiability enables Gradient Descent and Backpropagation.
- Sigmoid became the foundation of trainable Neural Networks.
- In deep networks, Sigmoid suffers from the Vanishing Gradient Problem.
- Modern hidden layers therefore prefer ReLU, GELU, or similar activation functions.

---

# 🚀 Engineering Connection

Sigmoid solved one of the biggest problems in Deep Learning—

**it made learning possible.**

But as Neural Networks became deeper,

researchers discovered that **making learning possible isn't enough.**

Learning also needs to remain **fast and stable**.

That realization led to the next major breakthrough in activation functions.

---

# 📖 Coming Up Next

Sigmoid taught Neural Networks **how to learn**.

Now researchers asked a new question.

> Can we design an activation function that keeps gradients alive even in very deep networks?

The answer was one of the most influential activation functions in Deep Learning history.

# **ReLU (Rectified Linear Unit)**