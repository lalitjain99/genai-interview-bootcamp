# Partial Derivatives — When One Variable Isn't Enough

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- Why isn't an ordinary derivative enough for neural networks?
- What is a partial derivative?
- Why do we keep one variable fixed while changing another?
- How do partial derivatives help Gradient Descent?
- Why does every weight in a neural network have its own partial derivative?
- Why is this the first step toward Backpropagation?

---

# Chapter 1 — One Weight Was Easy

In the previous lecture, we learned that a derivative measures

> **How much one quantity changes when another quantity changes.**

For example,

suppose our neural network has only **one weight**.

```text
Loss = f(w)
```

The derivative tells us

```text
How much does the loss change

if we slightly change w?
```

Simple.

One weight.

One derivative.

No confusion.

---

# Chapter 2 — Real Neural Networks Aren't That Simple

Unfortunately,

real neural networks don't have one weight.

Even a tiny neural network might look like this.

```text
Input

   │

   ▼

Hidden Layer

 ○────○
 |\   |\
 | \  | \
 |  \ |  \

Output
```

Each connection has its own weight.

```text
w₁

w₂

w₃

w₄

...
```

Modern Large Language Models don't have four weights.

They have

```text
Millions

or even

Billions
```

of weights.

Now our loss no longer depends on a single variable.

Instead,

```text
Loss = f(w₁, w₂, w₃, ..., wₙ)
```

A new question appears.

> **Which weight should we change?**

---

# Chapter 3 — The Problem

Suppose our network has only two weights.

```text
Loss = f(w₁, w₂)
```

Imagine we slightly increase

```text
w₁
```

The loss changes.

Now suppose we instead change

```text
w₂
```

The loss changes again.

Notice something important.

The loss depends on **both** weights.

If we change both at the same time,

it becomes impossible to know

which weight caused the change.

We need a way to study

**one weight at a time.**

---

# Chapter 4 — A Simple Analogy

Imagine you're adjusting the recipe for tea.

There are two ingredients.

```text
Sugar

Milk
```

Suppose someone asks,

> "How does increasing sugar affect the sweetness?"

Can you answer the question if you also change the amount of milk at the same time?

No.

You wouldn't know which ingredient caused the difference.

Instead,

you perform a fair experiment.

```text
Keep Milk Fixed

↓

Only Change Sugar
```

Now you can clearly observe

how sugar affects the taste.

Later,

you do the opposite.

```text
Keep Sugar Fixed

↓

Only Change Milk
```

This is exactly the idea behind a **partial derivative**.

---

# Chapter 5 — The Meaning of a Partial Derivative

Suppose

```text
Loss = f(w₁, w₂)
```

The partial derivative

$$
\frac{\partial L}{\partial w_1}
$$

asks one very specific question.

> **If I change only \(w_1\), while keeping \(w_2\) fixed, how much does the loss change?**

Similarly,

$$
\frac{\partial L}{\partial w_2}
$$

asks

> **If I change only \(w_2\), while keeping \(w_1\) fixed, how much does the loss change?**

Notice the word

**partial**.

We're studying only **part** of the problem,

not all variables at once.

---

# Chapter 6 — Why Neural Networks Need Partial Derivatives

Imagine a neural network with

```text
1,000,000 weights.
```

For every weight,

we want to know

```text
If I change only this weight,

how much will the loss change?
```

That means we need

```text
∂L/∂w₁

∂L/∂w₂

∂L/∂w₃

...

∂L/∂w₁₀₀₀₀₀₀
```

Every weight gets its own partial derivative.

This tells us

which weights are helping,

which are hurting,

and which barely affect the loss.

---

# Chapter 7 — Connecting This to Gradient Descent

Earlier,

Gradient Descent answered the question

> **Which direction should we move?**

Partial derivatives answer a different question.

> **How sensitive is the loss to each individual weight?**

Suppose

```text
∂L/∂w₁ = 12
```

A large value means

changing **w₁** has a significant effect on the loss.

Now suppose

```text
∂L/∂w₂ = 0.001
```

Changing **w₂** barely affects the loss.

Gradient Descent uses this information to decide

how each weight should be updated.

---

# Chapter 8 — Looking Ahead

We've now learned something important.

A neural network with millions of weights produces

millions of partial derivatives.

That raises another question.

> **How do we organize all these partial derivatives?**

Instead of treating them individually,

mathematicians combine them into a single object.

This object is called the

**Gradient Vector**.

That is exactly what we'll study in the next lecture.

---

# Key Takeaways

- Ordinary derivatives work for functions with one variable.
- Neural networks have millions of weights, so they require partial derivatives.
- A partial derivative measures the effect of changing one variable while keeping all others fixed.
- Every weight in a neural network has its own partial derivative.
- Partial derivatives tell Gradient Descent how sensitive the loss is to each weight.
- The collection of all partial derivatives forms the foundation for the Gradient Vector and Backpropagation.