# Gradient Vector — Combining Millions of Partial Derivatives

---

# Learning Objectives

By the end of this lecture, you should be able to answer:

- What is a Gradient?
- How is a Gradient different from a Derivative?
- How is a Gradient different from a Partial Derivative?
- Why do we combine all partial derivatives into a vector?
- Why is the Gradient called a vector?
- Why is the Gradient central to Deep Learning?

---

# Chapter 1 — We Have a New Problem

In the previous lecture, we learned that every weight has its own partial derivative.

For a neural network with two weights,

we calculate

$$
\frac{\partial L}{\partial w_1}
\quad\text{and}\quad
\frac{\partial L}{\partial w_2}
$$

For three weights,

we calculate three partial derivatives.

For one million weights,

we calculate one million partial derivatives.

Immediately, a practical problem appears.

> **How do we manage millions of partial derivatives?**

Writing them individually would quickly become impossible.

We need a better way.

---

# Chapter 2 — Think of Student Marks

Imagine a student has scores in five subjects.

```text
Math       = 92

Physics    = 88

Chemistry  = 95

English    = 81

Biology    = 90
```

Instead of treating each score separately,

we often store them together.

```text
[92, 88, 95, 81, 90]
```

This list represents the student's complete performance.

We don't lose any information.

We simply organize it better.

The same idea applies to partial derivatives.

---

# Chapter 3 — Collecting All Partial Derivatives

Suppose our neural network has three weights.

```text
w₁

w₂

w₃
```

We calculate

$$
\frac{\partial L}{\partial w_1}
$$

$$
\frac{\partial L}{\partial w_2}
$$

$$
\frac{\partial L}{\partial w_3}
$$

Instead of writing them separately,

we combine them into one vector.

$$
\nabla L =
\left[
\frac{\partial L}{\partial w_1},
\frac{\partial L}{\partial w_2},
\frac{\partial L}{\partial w_3}
\right]
$$

This entire collection is called the **Gradient**.

---

# Chapter 4 — Why Is It Called a Vector?

A vector is simply an ordered collection of numbers.

Examples include

```text
Position

(3, 5)
```

Velocity

```text
(12, -4)
```

Acceleration

```text
(2, 1)
```

Similarly,

the Gradient is also an ordered collection of numbers.

Each number corresponds to one weight.

```text
Gradient

(2.4, -0.8, 5.1)
```

Nothing mysterious.

The Gradient is simply a vector whose elements are partial derivatives.

---

# Chapter 5 — What Does the Gradient Tell Us?

Each partial derivative tells us

how sensitive the loss is to one weight.

The Gradient tells us

how sensitive the loss is to **all weights simultaneously**.

Think of it as a report card.

Instead of evaluating one student,

it evaluates every weight in the network.

For example,

```text
Gradient

(8.2, -0.3, 0.01)
```

This immediately tells us

- Weight 1 has a large influence.
- Weight 2 has a moderate influence.
- Weight 3 has almost no influence.

The Gradient summarizes the behavior of the entire network.

---

# Chapter 6 — Connecting This to Gradient Descent

Earlier, we learned the Gradient Descent update rule.

$$
w = w - \eta \nabla L
$$

At that time,

the symbol

$$
\nabla L
$$

may have looked mysterious.

Now we know exactly what it represents.

It is simply the vector containing all partial derivatives.

Gradient Descent subtracts this vector,

causing every weight to move in the direction that reduces the loss.

---

# Chapter 7 — A Small Example

Suppose

```text
Gradient

(4, -2)
```

This means

```text
∂L/∂w₁ = 4

∂L/∂w₂ = -2
```

Interpretation:

- Increasing **w₁** increases the loss.
- Increasing **w₂** decreases the loss.

Gradient Descent therefore

- decreases **w₁**
- increases **w₂**

The update happens independently for every weight,

using its corresponding partial derivative.

---

# Chapter 8 — Looking Ahead

We've now learned what the Gradient is.

But another question remains.

> **How do we actually compute millions of partial derivatives efficiently?**

A large neural network may contain

millions or even billions of weights.

Computing every partial derivative independently would be far too slow.

Fortunately,

there is a remarkably efficient algorithm that computes all of them in a single backward pass.

That algorithm is called

**Backpropagation**.

We'll study it in the next lecture.

---

# Key Takeaways

- A Gradient is the collection of all partial derivatives.
- Every weight contributes one partial derivative.
- The Gradient is represented as a vector.
- The Gradient tells us how sensitive the loss is to every weight simultaneously.
- Gradient Descent uses the Gradient Vector to update all weights at once.
- Understanding the Gradient prepares us to learn Backpropagation.

---

## Q: If every weight has a different partial derivative, why do we use the same learning rate for all weights?**

Answer:

The gradient already captures the direction and sensitivity for each weight.
The learning rate is a global scaling factor applied to all updates.
Therefore, different gradients still produce different update sizes, even with the same learning rate.
However, using a single learning rate is a simplification. Modern optimizers like AdaGrad, RMSProp, and Adam compute adaptive learning rates for each parameter, which often leads to faster and more stable training.