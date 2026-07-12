# 06. AdaGrad (Adaptive Gradient)

---

# The Problem Momentum Could Not Solve

Momentum was a huge improvement over Gradient Descent.

It remembered previous movement.

It accelerated optimization.

It reduced oscillations.

Training became much smoother.

Problem solved?

**Not quite.**

Researchers soon discovered another limitation.

---

# Imagine Training a Neural Network

Suppose our neural network has only two weights.

```text
W1

W2
```

During training we observe something interesting.

---

## Weight 1

Every iteration receives a very large gradient.

```text
Iteration 1

Gradient = 120

Iteration 2

Gradient = 95

Iteration 3

Gradient = 140
```

---

## Weight 2

Hardly changes.

```text
Iteration 1

Gradient = 0.001

Iteration 2

Gradient = 0.003

Iteration 3

Gradient = 0.002
```

Yet our optimizer says

```text
Learning Rate

=

0.001
```

for both weights.

---

# Engineering Question

Does this make engineering sense?

Should both parameters learn at exactly the same speed?

Think about it.

One parameter is already changing rapidly.

The other is barely moving.

Why should they receive the same learning rate?

---

# Real-Life Analogy

Imagine two cars.

Car A

```text
Speed = 180 km/hr
```

Car B

```text
Speed = 15 km/hr
```

Would you press the accelerator equally hard on both?

Of course not.

The fast-moving car requires smaller adjustments.

The slow-moving car needs larger encouragement.

Optimization is no different.

---

# The Hidden Assumption

Every optimizer we have studied so far assumes

```text
One Learning Rate

↓

For Every Parameter
```

This assumption works reasonably well.

But is it the best possible strategy?

Researchers started asking a different question.

> **Should every parameter have its own learning rate?**

This was a revolutionary shift in thinking.

---

# A Better Question

Instead of asking

```text
What is the best learning rate?
```

they asked

```text
How can each parameter decide its own learning rate?
```

Notice the difference.

We're no longer searching for

```text
η = 0.001
```

We're searching for

```text
η₁

η₂

η₃

...

ηₙ
```

Every parameter can learn differently.

---

# How Should We Decide?

Suppose a parameter repeatedly receives large gradients.

```text
120

110

95

130

140
```

What does this tell us?

It tells us

> This parameter is already changing aggressively.

Should we continue taking large steps?

Probably not.

We should become more careful.

---

Now consider another parameter.

```text
0.001

0.002

0.001

0.003
```

This parameter barely changes.

Should we reduce its learning rate?

Probably not.

Perhaps we should allow it to learn faster.

---

# Engineering Requirement

We now have a clear requirement.

If a parameter repeatedly receives

```text
Large Gradients
```

↓

Its learning rate should decrease.

---

If a parameter repeatedly receives

```text
Small Gradients
```

↓

Its learning rate should remain larger.

---

Notice something important.

We are no longer making decisions based only on the **current gradient**.

We are looking at the **history of gradients**.

That is the key insight behind AdaGrad.

---

# What Information Should We Remember?

Momentum remembers

```text
Previous Velocity
```

AdaGrad asks a different question.

What if we remember

```text
Previous Gradients
```

instead?

Not to determine the direction.

But to determine the **learning rate**.

Very clever.

---

# The Core Philosophy

Momentum remembers movement.

AdaGrad remembers gradient history.

Momentum changes

```text
Direction of Movement
```

AdaGrad changes

```text
Step Size
```

These optimizers solve two completely different engineering problems.

---

# The Big Idea

Instead of

```text
One Learning Rate

↓

All Parameters
```

AdaGrad proposes

```text
Every Parameter

↓

Own Learning Rate
```

Parameters that have already moved a lot should slow down.

Parameters that rarely move should continue learning.

---

# Why Is It Called Adaptive Gradient?

Because the learning rate automatically **adapts**

based on the history of gradients.

No longer fixed.

No longer identical for every parameter.

The optimizer continuously adjusts the learning rate during training.

Hence the name

```text
Adaptive Gradient

↓

AdaGrad
```

---

# Engineering Insight

Notice how optimization algorithms are evolving.

Gradient Descent asked

> Which direction should I move?

---

Momentum asked

> Can I use my previous movement to make better decisions?

---

AdaGrad asks

> Should every parameter learn at the same speed?

Every new optimizer is born because engineers discovered a limitation in the previous one.

---

# Key Takeaways

- Momentum remembers previous movement.
- AdaGrad remembers previous gradients.
- Momentum accelerates optimization.
- AdaGrad adapts learning rates.
- Different parameters should not always learn at the same speed.
- Adaptive learning rates allow each parameter to learn according to its own optimization history.

---

# Think Like a Researcher

Instead of accepting a fixed learning rate, researchers asked:

> Can the optimizer decide how much each parameter should learn?

That single question led to AdaGrad.

This is an important engineering lesson.

Every breakthrough optimizer begins with a simple question about the limitations of the previous one.

---

# Coming Next

We now understand the intuition.

The next question is:

> **How should we mathematically use the history of gradients to calculate the learning rate?**

Should we store:

- Sum of gradients?
- Average of gradients?
- Absolute gradients?
- Squared gradients?
- Maximum gradient?

We'll derive the mathematics from first principles in the next section instead of memorizing the final equation.