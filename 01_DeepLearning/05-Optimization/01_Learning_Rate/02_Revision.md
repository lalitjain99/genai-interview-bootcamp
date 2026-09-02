# Learning Rate — Revision Notes

---

# 1. Problem Statement

The gradient tells us **which direction** increases the loss the fastest.

To minimize the loss, we move in the opposite direction.

However,

the gradient alone does **not** tell us how far we should move.

If we blindly follow the gradient,

the weight update may be too small or too large.

This motivates the introduction of the **Learning Rate (η)**.

---

# 2. Core Intuition

Gradient answers:

> **Which direction should I move?**

Learning Rate answers:

> **How big should the step be?**

Both are equally important.

Without the gradient,

we don't know where to move.

Without the learning rate,

we don't know how far to move.

---

# 3. Weight Update Equation

Gradient Descent updates weights as:

```text
W(new) = W(old) − η × ∇L
```

Where

- **η** = Learning Rate
- **∇L** = Gradient of the Loss Function

The gradient determines the direction.

The learning rate scales the step size.

---

# 4. Effect of Different Learning Rates

### Very Small Learning Rate

```text
η = 0.0001
```

Characteristics:

- Tiny weight updates
- Very slow learning
- Requires many iterations
- May take an impractical amount of time to converge

---

### Very Large Learning Rate

```text
η = 100
```

Characteristics:

- Extremely large updates
- Overshoots the minimum
- Loss may oscillate or even increase
- Training may diverge

---

### Well-Chosen Learning Rate

Characteristics:

- Stable optimization
- Faster convergence
- Smooth reduction in loss
- Efficient training

---

# 5. Engineering Intuition

Imagine driving using GPS.

- **GPS** tells you **which direction** to travel.
- **Speed** determines **how quickly** you reach the destination.

Gradient = GPS

Learning Rate = Speed

Neither can replace the other.

---

# 6. Why Not Use Only the Gradient?

The gradient already contains magnitude,

but blindly trusting that magnitude is risky.

Large gradients could produce excessively large updates.

Small gradients could make learning painfully slow.

The learning rate acts as a scaling factor,

allowing us to control the step size independently of the gradient.

---

# 7. Why Can't We Keep Increasing the Learning Rate?

A learning rate that works for one iteration may become inappropriate later.

Example:

Early in training:

```text
Gradient = 0.0001
```

A large learning rate may still produce a reasonable update.

Later:

```text
Gradient = 100
```

The same learning rate could produce an enormous update,

causing the optimizer to overshoot the minimum.

Therefore,

a single fixed learning rate is a compromise rather than a perfect solution.

---

# 8. Advantages

- Controls optimization speed
- Prevents excessively large updates
- Improves training stability
- Helps achieve convergence

---

# 9. Limitations

- Choosing the correct value is difficult
- Too small → Slow convergence
- Too large → Divergence or oscillation
- One global learning rate is not ideal for every parameter

This limitation later motivates adaptive optimizers such as AdaGrad.

---

# 10. Engineering Takeaways

Always separate these two questions:

**Where should I move?**

→ Gradient

**How far should I move?**

→ Learning Rate

A good optimizer must answer both.

---

# 11. Interview Traps

### Trap 1

**Does the gradient already contain step size?**

Yes.

But we intentionally scale it using the learning rate to obtain stable optimization.

---

### Trap 2

**Can we remove the learning rate and trust the gradient magnitude?**

No.

Gradient magnitude changes throughout training.

Without scaling,

updates may become unstable.

---

### Trap 3

**Can a learning rate that works today still work later?**

Not necessarily.

As optimization progresses,

gradient behaviour changes,

making a fixed learning rate a compromise.

---

# 12. One-Line Summary

> **The gradient decides the direction of optimization, while the learning rate decides the size of the step. Effective optimization requires both.**