# AdaGrad — Revision Notes

# 1. Why was AdaGrad invented?

Momentum solved oscillations and accelerated optimization.

However, every parameter still used the same global learning rate.

Different parameters learn differently.

AdaGrad introduced adaptive learning rates for every parameter.

---

# 2. Core Intuition

Parameters that receive large gradients repeatedly should become more cautious.

Parameters that rarely receive updates should continue learning aggressively.

---

# 3. Key Idea

Instead of one global learning rate,

AdaGrad assigns every parameter its own learning rate.

The learning rate depends on the cumulative history of squared gradients.

---

# 4. Mathematical Formulation

Gradient:

```text
g = ∂L / ∂w
```

Accumulate history:

```text
G = G + g²
```

Adaptive learning rate:

```text
η_eff = η / √G
```

Weight update:

```text
w = w − η_eff × g
```

---

# 5. Algorithm

Initialize

```text
G = 0
```

For every iteration

```text
Forward Pass

↓

Loss

↓

Backpropagation

↓

Compute Gradient

↓

Update G

↓

Compute η_eff

↓

Update Weights
```

---

# 6. Advantages

- Automatic learning rate adaptation
- Independent learning rate for every parameter
- Excellent for sparse datasets
- Reduces aggressive updates

---

# 7. Limitations

- Gradient history only grows
- Learning rate continuously decreases
- Eventually learning becomes extremely slow

---

# 8. Typical Interview Questions

- Why wasn't Momentum enough?
- Why squared gradients?
- Why not average gradients?
- Why per-parameter learning rate?
- Why does AdaGrad fail during long training?

---

# 9. Remember

AdaGrad solved one major problem.

But it introduced another.

**Learning rate never stops shrinking.**

This naturally leads to RMSProp.