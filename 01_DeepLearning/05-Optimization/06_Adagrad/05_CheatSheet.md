# AdaGrad Cheat Sheet

## Problem

One global learning rate is not suitable for every parameter.

---

## Solution

Maintain a separate adaptive learning rate for every parameter.

---

## Core Idea

Frequently updated parameters

↓

Smaller learning rate

Rarely updated parameters

↓

Larger learning rate

---

## Formula

```text
G = G + g²

η_eff = η / √G

w = w − η_eff × g
```

---

## Strengths

- Adaptive learning rate
- Excellent for sparse data
- Per-parameter optimization

---

## Weakness

Gradient history never decreases.

Learning rate eventually becomes too small.

---

## Best Use Cases

- NLP
- Recommendation Systems
- Sparse Features

---

## Not Suitable For

Very long training runs

Large-scale LLM pretraining

---

## Interview One-Liner

AdaGrad adapts the learning rate of every parameter independently by accumulating squared gradients, allowing frequently updated parameters to slow down while rare parameters continue learning efficiently.