# RMSProp — Cheat Sheet

---

# One-Line Definition

> **RMSProp solves AdaGrad's diminishing learning rate problem by replacing permanent gradient history with a fading memory using an Exponential Moving Average (EMA).**

---

# Why Was RMSProp Invented?

AdaGrad Problem

```text
G = G + g²
```

↓

G grows forever

↓

Learning rate keeps shrinking

↓

Weight updates become almost zero

↓

Training slows dramatically

---

# Core Idea

Instead of remembering **all** gradients,

remember **recent** gradients.

```text
Old History
      ↓
Gradually Fade

Recent History
      ↓
Higher Importance
```

---

# Running Memory (EMA)

```text
EMA = β × EMA + (1 − β) × g²
```

where

```text
EMA
```

stores recent squared gradient history.

---

# Effective Learning Rate

```text
η_eff = η / √(EMA + ε)
```

---

# Weight Update

```text
W = W − η_eff × Gradient
```

---

# Role of β

| β    | Behaviour                                   |
| ---- | ------------------------------------------- |
| 0    | No historical memory                        |
| 0.5  | History fades quickly                       |
| 0.9  | Good balance (commonly used)                |
| 0.99 | Strong historical memory, slower adaptation |

---

# RMSProp vs AdaGrad

| AdaGrad                       | RMSProp                                  |
| ----------------------------- | ---------------------------------------- |
| Remembers all gradients       | Remembers recent gradients               |
| Permanent memory              | Fading memory                            |
| Learning rate keeps shrinking | Learning rate continues adapting         |
| Good for sparse updates       | Good for long training and changing data |

---

# Momentum vs RMSProp

| Momentum                 | RMSProp                     |
| ------------------------ | --------------------------- |
| Remembers velocity       | Remembers squared gradients |
| Reduces oscillation      | Adapts learning rate        |
| Accelerates optimization | Controls step size          |

---

# Typical Use Cases

✅ Long training runs

✅ Non-stationary data

✅ Continuously changing environments

✅ Deep neural networks

---

# Engineering Evolution

```text
Gradient Descent
        ↓
Momentum
(Remember movement)

        ↓
AdaGrad
(Remember all gradients)

        ↓
RMSProp
(Remember recent gradients)
```

---

# Interview Keywords

* Adaptive Learning Rate
* Exponential Moving Average (EMA)
* Fading Memory
* Recent Gradient History
* Constant Memory
* Stability vs Adaptability

---

# One-Line Summary

> **AdaGrad remembers forever. RMSProp remembers what matters most—the recent past.**
