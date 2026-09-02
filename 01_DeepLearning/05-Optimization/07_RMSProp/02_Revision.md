# RMSProp — Revision Notes

---

# 1. Problem Statement

AdaGrad introduced **adaptive learning rates**.

Each parameter receives its own learning rate based on the history of gradients it has received.

```text
Large Historical Gradient
        ↓
Smaller Learning Rate

Small Historical Gradient
        ↓
Larger Learning Rate
```

Initially,

this works very well.

However,

AdaGrad has one major limitation.

---

# 2. Why Does AdaGrad Fail?

AdaGrad accumulates squared gradients forever.

```text
G = G + g²
```

Since squared gradients are always non-negative,

```text
G
```

can only increase.

It never decreases.

As a result,

the effective learning rate

```text
η_eff = η / √G
```

keeps becoming smaller.

After many iterations,

the learning rate becomes almost zero,

and the optimizer nearly stops learning.

---

# 3. Core Intuition

Ask yourself:

> **Should a parameter be punished forever because it received large gradients thousands of iterations ago?**

Probably not.

Older gradients have already influenced the parameter.

Recent gradients provide a better picture of the parameter's current behaviour.

Therefore,

an optimizer should

* remember recent gradients,
* gradually forget older gradients,
* continue adapting throughout training.

---

# 4. Engineering Analogy

Imagine a Grade 10 student.

Which classes best represent the student's current knowledge?

```text
Grade 8

Grade 9

Grade 10
```

These recent classes matter more than Grade 1.

Older learning is not discarded,

but its influence becomes much smaller.

RMSProp follows the same philosophy.

---

# 5. Exponential Moving Average (EMA)

Instead of remembering every gradient equally,

RMSProp maintains a running estimate of recent squared gradients.

```text
EMA = β × EMA + (1 − β) × g²
```

Where

* **EMA** = Running estimate of squared gradients
* **β** = Memory coefficient
* **g²** = Current squared gradient

Recent gradients receive greater importance,

while older gradients gradually fade.

---

# 6. Effective Learning Rate

Once EMA is computed,

the adaptive learning rate becomes

```text
η_eff = η / √(EMA + ε)
```

where

```text
ε
```

is a very small constant added to avoid division by zero.

---

# 7. Weight Update

The weight update rule becomes

```text
W(new) = W(old) − η_eff × Gradient
```

or

```text
W(new) = W(old) − [η / √(EMA + ε)] × Gradient
```

Unlike AdaGrad,

the denominator no longer grows forever.

---

# 8. Advantages

* Learning rate remains adaptive throughout training.
* Older gradients gradually lose importance.
* Recent gradients influence optimization more strongly.
* Suitable for long training runs.
* Prevents learning rate from shrinking toward zero.

---

# 9. Limitation of AdaGrad vs RMSProp

### AdaGrad

```text
Remembers everything forever.
```

Problem:

```text
Learning rate eventually becomes extremely small.
```

---

### RMSProp

```text
Remembers recent history.
```

Result:

```text
Learning rate continues adapting throughout training.
```

---

# 10. Engineering Evolution

```text
Gradient Descent
        │
        ▼
No Memory

        │
        ▼
Momentum
(Remembers previous movement)

        │
        ▼
AdaGrad
(Remembers all historical gradients)

        │
        ▼
RMSProp
(Remembers recent gradient history)
```

Each optimizer fixes one limitation of the previous optimizer.

---

# 11. Interview Traps

### Trap 1

**Does RMSProp discard old gradients completely?**

No.

It gradually reduces their influence using Exponential Moving Average.

---

### Trap 2

**Does RMSProp store every historical gradient?**

No.

It stores only one running value (EMA),

making it memory efficient.

---

### Trap 3

**Does RMSProp solve the same problem as Momentum?**

No.

Momentum remembers previous **movement (velocity)**.

RMSProp remembers recent **squared gradients**.

---

### Trap 4

**Why not simply reset AdaGrad's accumulated gradients periodically?**

Resetting completely throws away useful historical information.

RMSProp provides a smoother solution by gradually forgetting older gradients instead of discarding them abruptly.

---

# 12. One-Line Summary

> **RMSProp solves AdaGrad's diminishing learning rate problem by replacing permanent memory with a fading memory using an Exponential Moving Average of squared gradients.**
