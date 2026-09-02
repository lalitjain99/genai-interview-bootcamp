# RMSProp — Common Mistakes

---

# Mistake 1

### ❌ "RMSProp stores every previous gradient."

### Why it's wrong

RMSProp stores only **one running estimate (EMA)**.

It does **not** save every historical gradient.

### Correct Understanding

```text
One Memory Variable

↓

Updated Every Iteration
```

Memory usage remains constant regardless of training length.

---

# Mistake 2

### ❌ "RMSProp completely forgets old gradients."

### Why it's wrong

Old gradients are **not discarded immediately**.

Their influence decreases gradually through the Exponential Moving Average.

### Correct Understanding

```text
Recent Gradients

★★★★★

Older Gradients

★★☆☆☆

Very Old Gradients

★☆☆☆☆
```

Older information fades instead of disappearing instantly.

---

# Mistake 3

### ❌ "β = 0 makes RMSProp equivalent to Gradient Descent."

### Why it's wrong

When

```text
β = 0
```

the running average becomes

```text
EMA = g²
```

The optimizer still computes an adaptive learning rate.

It simply has **no historical memory**.

Gradient Descent uses a **fixed learning rate**.

---

# Mistake 4

### ❌ "Larger β means better optimization."

### Why it's wrong

A larger β increases historical memory.

This improves stability but slows adaptation.

There is always a trade-off.

---

# Mistake 5

### ❌ "Momentum and RMSProp solve the same problem."

### Why it's wrong

Although both use EMA,

they remember different quantities.

Momentum

```text
Velocity
```

RMSProp

```text
Squared Gradients
```

Different memory.

Different purpose.

---

# Mistake 6

### ❌ "RMSProp changes the gradient."

### Why it's wrong

The gradient is never modified.

RMSProp only changes the **effective learning rate**.

The gradient still determines the optimization direction.

---

# Mistake 7

### ❌ "RMSProp always outperforms AdaGrad."

### Why it's wrong

Optimizer selection depends on the problem.

AdaGrad often performs well for sparse features.

RMSProp performs well for long training and changing environments.

There is no universally best optimizer.

---

# Mistake 8

### ❌ "Using raw gradients instead of squared gradients would work equally well."

### Why it's wrong

Positive and negative gradients would cancel each other.

The optimizer would lose information about the true magnitude of historical updates.

Squaring eliminates this cancellation.

---

# Mistake 9

### ❌ "EMA is just another name for averaging."

### Why it's wrong

A simple average treats all observations equally.

EMA assigns exponentially larger importance to recent observations while gradually reducing the influence of older ones.

---

# Mistake 10

### ❌ "RMSProp solves every optimization problem."

### Why it's wrong

RMSProp improves AdaGrad,

but it still has limitations.

It motivates the next question:

> Can we combine adaptive learning rates with Momentum?

That question naturally leads to **Adam**.

---

# Final Interview Reminder

Whenever someone asks:

**"Why was RMSProp invented?"**

Never start with the equation.

Start with the problem.

```text
AdaGrad

↓

Permanent Memory

↓

Learning Rate → 0

↓

Need Fading Memory

↓

RMSProp
```

Engineers design solutions by solving problems—not by memorizing formulas.
