# AdaGrad — Common Mistakes

This document captures the subtle misconceptions that occurred during learning.

These are not major conceptual errors but the kinds of mistakes interviewers often probe.

---

# Mistake 1: AdaGrad reacts to the current gradient

### Incorrect Thinking

AdaGrad reduces the learning rate whenever the current gradient is large.

### Correct Understanding

AdaGrad does **not** look at the current gradient alone.

It adjusts the learning rate using the **accumulated history of squared gradients**.

```text
G = G + g²

η_eff = η / √G
```

A single large gradient does not immediately produce a tiny learning rate.

The historical behaviour of the parameter determines the effective learning rate.

---

# Mistake 2: The accumulated gradient history grows quadratically

### Incorrect Thinking

Since gradients are squared,

the accumulated history grows quadratically with training iterations.

### Correct Understanding

Each gradient is squared **before** being added.

However,

the accumulated history grows approximately **linearly with the number of iterations**.

Example:

```text
10² + 10² + 10² + 10²

=

400
```

The quadratic operation applies to each gradient,

not to the number of iterations.

---

# Mistake 3: Increasing the initial learning rate fixes AdaGrad

### Incorrect Thinking

If the learning rate becomes too small,

simply increase the initial learning rate.

### Correct Understanding

Increasing η only changes the initial scale.

The accumulated gradient history continues growing forever.

Eventually,

```text
η / √G
```

becomes very small again.

Therefore,

changing the initial learning rate does not solve AdaGrad's fundamental limitation.

---

# Mistake 4: Thinking in terms of products instead of parameters

### Incorrect Thinking

Frequently purchased products receive larger learning rates.

### Correct Understanding

Optimizers know nothing about products,

images,

customers,

or words.

They only update **parameters**.

Business entities affect optimization **indirectly** because they change how often different parameters receive gradients.

Always map the business problem to parameter behaviour.

---

# Mistake 5: Confusing "current gradient" with "historical gradient"

### Incorrect Thinking

A parameter receives a large gradient,

therefore its learning rate immediately becomes small.

### Correct Understanding

AdaGrad accumulates information over many iterations.

One gradient rarely changes the learning rate significantly.

The adaptive behaviour emerges from **repeated historical updates**, not isolated gradients.

---

# Mistake 6: Forgetting the design philosophy

While solving problems,

always begin with the design question.

Ask yourself:

> **What property are we trying to achieve?**

For AdaGrad, the desired properties are:

- Frequently updated parameters should become more cautious.
- Rarely updated parameters should continue learning.
- Every parameter should have its own adaptive learning rate.
- The optimizer should learn from historical gradient behaviour.

Once these properties are clear,

the mathematical equations become much easier to understand.

---

# Interview Trap

**Question**

"If two parameters receive the same current gradient,

will AdaGrad always update them by the same amount?"

### Correct Answer

No.

Even if the current gradients are identical,

their accumulated gradient histories may be very different.

Since AdaGrad computes an independent effective learning rate for every parameter,

their final weight updates can still be different.

---

# Personal Learning Observation

During this lecture,

the biggest improvement was shifting from

> "What is the formula?"

to

> "What engineering problem is the formula solving?"

This is exactly how optimizer design should be studied.