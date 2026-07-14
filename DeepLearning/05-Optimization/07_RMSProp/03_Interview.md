# RMSProp — Engineer Discussion & Intuitive Q&A

---

# Q1 — Why Was AdaGrad Not Sufficient?

AdaGrad was a breakthrough because it introduced adaptive learning rates.

Initially, training becomes much more stable.

However, after several thousand (or even millions of) iterations, the model almost stops learning.

Why does this happen?

### Answer

AdaGrad accumulates the squared gradients throughout the entire training process.

```text
G = G + g²
```

Since squared gradients are always non-negative,

```text
G
```

can only increase.

It never decreases or forgets past information.

As a result,

the effective learning rate

```text
η_eff = η / √G
```

keeps shrinking.

Eventually, the learning rate becomes so small that weight updates become almost zero.

The optimizer has not stopped because it found the optimum.

It has stopped because it remembers **every historical gradient equally**, making it overly conservative.

---

# Q2 — Should All Historical Gradients Be Treated Equally?

Suppose a parameter receives:

```text
Iterations 1–1000

Large Gradients
```

and later

```text
Iterations 1001–50000

Very Small Gradients
```

Should gradients from the first thousand iterations influence today's learning rate just as much as the most recent gradients?

### Answer

Probably not.

Those early gradients have already influenced the parameter through previous weight updates.

What matters more is the parameter's recent behaviour.

Continuing to penalize the parameter because of very old gradients causes the learning rate to shrink unnecessarily.

Therefore, recent gradients should receive greater importance while older gradients should gradually lose influence.

---

# Q3 — Designing the Next Optimizer

Suppose you are asked to design the next optimizer after AdaGrad.

Without knowing RMSProp, list the properties your optimizer should satisfy.

### Answer

A better optimizer should:

* Not treat every historical gradient equally.
* Give greater importance to recent gradients.
* Gradually reduce the influence of older gradients.
* Never discard historical information abruptly.
* Maintain constant memory regardless of training duration.

These requirements naturally motivate a running weighted average instead of accumulating every gradient forever.

---

# Q4 — Why Can't We Store Every Historical Gradient?

Suppose someone proposes the following algorithm:

```text
Store every gradient forever.

Whenever a learning rate is required,

compute a weighted average over the complete history.
```

Would this be a practical engineering solution?

### Answer

No.

Modern neural networks may contain billions of parameters and train for millions of iterations.

Storing every historical gradient would require enormous memory and computational resources.

Such an approach quickly becomes impractical.

Instead, RMSProp stores only a single running estimate of historical gradients.

This provides the required historical information while using constant memory.

---

# Q5 — Deriving a Fading Memory

Suppose you maintain one variable called:

```text
Memory
```

A new squared gradient arrives after every iteration.

How should Memory be updated so that:

* recent gradients receive greater importance,
* older gradients gradually lose influence,
* historical information is never discarded immediately?

### Answer

Keeping only the new gradient completely discards history.

Keeping the complete sum recreates AdaGrad.

A better solution is to combine both old and new information.

```text
Memory = β × Memory + (1 − β) × g²
```

The coefficients control how much historical information is retained and how much new information is incorporated.

This naturally leads to the Exponential Moving Average (EMA).

---

# Q6 — What Does β Control?

Without using the RMSProp equation, explain the effect of different β values.

### Answer

**β = 0**

The optimizer completely ignores historical information.

Memory depends only on the current gradient.

---

**β = 0.5**

Old and new information receive equal importance.

Historical information fades relatively quickly.

---

**β = 0.9**

Most historical information is retained while allowing recent gradients to gradually influence future learning.

This provides a good balance between stability and adaptability.

---

**β = 0.99**

Historical information dominates.

The optimizer adapts more slowly because recent gradients contribute only a small fraction to the running estimate.

---

# Q7 — Momentum vs RMSProp

Both Momentum and RMSProp remember information from previous iterations.

Why do we still need both optimizers?

### Answer

Although both use an Exponential Moving Average, they remember different quantities.

**Momentum** remembers previous movement (velocity).

Its goal is to reduce oscillations and accelerate optimization by maintaining consistent movement toward the minimum.

**RMSProp** remembers the recent history of squared gradients.

Its goal is to adapt the learning rate for each parameter based on recent gradient magnitudes.

Therefore, the mathematical structure is similar, but the optimization objective is completely different.

---

# Q8 — Why Does RMSProp Still Use Squared Gradients?

When moving from AdaGrad to RMSProp, researchers changed how history is stored.

Why didn't they replace squared gradients with raw gradients?

### Answer

The objective is still to estimate the magnitude of historical gradients.

Using raw gradients would again suffer from positive and negative gradients cancelling each other.

Squared gradients eliminate this cancellation and correctly represent the historical magnitude of parameter updates.

Therefore, RMSProp keeps the squared gradients while improving only the memory mechanism.

---

# 🎯 Staff Engineer Challenge

You are building a recommendation system for a large e-commerce platform.

Customer behaviour changes continuously because of:

* festivals,
* flash sales,
* seasonal trends,
* product launches.

Initially, the model is trained using AdaGrad.

The model performs well during the first few months, but later becomes very slow to adapt to changing customer preferences.

### Discussion

AdaGrad assumes that every historical gradient is equally important.

As customer behaviour evolves, very old shopping patterns continue influencing the learning rate, causing adaptation to become increasingly slow.

RMSProp addresses this limitation by assigning greater importance to recent gradients while gradually reducing the influence of older information through Exponential Moving Average.

As a result, the optimizer adapts much more effectively to changing user behaviour.

However, RMSProp is not a complete solution.

If customer behaviour changes temporarily (for example, during a festival), RMSProp gradually adapts to that behaviour.

After the festival ends, customer behaviour shifts again.

The optimizer must once again adapt to the new distribution.

Therefore, the natural engineering question becomes:

> Can we combine long-term learning with rapid short-term adaptation?

This question motivates the development of even more advanced optimizers.

---

# 💡 Engineer's Insight

Notice how optimizer design has evolved.

```text
Gradient Descent
        │
        ▼
Fixed Learning Rate

        │
        ▼
Momentum
(Remembers Previous Movement)

        │
        ▼
AdaGrad
(Remembers Every Historical Gradient)

        │
        ▼
RMSProp
(Remembers Recent Gradient History)
```

Every optimizer exists because the previous optimizer exposed a limitation.

Optimization research is therefore a sequence of engineering improvements rather than unrelated algorithms.
