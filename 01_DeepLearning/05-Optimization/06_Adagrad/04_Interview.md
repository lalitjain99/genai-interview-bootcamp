# AdaGrad — Engineer Interview Questions

---

# Q1. Momentum already made optimization smoother and faster. Why did researchers still feel the need to invent AdaGrad?

### Expected Discussion

Momentum solves one important problem—it remembers previous movement and reduces oscillations.

However, every parameter in the network still shares the same global learning rate.

In real neural networks, different parameters behave very differently.

Some parameters receive large gradients repeatedly because they are updated frequently.

Others receive very small or infrequent gradients.

Using the same learning rate for both is not ideal.

Frequently updated parameters may continue making aggressive updates, while rarely updated parameters may learn too slowly.

This led researchers to ask a new question:

> **Can every parameter automatically choose its own learning rate based on its training history?**

AdaGrad was the first optimizer to answer this question.

---

# Q2. Why can't AdaGrad simply store the average gradient instead of squared gradients?

### Expected Discussion

The purpose of AdaGrad is to adjust the learning rate based on the historical behaviour of each parameter.

Suppose a parameter receives the following gradients:

```text
10
-10
10
-10
```

The average gradient becomes:

```text
(10 - 10 + 10 - 10) / 4 = 0
```

The optimizer would incorrectly conclude that this parameter has barely received any gradients.

In reality, the parameter has experienced large gradients during every iteration.

The average becomes zero only because positive and negative gradients cancel each other.

As a result, AdaGrad would assign a larger learning rate than appropriate.

Therefore, the average gradient is not a reliable representation of historical gradient magnitude.

---

# Q3. Why does AdaGrad use squared gradients instead of absolute gradients?

### Expected Discussion

Both absolute gradients and squared gradients eliminate the problem of positive and negative gradient cancellation.

However, AdaGrad wants parameters with consistently large gradients to reduce their learning rate more aggressively.

Consider two parameters.

Parameter A:

```text
5
5
5
5
```

Absolute history:

```text
20
```

Squared history:

```text
100
```

Parameter B:

```text
5
5
5
100
```

Absolute history:

```text
115
```

Squared history:

```text
10075
```

Notice that squaring amplifies large gradients much more than absolute values.

This makes AdaGrad much more sensitive to parameters receiving consistently large updates.

As a result, those parameters automatically receive smaller future learning rates.

---

# Q4. Why does AdaGrad store the entire history of squared gradients instead of only the last few gradients?

### Expected Discussion

AdaGrad is built on the assumption that the complete training history of a parameter is valuable.

If a parameter has been receiving large gradients for thousands of iterations, the optimizer should remember this behaviour.

By accumulating all past squared gradients,

the effective learning rate gradually decreases for parameters that have consistently received large updates.

This creates stable and adaptive learning.

However, this design also introduces AdaGrad's biggest limitation.

Since the accumulated history only increases,

the effective learning rate continues decreasing throughout training.

Eventually, learning can become extremely slow.

This limitation later motivated the development of RMSProp.

---

# Q5. Why doesn't AdaGrad simply reduce the global learning rate instead of maintaining one learning rate per parameter?

### Expected Discussion

Reducing the global learning rate affects every parameter equally.

However, not all parameters behave in the same way.

Some parameters receive frequent and large updates,

while others receive very small or infrequent updates.

Applying the same learning rate to every parameter ignores these differences.

AdaGrad maintains an independent learning rate for every parameter,

allowing each one to adapt according to its own optimization history.

---

# Q6. Why is AdaGrad particularly useful for sparse datasets?

### Expected Discussion

In sparse datasets,

many parameters receive updates only occasionally.

Since AdaGrad accumulates gradient history independently for every parameter,

rarely updated parameters accumulate a much smaller history.

Their effective learning rate therefore remains relatively larger.

Frequently updated parameters,

on the other hand,

accumulate much larger histories,

causing their learning rates to decrease automatically.

This allows AdaGrad to learn effectively from both common and rare features.

---

# Q7. Why does AdaGrad eventually stop learning during very long training runs?

### Expected Discussion

AdaGrad continuously accumulates squared gradients.

```text
G = G + g²
```

Since the accumulated history never decreases,

the denominator in the adaptive learning rate

```text
η / √G
```

keeps growing.

As training progresses,

the effective learning rate becomes smaller and smaller.

Eventually,

weight updates become almost negligible,

causing learning to slow down significantly or even stop.

This is AdaGrad's most important practical limitation.

---

# Q8. Explain AdaGrad to a non-technical manager in one minute.

### Expected Discussion

Imagine every employee in a company is learning a new skill.

Employees who make the same mistake repeatedly should become more careful over time.

Employees who rarely make mistakes should continue learning confidently.

AdaGrad applies the same philosophy to neural networks.

Instead of giving every parameter the same learning speed,

it allows each parameter to automatically adjust its learning rate based on its own learning history.

This leads to more intelligent and adaptive optimization.