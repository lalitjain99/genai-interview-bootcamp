# AdaGrad — Assignment (With Solutions)

---

# Assignment 1: Which Parameter Should Slow Down?

A neural network has two parameters.

Parameter A receives the following gradients:

```text
5, 6, 5, 7, 6
```

Parameter B receives:

```text
1, 0.5, 1, 0.8, 0.6
```

Without performing any mathematical calculations,

answer the following:

1. Which parameter has been receiving consistently larger updates?
2. Which parameter should receive the smaller learning rate?
3. Explain your reasoning using AdaGrad's design philosophy.

---

## Solution

Parameter **A** has consistently received much larger gradients.

According to AdaGrad, every parameter maintains its own gradient history.

Since Parameter A has historically received larger updates, its accumulated squared gradient history will grow much faster.

As a result, its effective learning rate will automatically decrease.

Parameter B, having much smaller gradients, accumulates history more slowly and therefore retains a relatively larger learning rate.

This is the central philosophy of AdaGrad:

> Frequently updated parameters become more cautious, while infrequently updated parameters continue learning aggressively.

### 💡 Revision Note

AdaGrad does **not** reduce the learning rate based on the **current gradient**.

It reduces the learning rate based on the **accumulated history of squared gradients**.

---

# Assignment 2: Predict the Learning Behaviour

Suppose a parameter receives the following gradient sequence:

```text
50, 45, 40, 38, 35, 30...
```

Answer the following:

1. What happens to the accumulated gradient history?
2. What happens to the effective learning rate?
3. How will future weight updates change?
4. Is this behaviour desirable? Why?

---

## Solution

The accumulated squared gradient history continuously increases because every iteration adds another squared gradient.

```text
G = G + g²
```

As **G** increases,

the adaptive learning rate

```text
η / √G
```

gradually decreases.

Consequently,

future weight updates become smaller and more conservative.

This behaviour is desirable because, as optimization progresses, we generally want smaller and more precise updates near the minimum.

However,

if training continues for a very long time,

the accumulated history becomes excessively large,

causing the learning rate to become extremely small.

Eventually,

learning may almost stop.

### 💡 Revision Note

The accumulated history **does not grow quadratically with the number of iterations.**

Each individual gradient contributes a **squared value**,

but the cumulative history grows approximately **linearly with training iterations**.

For example,

```text
10² + 10² + 10² + 10²

=

400
```

The quadratic operation applies to each gradient,

not to the total number of iterations.

---

# Assignment 3: Sparse Feature Learning

Imagine you are building a recommendation system.

Some products are purchased millions of times every day.

Some niche products are purchased only once every few weeks.

Explain:

1. Why would SGD struggle?
2. How does AdaGrad naturally solve this?
3. Which products continue learning faster?

---

## Solution

Frequently purchased products correspond to parameters that receive gradients very often.

Rare products correspond to parameters that receive gradients infrequently.

Under SGD,

every parameter shares the same learning rate.

Frequently updated parameters dominate the optimization process,

while rarely updated parameters may learn very slowly.

AdaGrad assigns every parameter its own adaptive learning rate.

Parameters associated with frequent products accumulate much larger gradient histories,

causing their learning rates to decrease automatically.

Rarely updated parameters accumulate much smaller histories,

allowing them to retain relatively larger learning rates.

This naturally balances learning between common and rare features.

### 💡 Revision Note

AdaGrad never knows anything about "products."

It only updates **parameters**.

Products influence training **indirectly** by changing how frequently different parameters receive gradients.

---

# Assignment 4: Debug the Optimizer

A team reports:

> "Our model trained well initially, but after several thousand iterations the weights barely change."

Answer:

1. Why is this happening?
2. Which AdaGrad property causes this?
3. Would increasing the initial learning rate permanently solve the issue?

---

## Solution

AdaGrad continuously accumulates squared gradients.

```text
G = G + g²
```

Since this accumulated history never decreases,

the denominator

```text
√G
```

continues growing throughout training.

As a result,

the effective learning rate

```text
η / √G
```

becomes progressively smaller.

Eventually,

weight updates become almost negligible,

even though optimization has not fully converged.

Increasing the initial learning rate is **not** a permanent solution.

Although training may start with larger updates,

the accumulated history continues growing,

and the effective learning rate will eventually become very small again.

### 💡 Revision Note

Changing the initial learning rate only changes the starting point.

It does **not** solve AdaGrad's fundamental problem because the accumulated gradient history grows forever.

---

# Assignment 5: Think Like an Optimizer Designer

One engineer asks:

> "Why should an optimizer remember every gradient from the beginning of training? Older gradients may no longer represent the current optimization state."

Discuss:

1. Do you agree?
2. Advantages of remembering the complete history.
3. Possible disadvantages.

---

## Solution

AdaGrad assumes that the complete optimization history is valuable.

If a parameter has consistently received large gradients throughout training,

its future learning rate should continue decreasing.

Remembering the complete history provides a stable estimate of how aggressively each parameter has been updated.

However,

this assumption may become problematic during very long training runs.

Early gradients may no longer represent the current optimization landscape.

As training evolves,

older gradients become less informative,

yet AdaGrad continues treating them as equally important.

Consequently,

the learning rate keeps shrinking,

even when the optimizer may need larger updates later in training.

This limitation naturally motivates the next optimizer: **RMSProp**, which asks an important engineering question:

> Should an optimizer remember **everything**, or should it remember **only recent history**?

### 💡 Revision Note

Your school report-card analogy is excellent.

A student who scored highly for many years but performed poorly recently illustrates the central design question:

Should decisions be based on **lifetime performance** or **recent performance**?

This is exactly the transition from AdaGrad to RMSProp.