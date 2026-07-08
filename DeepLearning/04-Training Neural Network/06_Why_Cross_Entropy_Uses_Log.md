# Why Does Cross Entropy Use Logarithm?

## Module 04 — Training Neural Networks

## Learning Objectives

By the end of this lecture, you should be able to answer:

- Why isn't the loss simply `1 - p`?
- Why do we need a special penalty function?
- What properties should a good classification loss have?
- Why is the logarithm the perfect choice?
- Why is the formula `-log(p)`?
- How does the logarithm punish confident mistakes?
- Why does GPT use this exact mathematical function?

---

# Chapter 1 — Designing Our Own Loss Function

Imagine we have never heard of Cross Entropy.

We are engineers trying to design a loss function from scratch.

Suppose the correct answer is

```text
Cat
```

After applying Softmax, the model predicts:

```text
Cat = p
```

where **p** is the probability assigned to the correct class.

Now ask yourself:

> **What should the penalty be for different values of p?**

Let's build a table.

| Probability of Correct Class | Desired Penalty |
|-----------------------------|-----------------|
| 100% | Almost Zero |
| 90% | Very Small |
| 70% | Moderate |
| 50% | Large |
| 10% | Very Large |
| 1% | Extremely Large |

Without writing any mathematics, this already tells us what kind of function we need.

---

# Chapter 2 — First Attempt: 1 - p

A simple idea is

```text
Loss = 1 - p
```

Let's test it.

| Probability | Loss |
|-------------|------|
| 100% | 0 |
| 90% | 0.1 |
| 50% | 0.5 |
| 10% | 0.9 |
| 1% | 0.99 |

At first glance, this looks reasonable.

Higher confidence means lower loss.

But something is missing.

Compare these two predictions.

```text
Correct Class = 10%
```

and

```text
Correct Class = 1%
```

The losses are

```text
0.90

and

0.99
```

These two predictions are **very different**.

A model assigning only **1%** probability to the correct answer is making a catastrophically confident mistake.

Yet the penalty barely increases.

Our loss function isn't punishing confident mistakes strongly enough.

---

# Chapter 3 — What We Really Want

Let's write down our requirements.

A good loss function should:

✅ Give almost zero penalty when the prediction is almost certain.

✅ Increase gradually as confidence decreases.

✅ Increase very rapidly when the correct class receives almost no probability.

Notice that the last requirement is the most important.

If the model says

```text
Correct Answer = 0.0001%
```

we want the penalty to become enormous.

Otherwise, the model won't learn aggressively from disastrous mistakes.

---

# Chapter 4 — Enter the Logarithm

Mathematicians already know a function with exactly these properties.

The logarithm.

Let's look at a few values.

| Probability (p) | log(p) |
|-----------------|---------|
| 1.0 | 0 |
| 0.9 | -0.105 |
| 0.5 | -0.693 |
| 0.1 | -2.303 |
| 0.01 | -4.605 |
| 0.001 | -6.908 |

Notice something remarkable.

As the probability approaches zero,

the logarithm drops extremely fast.

Exactly what we wanted.

---

# Chapter 5 — Why the Negative Sign?

There is one small problem.

The logarithm of numbers between 0 and 1 is always negative.

For example,

```text
log(0.9)

=

-0.105
```

Loss values should be positive.

So we simply multiply by **-1**.

This gives us

```text
Loss = -log(p)
```

Now the values become:

| Probability | Loss |
|-------------|------|
| 1.0 | 0.000 |
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| 0.01 | 4.605 |
| 0.001 | 6.908 |

Perfect.

---

# Chapter 6 — Why This Is Beautiful

Notice what happened.

Suppose the correct class receives

```text
99%
```

Loss

```text
≈ 0
```

Very little learning is needed.

---

Now suppose the correct class receives

```text
50%
```

Loss

```text
≈ 0.69
```

The model should improve.

---

Now suppose the correct class receives

```text
1%
```

Loss

```text
≈ 4.6
```

Huge penalty.

The model receives a very strong learning signal.

Exactly what we wanted.

---

# Chapter 7 — Visualizing the Curve

Imagine plotting the loss.

```text
Loss

^

|

|                             *

|                        *

|                  *

|             *

|         *

|     *

| *

+------------------------------------>

          Probability
```

As the probability increases,

the loss rapidly approaches zero.

As the probability decreases,

the penalty rises sharply.

This shape is exactly what makes Cross Entropy so effective.

---

# Chapter 8 — Connecting Back to Softmax

Remember our complete classification pipeline.

```text
Neural Network

↓

Logits

↓

Softmax

↓

Probability Distribution

↓

Cross Entropy

↓

Loss
```

Softmax answers:

> "How confident am I?"

Cross Entropy replies:

> "Based on that confidence, here's how much you should be penalized."

Together, they form one of the most successful combinations in Deep Learning.

---

# Chapter 9 — GPT Connection

Suppose GPT predicts the next token.

Correct token:

```text
Paris
```

Prediction:

```text
Paris = 0.98
```

Loss

```text
-log(0.98)

≈ 0.02
```

Almost perfect.

---

Now suppose GPT predicts

```text
Paris = 0.01
```

Loss

```text
-log(0.01)

≈ 4.61
```

The loss becomes more than **200 times larger**.

This strong penalty tells the optimizer,

> "You were extremely confident, but completely wrong."

Backpropagation then adjusts billions of weights so that future predictions become more accurate.

---

# Mental Model

```text
Correct Class Probability

↓

Higher Probability

↓

Smaller Loss

↓

Lower Probability

↓

Larger Loss

↓

-log(p)
```

---

# Key Takeaways

✅ Cross Entropy measures the probability assigned to the correct class.

✅ We want confident correct predictions to receive very small penalties.

✅ We want confident wrong predictions to receive very large penalties.

✅ The logarithm naturally provides this behavior.

✅ The negative sign ensures the loss remains positive.

---

# Looking Ahead

We've now completed the story of **Loss Functions**.

One question still remains.

> **The loss tells us how wrong the model is... but how does the model actually reduce that loss?**

How does it know **which weights** should increase?

Which weights should decrease?

How much should they change?

That question leads us to one of the greatest ideas in Machine Learning:

**Gradient Descent.**