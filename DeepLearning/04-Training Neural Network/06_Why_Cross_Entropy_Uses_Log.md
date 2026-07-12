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

# 🧠 Engineer Discussion & Intuitive Q&A

## Q1 — Should a Highly Confident Correct Prediction Receive a Large Penalty?

Suppose the model predicts

```text
Cat    = 0.90

Dog    = 0.05

Horse  = 0.05
```

The correct class is

```text
Cat
```

Should Cross Entropy produce a small loss or a large loss?

### Answer

Cross Entropy should produce a **very small loss**.

A good classification loss function should satisfy three important properties:

- Give almost zero penalty when the correct class receives a very high probability.
- Increase gradually as the confidence in the correct class decreases.
- Increase very rapidly when the model is confidently wrong.

Since the model assigns a very high probability to the correct class, only a small penalty is required.

---

## Q2 — Why Should Lower Confidence Receive a Larger Penalty?

Now suppose the model predicts

```text
Cat    = 0.40

Dog    = 0.30

Horse  = 0.30
```

The correct class is still

```text
Cat
```

Why should this prediction receive a larger penalty than the previous one?

### Answer

Although the prediction is still correct, the model is much less confident.

A good classification loss should encourage the model to become increasingly confident about the correct class.

As confidence decreases, the penalty should gradually increase.

This guides the model toward making stronger and more reliable predictions.

---

## Q3 — Should Every Improvement Be Rewarded Equally?

Consider two situations.

### Case 1

```text
Correct Class Probability

0.90

↓

0.99
```

### Case 2

```text
Correct Class Probability

0.10

↓

0.19
```

Should both improvements receive the same reward?

### Answer

No.

The second improvement is much more valuable.

A model predicting only 10% probability for the correct class is making a severe mistake.

Increasing that confidence to 19% represents significant learning.

On the other hand, improving from 90% to 99% only provides a small refinement because the prediction was already very good.

Cross Entropy naturally focuses learning on poor predictions while spending very little effort on predictions that are already nearly correct.

---

## Q4 — Why Doesn't Cross Entropy Use a Simple Loss Like `1 - p`?

Suppose we define the loss as

```text
Loss = 1 - p
```

instead of

```text
Loss = -log(p)
```

Wouldn't this also become zero when the prediction is perfect?

### Answer

Although `1 - p` satisfies some desirable properties, it fails to strongly punish confident mistakes.

Consider two predictions.

```text
Correct Class Probability = 10%

Loss = 0.90
```

and

```text
Correct Class Probability = 1%

Loss = 0.99
```

These two predictions are dramatically different.

A model assigning only **1% probability** to the correct class is making a catastrophically confident mistake.

Yet the loss barely changes.

The logarithm behaves very differently.

```text
Probability      -log(p)

1.0              0

0.9              0.105

0.5              0.693

0.1              2.303

0.01             4.605

0.001            6.908
```

As the probability approaches zero, the loss grows extremely rapidly.

This is exactly the behavior we want from a classification loss function.

---

## Q5 — What Happens When the Correct Class Receives Almost Zero Probability?

Suppose the model predicts

```text
Correct Class Probability

0.000001
```

Should the loss increase only slightly?

### Answer

No.

The model is almost completely certain that the correct answer is wrong.

This is a catastrophic prediction.

Cross Entropy therefore assigns an extremely large penalty.

Large mistakes should receive disproportionately larger penalties than small mistakes.

---

## Q6 — What Happens if the Model Predicts Zero Probability for the Correct Class?

Suppose

```text
Correct Class Probability = 0
```

### Answer

Cross Entropy becomes

```text
Loss = -log(0)
```

Mathematically,

```text
log(0) → -∞
```

Therefore,

```text
Loss → +∞
```

This is actually desirable.

If the model is absolutely certain that the correct answer is impossible, it has made the worst possible prediction.

Such confident mistakes deserve the largest possible penalty.

In practice, modern deep learning libraries avoid numerical instability by using numerically stable implementations of Cross Entropy.

---

## Q7 — How Does Cross Entropy Guide Learning?

Suppose two models predict

Model A

```text
Correct Class Probability = 0.95
```

Model B

```text
Correct Class Probability = 0.05
```

Which model should receive the larger gradient update?

### Answer

Model B should receive the much larger update.

The logarithm naturally creates this behavior.

```text
Poor Prediction

↓

Large Loss

↓

Large Gradient

↓

Large Weight Update
```

Whereas

```text
Good Prediction

↓

Small Loss

↓

Small Gradient

↓

Small Weight Update
```

This is exactly what we mean when we say that a loss function should **guide learning**.

Cross Entropy automatically focuses learning on predictions that need the most improvement.

---

## Q8 — Why Don't We Use Cross Entropy for Regression?

Suppose you are building

```text
House Price Prediction
```

Should you use Cross Entropy?

### Answer

No.

Regression predicts continuous numerical values.

For example,

```text
Predicted Price

₹75,00,000
```

The numerical prediction itself is the output we care about.

Assigning a confidence like

```text
92%
```

does not provide meaningful information for the prediction.

Classification is different.

For example,

```text
Cat = 92%
```

Here, the probability itself is the prediction.

Therefore:

- Regression uses numerical losses such as MSE or MAE.
- Classification uses probability-based losses such as Cross Entropy.

---

## 💡 Engineer's Insight

Cross Entropy does much more than measure prediction error.

It automatically decides **where the model should spend its learning effort.**

Predictions that are already highly confident receive only tiny updates.

Predictions that are confidently wrong receive large updates.

This allows the optimizer to focus its computational effort on correcting the model's biggest mistakes instead of making tiny improvements to predictions that are already nearly perfect.

---

# 🎯 Staff Engineer Challenge

You are leading the AI team at a hospital.

The model predicts whether a patient has a rare but dangerous disease.

The disease occurs in only

```text
0.1%
```

of patients.

A junior engineer says:

> "The model predicts 55% probability for positive cases, so that's good enough."

Should you accept this reasoning?

### Discussion

No.

Softmax only tells us **what probability the model assigns** to each class.

It does not tell us whether that probability is actually good or bad.

Cross Entropy compares the predicted probability with the ground truth and converts that comparison into a learning signal.

A prediction of

```text
55%
```

may have completely different implications depending on the application.

For example:

### Spam Detection

```text
Spam = 55%
```

The email can simply be moved to the spam folder.

The cost of being wrong is relatively small.

### Disease Diagnosis

```text
Disease = 55%
```

This prediction may lead to additional medical tests, treatment decisions, or emergency care.

The business and safety implications are entirely different.

Probability alone is therefore insufficient for evaluating model quality.

The model also needs a loss function that measures how good or bad those probabilities are and provides meaningful gradients for learning.

---

## 💡 Staff Engineer Insight

Softmax answers one question:

> **"What does the model believe?"**

Cross Entropy answers a different question:

> **"How wrong is that belief?"**

Only after converting probabilities into a loss can the optimizer compute gradients and update the model's weights.

The complete learning pipeline is therefore

```text
Input

↓

Neural Network

↓

Logits

↓

Softmax

↓

Probabilities

↓

Cross Entropy Loss

↓

Gradients

↓

Weight Update
```

Softmax produces probabilities.

Cross Entropy transforms those probabilities into a learning signal.

Without Cross Entropy (or another suitable loss function), the neural network would have no way to learn from its predictions.

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